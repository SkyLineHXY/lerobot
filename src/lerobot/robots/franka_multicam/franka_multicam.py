"""FrankaMultiCam：Franka 机械臂 + 多路 LeRobot Camera 组合机器人。

将 Franka 机械臂（EE 位姿 + 夹爪）与多路摄像头组合为一个 Robot 实例，
可直接用于 lerobot-record 数据采集或其他需要同时获取 arm state 与图像的场景。
"""

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..franka.config_franka import FrankaConfig
from ..franka.franka import DEFAULT_CARTESIAN_KX, DEFAULT_CARTESIAN_KXD, Franka
from ..robot import Robot
from .config_franka_multicam import FrankaMultiCamConfig

logger = logging.getLogger(__name__)


class FrankaMultiCam(Robot):
    """Franka 机械臂 + 多路相机组合机器人。

    内部持有 ``Franka`` 实例（复用 EE/gripper 逻辑）和
    ``dict[str, Camera]`` 实例（由 ``make_cameras_from_configs`` 创建）。

    ``get_observation`` 返回的字典同时包含：
    - ``ee_pose.{x,y,z,rx,ry,rz}`` — 末端执行器位姿
    - ``gripper.width`` — 夹爪宽度（若 use_gripper=True）
    - 每个相机的 key → (H, W, 3) uint8 numpy 数组
    """

    config_class = FrankaMultiCamConfig
    name = "franka_multicam"

    def __init__(self, config: FrankaMultiCamConfig):
        super().__init__(config)
        self.config = config

        # 构造内部 Franka 子配置
        franka_cfg = FrankaConfig(
            robot_ip=config.robot_ip,
            robot_port=config.robot_port,
            use_gripper=config.use_gripper,
            gripper_speed=config.gripper_speed,
            gripper_force=config.gripper_force,
        )
        self._franka = Franka(franka_cfg)

        # 创建相机实例
        self.cameras = make_cameras_from_configs(config.cameras)

    # ------------------------------------------------------------------
    # Feature descriptors
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = dict(self._franka.observation_features)
        for cam_key, ccfg in self.config.cameras.items():
            features[cam_key] = (ccfg.height, ccfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return dict(self._franka.action_features)

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._franka.is_connected and all(
            c.is_connected for c in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return self._franka.is_calibrated

    def calibrate(self) -> None:
        self._franka.calibrate()

    def configure(self) -> None:
        self._franka.configure()

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("正在连接 Franka...")
        self._franka.connect(calibrate=calibrate)

        logger.info("正在连接 %d 路相机...", len(self.cameras))
        for cam_key, cam in self.cameras.items():
            logger.info("  连接相机 '%s'...", cam_key)
            cam.connect()

        logger.info("%s 已连接", self)

    def disconnect(self) -> None:
        if not self.is_connected and not self._franka.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning("断开相机 '%s' 时异常: %s", cam_key, e)

        try:
            self._franka.disconnect()
        except Exception as e:
            logger.warning("断开 Franka 时异常: %s", e)

        logger.info("%s 已断开", self)

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        obs = self._franka.get_observation()

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return self._franka.send_action(action)

    # ------------------------------------------------------------------
    # Convenience (透传到 Franka)
    # ------------------------------------------------------------------

    def start_cartesian_impedance(
        self,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
        settle_time: float = 0.3,
    ) -> None:
        """启动 Cartesian impedance 控制器。"""
        self._franka.start_cartesian_impedance(Kx=Kx, Kxd=Kxd, settle_time=settle_time)

    def terminate_policy(self) -> None:
        """终止当前 polymetis 控制器。"""
        self._franka.terminate_policy()

    def reset(self) -> None:
        """机械臂回 home 位。"""
        self._franka.reset()
