import logging
import threading
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..franka.franka import EE_POSE_KEYS, Franka
from ..gen_gripper.umi_gripper import UmiGripper
from ..robot import Robot
from .config_franka_gen_gripper import FrankaGenGripperConfig
from .umi_transforms import (
    absolute_position_world_orientation_to_base,
    ee_relative_action_to_base,
    load_flange_to_camera_extrinsic,
    pose_xyz_rotvec_to_se3,
    se3_to_xyz_rotvec,
    umi_camera_t0_action_to_base,
)

logger = logging.getLogger(__name__)

# UMI ee6d → base reference modes accepted by ``send_umi_action``.
UMI_REFERENCE_MODES = ("camera_t0", "ee_at_t0", "abs_pos_world_rot")


class FrankaGenGripper(Robot):
    """Combined Franka arm + UMI Gen Controller gripper.

    This robot class composes a Franka 7-DoF arm (via zerorpc) with a Gen Controller
    UMI gripper (serial + V4L2 cameras). The two sub-devices are managed independently
    but presented through a unified LeRobot Robot interface.

    Observation keys:
        - ee_pose.{x,y,z,rx,ry,rz}: Franka end-effector pose (6 floats)
        - gripper.pos: UMI gripper opening distance in meters
        - cam_{0,1,2}: gripper-mounted camera images (H, W, 3)
        - tactile_left, tactile_right: (optional) 500-value tactile arrays

    Action keys:
        - ee_pose.{x,y,z,rx,ry,rz}: target end-effector pose
        - gripper.pos: target gripper opening distance in meters
    """

    config_class = FrankaGenGripperConfig
    name = "franka_gen_gripper"

    def __init__(self, config: FrankaGenGripperConfig):
        super().__init__(config)
        self.config = config

        # Build sub-configs and sub-robots
        franka_config = config.build_franka_config()
        gripper_config = config.build_gripper_config()

        self._franka = Franka(franka_config)
        self._gripper = UmiGripper(gripper_config)

        # Lazy-loaded UMI hand-eye extrinsic ᶠT_C (panda_link8 → camera).
        self._T_FC: np.ndarray | None = None

        # UMI t₀ 参考法兰位姿（调用 capture_t0() 后写入）
        self._T_BE_t0: np.ndarray | None = None
        self._t0_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Feature descriptors
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        features.update(self._franka.observation_features)
        features.update(self._gripper.observation_features)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        features.update(self._franka.action_features)
        features.update(self._gripper.action_features)
        return features

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._franka.is_connected
        # return self._franka.is_connected and self._gripper.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._franka.is_calibrated
        # return self._franka.is_calibrated and self._gripper.is_calibrated

    def calibrate(self) -> None:
        self._franka.calibrate()
        # self._gripper.calibrate()

    def configure(self) -> None:
        self._franka.configure()
        # self._gripper.configure()

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info("Connecting Franka + UMI gripper system ...")

        # Connect Franka arm first
        try:
            self._franka.connect()
            logger.info("Franka arm connected.")
        except Exception as e:
            raise ConnectionError(f"Franka connection failed: {e}") from e

        # # Then connect UMI gripper
        try:
            self._gripper.connect(calibrate=calibrate)
            logger.info("UMI gripper connected.")
        except Exception as e:
            # Roll back franka connection on gripper failure
            logger.warning(f"Gripper connection failed: {e}, disconnecting Franka ...")
            try:
                self._franka.disconnect()
            except Exception:
                pass
            raise ConnectionError(f"Gripper connection failed: {e}") from e

        logger.info("Franka + UMI gripper system ready.")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        errors = []
        # Disconnect gripper first (stop motors/cameras), then arm
        try:
            self._gripper.disconnect()
        except Exception as e:
            errors.append(f"Gripper disconnect error: {e}")
        try:
            self._franka.disconnect()
        except Exception as e:
            errors.append(f"Franka disconnect error: {e}")
        if errors:
            logger.warning("Disconnect completed with errors: " + "; ".join(errors))
        else:
            logger.info("Franka + UMI gripper system disconnected.")

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict[str, Any] = {}

        # Franka observations (ee_pose.x/y/z/rx/ry/rz)
        franka_obs = self._franka.get_observation()
        obs.update(franka_obs)

        # Gripper observations (gripper.pos, cam_*, tactile_*)
        gripper_obs = self._gripper.get_observation()
        obs.update(gripper_obs)

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send combined action to Franka arm and UMI gripper.
        Args:
            action: dict containing:
                - ee_pose.{x,y,z,rx,ry,rz}: target end-effector pose (Franka)
                - gripper.pos: target gripper opening distance in meters (UMI)
        Returns:
            The action actually sent (potentially clamped).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent_action: dict[str, Any] = {}

        # Split and dispatch Franka action keys
        franka_action = {k: v for k, v in action.items() if k.startswith("ee_pose.")}
        if franka_action:
            franka_sent = self._franka.send_action(franka_action)
            if franka_sent:
                sent_action.update(franka_sent)

        # Split and dispatch gripper action keys
        gripper_action = {k: v for k, v in action.items() if k.startswith("gripper.")}
        if gripper_action:
            gripper_sent = self._gripper.send_action(gripper_action)
            if gripper_sent:
                sent_action.update(gripper_sent)

        return sent_action

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Move Franka to home position and set gripper to mid-range."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._franka.reset()
        self._gripper.send_action({"gripper.pos": 0.05})
        logger.info("System reset to home position.")

    # ------------------------------------------------------------------
    # Controller lifecycle (forwards to underlying Franka)
    # ------------------------------------------------------------------

    def start_cartesian_impedance(
        self,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
        settle_time: float = 0.3,
    ) -> None:
        """Start a Cartesian impedance controller on the Franka NUC.

        Must be called before any ``send_action`` / ``send_umi_action`` loop.
        """
        self._franka.start_cartesian_impedance(Kx=Kx, Kxd=Kxd, settle_time=settle_time)

    def terminate_policy(self) -> None:
        """Terminate the currently running Franka controller, if any."""
        self._franka.terminate_policy()

    # ------------------------------------------------------------------
    # UMI ee6d coordinate-transform helpers
    # ------------------------------------------------------------------

    @property
    def umi_camera_extrinsic(self) -> np.ndarray:
        """ᶠT_C — flange (panda_link8) → camera homogeneous transform.

        Lazily loaded from ``config.camera_extrinsic_yaml_path`` on first access.
        """
        if self._T_FC is None:
            yaml_path = self.config.camera_extrinsic_yaml_path
            if yaml_path is None:
                raise RuntimeError(
                    "camera_extrinsic_yaml_path is not configured. "
                    "Set FrankaGenGripperConfig.camera_extrinsic_yaml_path to a "
                    "hand-eye calibration YAML before using UMI ee6d helpers."
                )
            self._T_FC = load_flange_to_camera_extrinsic(yaml_path)
            logger.info(
                f"Loaded UMI hand-eye extrinsic ᶠT_C from {yaml_path}: "
                f"translation={self._T_FC[:3, 3].round(4).tolist()}"
            )
        return self._T_FC

    def get_ee_se3(self) -> np.ndarray:
        """Read current ᴮT_E from the Franka FK as a 4×4 SE(3) matrix."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        franka_obs = self._franka.get_observation()
        pos = np.array([franka_obs[f"ee_pose.{a}"] for a in ("x", "y", "z")])
        rotvec = np.array([franka_obs[f"ee_pose.{a}"] for a in ("rx", "ry", "rz")])
        return pose_xyz_rotvec_to_se3(pos, rotvec)

    def umi_ee6d_to_base(
        self,
        delta_k: np.ndarray,
        T_BE_t0: np.ndarray,
        reference: str = "camera_t0",
    ) -> np.ndarray:
        """将 UMI ee6d 动作转换为基坐标系目标法兰位姿 ᴮT_E(t_k)。

        Args:
            delta_k:   (4, 4) 策略/数据集输出的 ee6d SE(3)。语义取决于 ``reference``：
                       - ``"camera_t0"``：delta_k 必须是原始相机相对运动
                         ``ΔC_k = inv(ᵂT_C(t₀)) · ᵂT_C(t_k)``；
                       - ``"ee_at_t0"``：delta_k 必须是 EE-at-t₀ 法兰相对位姿
                         ``Δ_k = inv(ᵂT_F(t₀)) · ᵂT_F(t_k)``（由
                         ``compute_ee_at_t0_deltas`` 预处理得到）。
            T_BE_t0:   (4, 4) t₀ 时刻法兰位姿 FK 快照 ᴮT_E。
            reference: ``UMI_REFERENCE_MODES`` 之一：
                * ``"camera_t0"``         — Camera-at-t₀ 相机原始运动参考。
                * ``"ee_at_t0"``          — EE-at-t₀ 法兰相对位姿（推荐格式）。
                * ``"abs_pos_world_rot"`` — t₀ 相对平移 + 绝对姿态（特殊预处理变体）。

        Returns:
            (4, 4) SE(3)，机器人基坐标系下的目标法兰位姿。
        """
        if reference == "camera_t0":
            return umi_camera_t0_action_to_base(delta_k, T_BE_t0, self.umi_camera_extrinsic)
        if reference == "ee_at_t0":
            return ee_relative_action_to_base(delta_k, T_BE_t0)
        if reference == "abs_pos_world_rot":
            return absolute_position_world_orientation_to_base(delta_k, T_BE_t0)
        raise ValueError(
            f"Unknown UMI reference mode {reference!r}. Expected one of {UMI_REFERENCE_MODES}."
        )

    def send_umi_action(
        self,
        delta_k: np.ndarray,
        T_BE_t0: np.ndarray,
        *,
        reference: str = "camera_t0",
        gripper_width: float | None = None,
    ) -> dict[str, Any]:
        """Convert a UMI ee6d action to a base-frame command and send it.

        Args:
            delta_k:       (4, 4) ee6d SE(3) action from the dataset / policy.
            T_BE_t0:       (4, 4) ᴮT_E snapshot captured at observation time t₀.
            reference:     UMI reference mode (see ``umi_ee6d_to_base``).
            gripper_width: Target opening (m). If ``None``, the gripper is left
                untouched — useful when validating ee6d only.

        Returns:
            The action actually sent to the underlying robot.
        """
        T_BE_target = self.umi_ee6d_to_base(delta_k, T_BE_t0, reference=reference)
        pose6d = se3_to_xyz_rotvec(T_BE_target)
        action: dict[str, Any] = {key: float(pose6d[i]) for i, key in enumerate(EE_POSE_KEYS)}
        if gripper_width is not None:
            action["gripper.pos"] = float(gripper_width)
        return self.send_action(action)

    # ------------------------------------------------------------------
    # UMI 数据集格式接口（与 pick_and_place 数据集 info.json 对齐）
    # ------------------------------------------------------------------

    @property
    def umi_observation_features(self) -> dict[str, dict]:
        """返回与数据集对齐的 lerobot_features（8D quat + camera0）。"""
        h = self.config.gripper_camera_height
        w = self.config.gripper_camera_width
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["pos_x", "pos_y", "pos_z",
                          "quat_x", "quat_y", "quat_z", "quat_w",
                          "gripper"],
            },
            "observation.images.camera0": {
                "dtype": "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
        }

    def get_umi_observation(self) -> dict[str, Any]:
        """返回数据集格式观测（旋转向量 → 四元数，key 与数据集 info.json 一致）。"""
        from scipy.spatial.transform import Rotation

        raw = self.get_observation()
        pos = np.array([raw["ee_pose.x"], raw["ee_pose.y"], raw["ee_pose.z"]])
        rotvec = np.array([raw["ee_pose.rx"], raw["ee_pose.ry"], raw["ee_pose.rz"]])
        quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()  # scipy 默认 xyzw

        return {
            "pos_x": float(pos[0]),
            "pos_y": float(pos[1]),
            "pos_z": float(pos[2]),
            "quat_x": float(quat_xyzw[0]),
            "quat_y": float(quat_xyzw[1]),
            "quat_z": float(quat_xyzw[2]),
            "quat_w": float(quat_xyzw[3]),
            "gripper": float(raw.get("gripper.pos", 0.0)),
            "camera0": raw.get("camera0"),
        }

    def capture_t0(self) -> None:
        """捕获当前法兰位姿作为 UMI t₀ 参考点（线程安全）。"""
        se3 = self.get_ee_se3()
        with self._t0_lock:
            self._T_BE_t0 = se3
        logger.info(f"ᴮT_E(t₀): {se3[:3, 3].round(4).tolist()}")

    def get_t0(self) -> np.ndarray | None:
        """线程安全地读取 T_BE_t0。"""
        with self._t0_lock:
            return self._T_BE_t0
