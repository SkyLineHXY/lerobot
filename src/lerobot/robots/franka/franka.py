import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_franka import FrankaConfig
from .franka_interface_client import FrankaInterfaceClient

logger = logging.getLogger(__name__)

# Observation / action key names — consistent "{group}.{component}" convention
EE_POSE_KEYS = ("ee_pose.x", "ee_pose.y", "ee_pose.z", "ee_pose.rx", "ee_pose.ry", "ee_pose.rz")

# 夹爪观测/动作 key
# GRIPPER_STATE_KEYS = ("gripper.width", "gripper.is_grasped", "gripper.is_moving")
GRIPPER_STATE_KEYS = ("gripper.width", "gripper.is_grasped", "gripper.is_moving")
GRIPPER_ACTION_KEY = "gripper.width"
GRIPPER_GRASP_THRESHOLD_M = 0.02
GRIPPER_COMMAND_EPS_M = 1e-3
GRIPPER_RETRY_INTERVAL_S = 1.0

# Default Cartesian impedance gains. Match the values used in
# `franka_interface_client.test_action_tracking_cartesian` and produce stable
# 30 Hz tracking for ee6d replay / teleop.
DEFAULT_CARTESIAN_KX = (600.0, 600.0, 600.0, 30.0, 30.0, 30.0)
DEFAULT_CARTESIAN_KXD = (60.0, 60.0, 60.0, 5.0, 5.0, 5.0)


class Franka(Robot):
    """LeRobot driver for a Franka Emika arm via zerorpc.
    Communicates with a `franka_interface_server` running on a real-time NUC.
    Observations and actions are 6-DoF end-effector poses (x, y, z, rx, ry, rz).
    """
    config_class = FrankaConfig
    name = "franka"

    def __init__(self, config: FrankaConfig):
        super().__init__(config)
        self.config = config
        self._robot: FrankaInterfaceClient | None = None
        self._is_connected = False
        self._prev_observation: dict[str, Any] | None = None
        self._last_gripper_command_width: float | None = None
        self._last_gripper_command_mode: str | None = None
        self._last_gripper_command_time_s: float = 0.0
        self._last_gripper_state: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Feature descriptors
    # ------------------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict[str, type]:
        features: dict[str, type] = {key: float for key in EE_POSE_KEYS}
        if self.config.use_gripper:
            features["gripper.width"] = float
            features["gripper.is_grasped"] = bool
            features["gripper.is_moving"] = bool
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {key: float for key in EE_POSE_KEYS}
        if self.config.use_gripper:
            features[GRIPPER_ACTION_KEY] = float
        return features

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting to Franka at {self.config.robot_ip}:{self.config.robot_port} ...")
        try:
            client = FrankaInterfaceClient(ip=self.config.robot_ip, port=self.config.robot_port)
            # Verify connectivity by reading joint positions
            joints = client.robot_get_joint_positions()
            client.robot_go_home()
            if joints is None or len(joints) != 7:
                raise ConnectionError("Failed to read joint positions — check robot state and remote mode")
            logger.info(f"Franka connected. Joints: {[round(j, 4) for j in joints]}")

            if self.config.use_gripper:
                try:
                    gripper_state = client.gripper_get_state()
                    self._last_gripper_state = dict(gripper_state)
                    logger.info(
                        f"Franka gripper detected: width={gripper_state['width']:.4f}, "
                        f"is_grasped={gripper_state['is_grasped']}"
                    )
                except Exception as e:
                    logger.warning(f"Gripper check failed (use_gripper=True but gripper unavailable): {e}")
            self._robot = client
            self._is_connected = True
        except Exception as e:
            self._is_connected = False
            raise ConnectionError(f"Franka connection failed: {e}") from e

    def disconnect(self) -> None:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._robot is not None:
            try:
                self._robot.close()
            except Exception as e:
                logger.warning(f"Error closing Franka client: {e}")
            self._robot = None

        self._is_connected = False
        self._reset_gripper_command_state()
        logger.info("Franka disconnected.")

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------
    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            ee_pose = self._robot.robot_get_ee_pose()  # np.ndarray of 6 floats
        except Exception as e:
            logger.warning(f"Franka get_ee_pose error: {e}")
            if self._prev_observation is not None:
                return self._prev_observation
            raise
        obs: dict[str, Any] = {key: float(ee_pose[i]) for i, key in enumerate(EE_POSE_KEYS)}

        if self.config.use_gripper:
            try:
                gripper_state = self._robot.gripper_get_state()
                self._last_gripper_state = dict(gripper_state)
                obs["gripper.width"] = float(gripper_state["width"])
                obs["gripper.is_grasped"] = bool(gripper_state.get("is_grasped", False))
                obs["gripper.is_moving"] = bool(gripper_state.get("is_moving", False))
            except Exception as e:
                logger.warning(f"Franka gripper_get_state error: {e}")

        self._prev_observation = obs
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作命令到 Franka 机械臂（及可选夹爪）。

        Args:
            action: 字典，必须包含 ee_pose.{x,y,z,rx,ry,rz}，
                    可选包含 gripper.width（需 use_gripper=True）。

        Returns:
            实际发送的动作字典。
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent_action: dict[str, Any] = {}

        # 机械臂 EE 位姿
        pose = np.array([float(action[key]) for key in EE_POSE_KEYS])
        self._robot.robot_update_desired_ee_pose(pose)
        for i, key in enumerate(EE_POSE_KEYS):
            sent_action[key] = float(pose[i])

        # 夹爪控制
        if self.config.use_gripper and GRIPPER_ACTION_KEY in action:
            width = float(action[GRIPPER_ACTION_KEY])
            self._send_gripper_command(width)
            sent_action[GRIPPER_ACTION_KEY] = width

        return sent_action

    def _send_gripper_command(self, width: float) -> None:
        """Send edge-triggered, non-blocking gripper commands.

        Re-sending ``goto`` or ``grasp`` every control frame restarts the Franka
        Hand command. That is especially problematic after contact: repeated
        ``grasp`` keeps the hand in a fresh grasp attempt and can prevent a
        later open command from taking effect cleanly.
        """
        width = max(0.0, width)
        mode = "grasp" if width <= GRIPPER_GRASP_THRESHOLD_M else "goto"
        state = self._last_gripper_state or {}
        now_s = time.perf_counter()

        command_changed = (
            self._last_gripper_command_mode != mode
            or self._last_gripper_command_width is None
            or abs(width - self._last_gripper_command_width) > GRIPPER_COMMAND_EPS_M
        )
        should_retry = self._should_retry_gripper_command(width, mode, state, now_s)

        if not command_changed and not should_retry:
            return

        if mode == "grasp":
            self._robot.gripper_grasp(
                speed=self.config.gripper_speed,
                force=self.config.gripper_force,
                grasp_width=width,
                blocking=False,
            )
        else:
            self._robot.gripper_goto(
                width=width,
                speed=self.config.gripper_speed,
                force=self.config.gripper_force,
                blocking=False,
            )

        self._last_gripper_command_width = width
        self._last_gripper_command_mode = mode
        self._last_gripper_command_time_s = now_s

    def _should_retry_gripper_command(
        self,
        width: float,
        mode: str,
        state: dict[str, Any],
        now_s: float,
    ) -> bool:
        if now_s - self._last_gripper_command_time_s < GRIPPER_RETRY_INTERVAL_S:
            return False

        if bool(state.get("is_moving", False)):
            return False

        if mode == "grasp":
            return False

        current_width = state.get("width")
        if current_width is None:
            return False
        return abs(float(current_width) - width) > GRIPPER_COMMAND_EPS_M

    def _reset_gripper_command_state(self) -> None:
        self._last_gripper_command_width = None
        self._last_gripper_command_mode = None
        self._last_gripper_command_time_s = 0.0
        self._last_gripper_state = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Move arm to home position."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self._robot.robot_go_home()
        logger.info("Franka reset to home.")

    # ------------------------------------------------------------------
    # Controller lifecycle (must wrap any send_action loop)
    # ------------------------------------------------------------------
    def start_cartesian_impedance(
        self,
        Kx: np.ndarray | None = None,
        Kxd: np.ndarray | None = None,
        settle_time: float = 0.3,
    ) -> None:
        """Start a Cartesian impedance controller on the NUC.

        Required before any ``send_action`` call — without an active controller
        polymetis raises ``Tried to perform a controller update with no
        controller running``.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        Kx = np.asarray(Kx if Kx is not None else DEFAULT_CARTESIAN_KX, dtype=float)
        Kxd = np.asarray(Kxd if Kxd is not None else DEFAULT_CARTESIAN_KXD, dtype=float)
        self._robot.robot_start_cartesian_impedance_control(Kx=Kx, Kxd=Kxd)
        if settle_time > 0:
            import time as _time
            _time.sleep(settle_time)
        logger.info(f"Cartesian impedance started (Kx={Kx.tolist()}, Kxd={Kxd.tolist()}).")


    def terminate_policy(self) -> None:
        """Terminate the currently running polymetis controller, if any."""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            self._robot.robot_terminate_current_policy()
            logger.info("Franka controller terminated.")
        except Exception as e:
            # Idempotent: tolerate "no controller running" so this can safely
            # be called from a finally block.
            logger.warning(f"terminate_policy: {e}")
