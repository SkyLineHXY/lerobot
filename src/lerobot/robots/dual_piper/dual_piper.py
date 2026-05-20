import concurrent.futures
import logging
import time
from functools import cached_property
from typing import Any
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig

from lerobot.robots.robot import Robot
from lerobot.robots.dual_piper.config_dual_piper import DualPIPERConfig

logger = logging.getLogger(__name__)


def get_motor_names(arms: dict[str, Any]) -> list[str]:
    """Extract motor names from all arms in order."""
    return [motor for arm_key, bus in arms.items() for motor in bus.motors]


class DualPiper(Robot):
    """Dual arm Piper robot with 14 joints total (7 per arm).
    
    This robot consists of:
    - Left arm: 7 joints (joint_1 to joint_7)
    - Right arm: 7 joints (joint_8 to joint_14)
    - Multiple cameras (typically wrist cameras and top camera)
    
    The state and action vectors contain 14 values corresponding to all joints.
    """
    config_class = DualPIPERConfig
    name = "dual_piper"

    def __init__(self, config: DualPIPERConfig):
        super().__init__(config)

        self.config = config
        
        # Initialize left arm motor bus
        self.bus_left = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=self.config.can_port_left,
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "joint_7": (7, "agilex_piper"),
                }
            )
        )
        
        # Initialize right arm motor bus
        self.bus_right = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name=self.config.can_port_right,
                motors={
                    "joint_8": (1, "agilex_piper"),
                    "joint_9": (2, "agilex_piper"),
                    "joint_10": (3, "agilex_piper"),
                    "joint_11": (4, "agilex_piper"),
                    "joint_12": (5, "agilex_piper"),
                    "joint_13": (6, "agilex_piper"),
                    "joint_14": (7, "agilex_piper"),
                }
            )
        )
        
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False
        self.cameras = make_cameras_from_configs(config.cameras)

        # Persistent thread pool for parallel arm control (avoids thread creation overhead)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @property
    def camera_features(self) -> dict:
        """Define camera features for dataset."""
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        """Define motor features for dataset with 14 joints."""
        arm_dict = {"left": self.bus_left, "right": self.bus_right}
        action_names = get_motor_names(arm_dict)
        state_names = get_motor_names(arm_dict)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor action features for record/replay."""
        arm_dict = {"left": self.bus_left, "right": self.bus_right}
        motor_names = get_motor_names(arm_dict)
        return {f"{name}.pos": float for name in motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera image features for record/replay."""
        return {
            f"observation.images.{cam_key}": (cam.height, cam.width, 3)
            for cam_key, cam in self.cameras.items()
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def configure(self, **kwargs):
        """Configuration method (required by abstract base class)."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if both robot arms and all cameras are connected."""
        return (
            self.bus_left.is_connected 
            and self.bus_right.is_connected 
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        """Check if both robot arms are calibrated."""
        return self.bus_left.is_calibrated and self.bus_right.is_calibrated

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self, calibrate=True) -> None:
        """Connect both Piper arms and all cameras."""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "Dual Piper is already connected. Do not run robot.connect() twice."
            )

        # Connect left arm
        self.bus_left.connect(enable=True)
        print("Dual Piper left arm connected")

        # Connect right arm
        self.bus_right.connect(enable=True)
        print("Dual Piper right arm connected")

        # Connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self._is_connected = self._is_connected and self.cameras[name].is_connected
            print(f"Camera {name} connected")

        print("All devices connected")
        self._is_connected = True
        
        if calibrate:
            self.calibrate()

    def disconnect(self) -> None:
        """Move to home position and disable both Piper arms and cameras."""
        # Wait for any pending actions to complete before disconnecting
        if hasattr(self, '_left_future') and self._left_future is not None:
            try:
                self._left_future.result(timeout=1.0)
            except Exception:
                pass
        if hasattr(self, '_right_future') and self._right_future is not None:
            try:
                self._right_future.result(timeout=1.0)
            except Exception:
                pass

        # Shutdown thread pool
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

        # Safely disconnect both arms
        self.bus_left.safe_disconnect()
        self.bus_right.safe_disconnect()
        print("Dual Piper arms will disable after 5 seconds")
        time.sleep(5)
        
        self.bus_left.connect(enable=False)
        self.bus_right.connect(enable=False)

        # Disconnect cameras
        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self._is_connected = False

    def calibrate(self):
        """Move both Piper arms to their home positions."""
        if not self._is_connected:
            raise ConnectionError("Robot must be connected before calibration")
        self.bus_left.apply_calibration()
        self.bus_right.apply_calibration()
        self._is_calibrated = True
        print("Both arms calibrated")

    def get_observation(self) -> dict:
        """Capture current joint positions from both arms and camera images.
        
        Returns:
            Dictionary containing:
            - joint_1.pos to joint_7.pos: left arm positions
            - joint_8.pos to joint_14.pos: right arm positions
            - camera images for each configured camera
        """

        if not self._is_connected:
            raise DeviceNotConnectedError("Dual Piper is not connected. Run `robot.connect()` first.")
        #t0 = time.perf_counter()
        # Read left arm state (joints 1-7)
        state_left = self.bus_left.read()
        obs_dict = {f"{joint}.pos": float(val) for joint, val in state_left.items()}
        #t1 = time.perf_counter()
        # Read right arm state (joints 8-14)
        state_right = self.bus_right.read()
        obs_dict.update({f"{joint}.pos": float(val) for joint, val in state_right.items()})
        #t2 = time.perf_counter()
        #print(f"read Left arm: {(t1 - t0) * 1000:.2f}ms, read Right arm: {(t2 - t1) * 1000:.2f}ms")

        # Read camera images

        for name, cam in self.cameras.items():
            #start = time.perf_counter()
            obs_dict[f"{name}"] = cam.read()
            #obs_dict[f"{name}"] = cam.async_read()
            #dt_ms = (time.perf_counter() - start) * 1e3
            #logger.info(f"{self} read {name}: {dt_ms:.1f}ms")

        # for name, cam in self.cameras.items():
        #     start = time.perf_counter()
        #     frame = cam.async_read()
        #     # 如果需要，在这里做必要的数据拷贝，避免后续竞争
        #     if frame is not None:
        #         obs_dict[f"{name}"] = frame.copy()  # 深拷贝避免数据竞争
        #     else:
        #         obs_dict[f"{name}"] = frame
        #     dt_ms = (time.perf_counter() - start) * 1e3
        #     logger.info(f"{self} read {name}: {dt_ms:.1f}ms")

        return obs_dict

    # def send_action(self, action: dict[str, float]) -> dict[str, float]:
    #      """Send action commands to both arms.
    #
    #      Args:
    #          action: Dictionary with keys joint_1.pos through joint_14.pos
    #                 - joint_1 to joint_7: left arm
    #                 - joint_8 to joint_14: right arm
    #
    #      Returns:
    #          The action dictionary that was sent
    #      """
    #
    #      if not self._is_connected:
    #          raise DeviceNotConnectedError("Dual Piper is not connected.")
    #      # Prepare actions for left arm (joints 1-7)
    #      #t0 = time.perf_counter()
    #
    #      left_motor_order = ["joint_1", "joint_2", "joint_3", "joint_4",
    #                         "joint_5", "joint_6", "joint_7"]
    #      target_joints_left = [action[f"{motor}.pos"] for motor in left_motor_order]
    #      self.bus_left.write(target_joints_left)
    #
    #      # Prepare actions for right arm (joints 8-14)
    #      #t1 = time.perf_counter()
    #      right_motor_order = ["joint_8", "joint_9", "joint_10", "joint_11",
    #                          "joint_12", "joint_13", "joint_14"]
    #      target_joints_right = [action[f"{motor}.pos"] for motor in right_motor_order]
    #      self.bus_right.write(target_joints_right)
    #      #t2 = time.perf_counter()
    #      #print(f"Left arm: {(t1 - t0) * 1000:.2f}ms, Right arm: {(t2 - t1) * 1000:.2f}ms")
    #      return action
    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send action commands to both arms in parallel (non-blocking).

        Args:
            action: Dictionary with keys joint_1.pos through joint_14.pos
                   - joint_1 to joint_7: left arm
                   - joint_8 to joint_14: right arm

        Returns:
            The action dictionary that was sent

        Note:
            This method uses non-blocking sends - it submits the commands
            to both arms and returns immediately without waiting for completion.
            This reduces latency from ~30ms to ~1-2ms.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("Dual Piper is not connected.")
        #t0 = time.perf_counter()
        # Prepare actions for left arm (joints 1-7)
        left_motor_order = ["joint_1", "joint_2", "joint_3", "joint_4",
                           "joint_5", "joint_6", "joint_7"]
        target_joints_left = [action[f"{motor}.pos"] for motor in left_motor_order]

        # Prepare actions for right arm (joints 8-14)
        right_motor_order = ["joint_8", "joint_9", "joint_10", "joint_11",
                            "joint_12", "joint_13", "joint_14"]
        target_joints_right = [action[f"{motor}.pos"] for motor in right_motor_order]
        #t1 = time.perf_counter()
        # Non-blocking send: submit tasks but don't wait for completion
        # Store futures in case we need to check status later

        self._left_future = self._executor.submit(self.bus_left.write, target_joints_left)
        #t2 = time.perf_counter()
        self._right_future = self._executor.submit(self.bus_right.write, target_joints_right)
        #t3 = time.perf_counter()
        #print(f"Left arm submit: {(t2 - t1) * 1000:.2f}ms, Right arm submit: {(t3 - t2) * 1000:.2f}ms")
        # Wait 2ms to ensure commands are sent
        time.sleep(0.005)

        # Return immediately without waiting
        return action




