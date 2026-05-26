import logging
import struct
import threading
import time
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_umi_gripper import UmiGripperConfig
from .gen_con_sdk_python_release.scripts.camera import CameraCapture
from .gen_con_sdk_python_release.scripts.databus import DataBus
from .gen_con_sdk_python_release.tactile_processing import convert_tactile_448_to_1000

logger = logging.getLogger(__name__)


class UmiGripper(Robot):
    """LeRobot driver for the Gen Controller UMI gripper.

    The gripper has one degree of freedom (opening distance 0~0.103 m),
    up to 3 on-board cameras, and optional tactile sensors (left/right pads,
    each producing 500 values = 1000 total).

    Hardware communication is handled by the gen_con_sdk_python_release SDK:
      - DataBus: serial communication (921600 baud) for motor control, encoder
        reading, and tactile data streaming via callback threads.
      - CameraCapture: V4L2 multi-camera acquisition at ~30 FPS via OpenCV.
    """

    config_class = UmiGripperConfig
    name = "umi_gripper"

    def __init__(self, config: UmiGripperConfig):
        super().__init__(config)
        self.config = config

        # SDK handles
        self._databus: DataBus | None = None
        self._camera: CameraCapture | None = None
        self._capture_thread: threading.Thread | None = None

        # Connection state
        self._connected = False

        # Latest sensor data (updated by SDK callbacks in background threads)
        self._data_lock = threading.Lock()
        self._latest_encoder: float = 0.0
        self._latest_frames: dict[int, np.ndarray] = {}
        self._latest_tactile_left: list[int] | None = None
        self._latest_tactile_right: list[int] | None = None

    # ------------------------------------------------------------------
    # Feature descriptors
    # ------------------------------------------------------------------

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {"gripper.pos": float}
        h, w = self.config.camera_height, self.config.camera_width
        for i in range(self.config.camera_count):
            features[f"camera{i}"] = (h, w, 3)
        if self.config.enable_tactile:
            features["tactile_left"] = (500,)
            features["tactile_right"] = (500,)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {"gripper.pos": float}

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Trigger encoder calibration on the gripper hardware."""
        if self._databus is not None:
            self._databus.calib_encoder()
            logger.info("UMI gripper encoder calibration sent.")

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # SDK callbacks (called from DataBus / CameraCapture background threads)
    # ------------------------------------------------------------------

    def _encoder_callback(self, record_data: bytes) -> None:
        """Called by DataBus when a new encoder reading arrives (4 bytes big-endian float)."""
        try:
            value = struct.unpack(">f", record_data)[0]
            with self._data_lock:
                self._latest_encoder = value
        except Exception as e:
            logger.warning(f"Encoder callback error: {e}")

    def _tactile_callback(self, record_data: bytes) -> None:
        """Called by DataBus when a new tactile frame arrives (448 bytes)."""
        try:
            left_500, right_500 = convert_tactile_448_to_1000(record_data)
            with self._data_lock:
                self._latest_tactile_left = left_500
                self._latest_tactile_right = right_500
        except Exception as e:
            logger.warning(f"Tactile callback error: {e}")

    def _frame_callback(self, cam_id: int, frame: np.ndarray, timestamp_ns: int) -> None:
        """Called by the capture loop for each camera frame."""
        with self._data_lock:
            self._latest_frames[cam_id] = frame

    # ------------------------------------------------------------------
    # Camera capture loop (runs in a daemon thread)
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read frames from all cameras at ~30 FPS and push them to _latest_frames."""
        camera = self._camera
        if camera is None:
            return

        target_fps = 30
        frame_interval = 1.0 / target_fps

        try:
            while camera.running and self._connected:
                start_time = time.time()
                timestamp_ns = time.time_ns()

                for cam in camera.cameras:
                    ret, frame = cam["cap"].read()
                    if ret and frame is not None:
                        self._frame_callback(cam["id"], frame, timestamp_ns)
                        cam["frame_count"] += 1

                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Capture loop error: {e}")
        finally:
            camera._release_resources()

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting UMI gripper ({self.config.side} side) ...")

        # 1) Initialize cameras (non-blocking)
        try:
            self._camera = CameraCapture(
                serial_port=self.config.serial_port,
                camera_count=self.config.camera_count,
                resolutions=[(self.config.camera_width, self.config.camera_height)],
                show_preview=self.config.camera_show_preview,
                video_devices=self.config.video_devices,
                frame_callback=self._frame_callback
            )
            logger.info(f"Cameras initialized: {len(self._camera.cameras)} device(s)")
        except Exception as e:
            raise ConnectionError(f"Camera init failed: {e}") from e

        # Small delay for camera hardware stabilization
        time.sleep(0.5)

        # 2) Initialize DataBus (serial communication, starts its own threads)
        try:
            self._databus = DataBus(
                tty_port=self.config.serial_port,
                baudrate=921600,
                encoder_freq=self.config.encoder_freq,
                tactile_callback=self._tactile_callback if self.config.enable_tactile else None,
                encoder_callback=self._encoder_callback,
            )
            logger.info("Serial communication ready")
        except Exception as e:
            # Clean up camera if serial fails
            if self._camera:
                self._camera.stop()
                self._camera = None
            raise ConnectionError(f"Serial init failed: {e}") from e

        # 3) Start camera capture thread
        self._connected = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info(f"UMI gripper ({self.config.side}) connected.")

    def disconnect(self) -> None:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._connected = False

        # Stop camera capture
        if self._camera:
            self._camera.running = False
            self._camera.stop()
            self._camera = None

        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None

        # Stop serial communication
        if self._databus:
            self._databus.stop()
            self._databus = None

        logger.info(f"UMI gripper ({self.config.side}) disconnected.")

    # ------------------------------------------------------------------
    # Observation / Action
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._data_lock:
            obs: dict[str, Any] = {"gripper.pos": self._latest_encoder}

            for i in range(self.config.camera_count):
                frame = self._latest_frames.get(i)
                if frame is not None:
                    obs[f"camera{i}"] = frame.copy()
                else:
                    # Return a black frame if no data yet
                    obs[f"camera{i}"] = np.zeros(
                        (self.config.camera_height, self.config.camera_width, 3),
                        dtype=np.uint8,
                    )

            if self.config.enable_tactile:
                if self._latest_tactile_left is not None:
                    obs["tactile_left"] = np.array(self._latest_tactile_left, dtype=np.int16)
                else:
                    obs["tactile_left"] = np.zeros(500, dtype=np.int16)
                if self._latest_tactile_right is not None:
                    obs["tactile_right"] = np.array(self._latest_tactile_right, dtype=np.int16)
                else:
                    obs["tactile_right"] = np.zeros(500, dtype=np.int16)

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Set the gripper target opening distance.

        Args:
            action: dict with key "gripper.pos" (float, meters in [0.0, 0.103]).

        Returns:
            The action actually sent (clamped to safe range).
        """
        if not self._connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        distance = float(action["gripper.pos"])
        distance = max(self.config.gripper_min_distance, min(self.config.gripper_max_distance, distance))

        self._databus.set_target_distance(distance)
        return {"gripper.pos": distance}


