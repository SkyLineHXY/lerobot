from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from lerobot.robots.config import RobotConfig


# Left/right side default device mappings (set by udev rules)
SIDE_DEFAULTS = {
    "left": {
        "serial_port": "/dev/ttyDeviceLeft",
        "video_devices": [
            "/dev/left_video_0_main", "/dev/left_video_0_sec",
            "/dev/left_video_1_main", "/dev/left_video_1_sec",
            "/dev/left_video_2_main", "/dev/left_video_2_sec",
        ],
    },
    "right": {
        "serial_port": "/dev/ttyDeviceRight",
        "video_devices": [
            "/dev/right_video_0_main", "/dev/right_video_0_sec",
            "/dev/right_video_1_main", "/dev/right_video_1_sec",
            "/dev/right_video_2_main", "/dev/right_video_2_sec",
        ],
    },
}


@RobotConfig.register_subclass("umi_gripper")
@dataclass
class UmiGripperConfig(RobotConfig):
    # Gripper side: "left" or "right", determines default serial/video device paths
    side: str = "left"

    # Serial port path; None = use default based on side
    serial_port: Optional[str] = None

    # Camera resolution as (width, height)
    camera_width: int = 640
    camera_height: int = 480

    # Number of cameras on the gripper (typically 3)
    camera_count: int = 3

    # Whether to show OpenCV preview windows for cameras
    camera_show_preview: bool = False

    # Explicit video device paths; None = use defaults based on side
    video_devices: Optional[list[str]] = None

    # Whether to enable tactile sensor data collection
    enable_tactile: bool = False

    # Encoder polling frequency in Hz
    encoder_freq: float = 30.0

    # Gripper distance safe range (meters)
    gripper_min_distance: float = 0.01
    gripper_max_distance: float = 0.093

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got '{self.side}'")

        defaults = SIDE_DEFAULTS[self.side]
        if self.serial_port is None:
            self.serial_port = defaults["serial_port"]
        if self.video_devices is None:
            self.video_devices = defaults["video_devices"]

