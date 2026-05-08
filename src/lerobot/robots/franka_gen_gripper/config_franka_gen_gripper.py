from dataclasses import dataclass, field
from typing import Optional

from lerobot.robots.config import RobotConfig
from lerobot.robots.franka.config_franka import FrankaConfig
from lerobot.robots.gen_gripper.config_umi_gripper import UmiGripperConfig


@RobotConfig.register_subclass("franka_gen_gripper")
@dataclass
class FrankaGenGripperConfig(RobotConfig):
    # ---- Franka arm settings ----
    robot_ip: str = "172.20.10.2"
    robot_port: int = 4242
    control_mode: str = "cartesian_impedance"

    # ---- UMI gripper settings ----
    gripper_side: str = "left"
    gripper_serial_port: Optional[str] = None
    gripper_camera_width: int = 640
    gripper_camera_height: int = 480
    gripper_camera_count: int = 1
    gripper_camera_show_preview: bool = False
    gripper_video_devices: Optional[list[str]] = None
    gripper_enable_tactile: bool = False
    gripper_encoder_freq: float = 30.0

    # ---- UMI ee6d 坐标变换配置 ----
    # hand-eye 标定 YAML（parent_frame: panda_link8, child_frame: camera_color_optical_frame）
    camera_extrinsic_yaml_path: Optional[str] = '/home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml'

    # ---- 笛卡尔阻抗控制增益（None 使用 Franka 类默认值）----
    Kx: Optional[list[float]] = None
    Kxd: Optional[list[float]] = None


    def build_franka_config(self) -> FrankaConfig:
        """Build a FrankaConfig from the combined config's arm settings."""
        return FrankaConfig(
            id=f"{self.id}_franka" if self.id else None,
            robot_ip=self.robot_ip,
            robot_port=self.robot_port,
            control_mode=self.control_mode,
        )

    def build_gripper_config(self) -> UmiGripperConfig:
        """Build a UmiGripperConfig from the combined config's gripper settings."""
        return UmiGripperConfig(
            id=f"{self.id}_gripper" if self.id else None,
            side=self.gripper_side,
            serial_port=self.gripper_serial_port,
            camera_width=self.gripper_camera_width,
            camera_height=self.gripper_camera_height,
            camera_count=self.gripper_camera_count,
            camera_show_preview=self.gripper_camera_show_preview,
            video_devices=self.gripper_video_devices,
            enable_tactile=self.gripper_enable_tactile,
            encoder_freq=self.gripper_encoder_freq,
        )
