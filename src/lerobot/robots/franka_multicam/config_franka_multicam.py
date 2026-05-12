"""FrankaMultiCam 机器人配置，组合 Franka 机械臂 + 多路相机。"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("franka_multicam")
@dataclass
class FrankaMultiCamConfig(RobotConfig):
    """Franka 机械臂 + 多路相机的组合机器人配置。

    Example::

        FrankaMultiCamConfig(
            robot_ip="192.168.172.252",
            robot_port=4242,
            use_gripper=True,
            cameras={
                "front": HikCameraConfig(mode="remote", host="192.168.172.134", port=4243,
                                         fps=30, width=1280, height=720),
                "wrist": OpenCVZmqCameraConfig(index_or_path=0, mode="remote",
                                               host="192.168.172.134", port=4244,
                                               fps=30, width=640, height=480),
            },
        )
    """

    robot_ip: str = "192.168.172.252"
    """Franka NUC 的 IP 地址。"""

    robot_port: int = 4242
    """Franka NUC 上 franka_interface_server 的 zerorpc 端口。"""

    use_gripper: bool = True
    """是否启用夹爪控制与观测。"""

    gripper_speed: float = 0.1
    """夹爪运动速度（m/s）。"""

    gripper_force: float = 1.0
    """夹爪最大力（N）。"""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    """多路相机配置字典，key 为相机名称（如 "front", "wrist"）。"""
