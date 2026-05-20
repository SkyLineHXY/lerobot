from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaConfig(RobotConfig):
    # zerorpc server address on the NUC
    robot_ip: str = "192.168.172.252"
    robot_port: int = 4242

    # Control mode: "cartesian_impedance" or "joint_impedance"
    control_mode: str = "cartesian_impedance"

    # Whether to print verbose logs
    debug: bool = True

    # ---- 夹爪配置 ----
    # 是否启用 Franka Hand 夹爪控制
    use_gripper: bool = True
    # 夹爪运动速度 (m/s)
    gripper_speed: float = 0.1
    # 夹爪最大抓取力 (N)
    gripper_force: float = 1
