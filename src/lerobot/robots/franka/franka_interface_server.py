"""在连接 Franka 机器人的 NUC 上运行，提供 zerorpc 服务接口。

前置条件：需先在 polymetis-local 环境中启动 Polymetis gRPC 服务::

    launch_robot.py robot_client=franka_hardware
    launch_gripper.py gripper=franka_hand

启动方式::

    python franka_interface_server.py --bind 0.0.0.0 --port 4242
"""
import argparse
import logging
import os
import time

import numpy as np
import scipy.spatial.transform as st
import torch
import zerorpc
from polymetis import GripperInterface, RobotInterface

log = logging.getLogger(__name__)

_POLYMETIS_RETRY_TIMES = 30
_POLYMETIS_RETRY_INTERVAL = 2.0


class FrankaInterfaceServer:
    def __init__(
        self,
        robot_ip: str = "localhost",
        robot_port: int = 50051,
        gripper_ip: str = "localhost",
        gripper_port: int = 50052,
    ):
        # 等待 Polymetis gRPC 服务就绪（launch_robot.py 需提前启动）
        if gripper_ip is None:
            gripper_ip = robot_ip
        last_exc: Exception | None = None
        for attempt in range(_POLYMETIS_RETRY_TIMES):
            try:
                self.robot = RobotInterface(
                    ip_address=robot_ip,
                    port=robot_port,
                    enforce_version=False,
                )
                self.gripper = GripperInterface(
                    ip_address=gripper_ip,
                    port=gripper_port,
                )
                log.info("FrankaInterfaceServer 初始化成功")
                return
            except Exception as e:
                last_exc = e
                remaining = _POLYMETIS_RETRY_TIMES - attempt - 1
                log.warning(
                    "连接 Polymetis（%s:%d）失败（第 %d/%d 次），%.1fs 后重试: %s",
                    robot_ip,
                    robot_port,
                    attempt + 1,
                    _POLYMETIS_RETRY_TIMES,
                    _POLYMETIS_RETRY_INTERVAL,
                    e,
                )
                if remaining > 0:
                    time.sleep(_POLYMETIS_RETRY_INTERVAL)

        log.error(
            "无法连接 Polymetis gRPC 服务（%s:%d）。\n"
            "请先在 polymetis-local 环境中执行：\n"
            "  launch_robot.py robot_client=franka_hardware\n"
            "  launch_gripper.py gripper=franka_hand",
            robot_ip,
            robot_port,
        )
        raise RuntimeError(f"Polymetis 服务不可达（{robot_ip}:{robot_port}）") from last_exc

    def robot_get_joint_positions(self) -> list:
        return self.robot.get_joint_positions().numpy().tolist()

    def robot_get_joint_velocities(self) -> list:
        return self.robot.get_joint_velocities().numpy().tolist()

    def robot_get_ee_pose(self) -> list:
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()

    def robot_move_to_joint_positions(
        self,
        positions: list,
        time_to_go: float = None,
        delta: bool = False,
        Kq: list = None,
        Kqd: list = None,
    ):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go,
            delta=delta,
            Kq=torch.Tensor(Kq) if Kq is not None else None,
            Kqd=torch.Tensor(Kqd) if Kqd is not None else None,
        )

    def robot_go_home(self):
        self.robot.go_home()

    def robot_move_to_ee_pose(
        self,
        pose: list = None,
        time_to_go: float = None,
        delta: bool = False,
        Kx: list = None,
        Kxd: list = None,
        op_space_interp: bool = True,
    ):
        pose = torch.Tensor(pose)
        self.robot.move_to_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat()),
            time_to_go=time_to_go,
            delta=delta,
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
            op_space_interp=op_space_interp,
        )

    def robot_start_joint_impedance_control(self, Kq: list = None, Kqd: list = None, adaptive=True):
        self.robot.start_joint_impedance(
            Kq=torch.Tensor(Kq) if Kq is not None else None,
            Kqd=torch.Tensor(Kqd) if Kqd is not None else None,
            adaptive=adaptive,
        )

    def robot_start_cartesian_impedance_control(self, Kx: list = None, Kxd: list = None):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx) if Kx is not None else None,
            Kxd=torch.Tensor(Kxd) if Kxd is not None else None,
        )

    def robot_update_desired_joint_positions(self, positions: np.ndarray):
        self.robot.update_desired_joint_positions(positions=torch.Tensor(positions))

    def robot_update_desired_ee_pose(self, pose: list):
        pose = torch.Tensor(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat()),
        )

    def robot_terminate_current_policy(self):
        self.robot.terminate_current_policy()

    def gripper_get_state(self) -> dict:
        """获取夹爪当前状态，返回 dict（zerorpc 无法序列化 protobuf）"""
        state = self.gripper.get_state()
        return {
            "width": state.width,
            "is_grasped": state.is_grasped,
            "is_moving": state.is_moving,
            "prev_command_successful": state.prev_command_successful,
            "error_code": state.error_code,
        }

    def gripper_goto(self, width: float, speed: float, force: float, blocking: bool = True):
        self.gripper.goto(width=width, speed=speed, force=force, blocking=blocking)

    def gripper_grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
        blocking: bool = True,
    ):
        self.gripper.grasp(
            speed=speed,
            force=force,
            grasp_width=grasp_width,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
            blocking=blocking,
        )

def _parse_args() -> argparse.Namespace:
    # 优先读取与 launch_franka_servers.sh 相同的环境变量，保持默认值一致
    _robot_ip = os.environ.get("ROBOT_IP", "localhost")
    _robot_port = int(os.environ.get("ROBOT_PORT", "50051"))
    _gripper_ip = os.environ.get("GRIPPER_IP", _robot_ip)
    _gripper_port = int(os.environ.get("GRIPPER_PORT", "50052"))
    _bind = os.environ.get("FRANKA_BIND", "192.168.172.134")
    _port = int(os.environ.get("FRANKA_PORT", "4242"))

    p = argparse.ArgumentParser(description="Franka zerorpc 服务端（运行于 NUC）")
    p.add_argument("--bind", default=_bind, help=f"zerorpc 绑定地址（默认 {_bind}，可用 FRANKA_BIND 覆盖）")
    p.add_argument("--port", type=int, default=_port, help=f"zerorpc 端口（默认 {_port}，可用 FRANKA_PORT 覆盖）")
    p.add_argument("--robot-ip", default=_robot_ip, dest="robot_ip", help=f"Polymetis robot server IP（默认 {_robot_ip}，可用 ROBOT_IP 覆盖）")
    p.add_argument("--robot-port", type=int, default=_robot_port, dest="robot_port", help=f"Polymetis robot server 端口（默认 {_robot_port}，可用 ROBOT_PORT 覆盖）")
    p.add_argument(
        "--gripper-ip",
        default=_gripper_ip,
        dest="gripper_ip",
        help=f"Polymetis gripper server IP（默认 {_gripper_ip}，可用 GRIPPER_IP 覆盖）",
    )
    p.add_argument("--gripper-port", type=int, default=_gripper_port, dest="gripper_port", help=f"Polymetis gripper server 端口（默认 {_gripper_port}，可用 GRIPPER_PORT 覆盖）")
    p.add_argument("--go-home", default=True, dest="go_home", help="启动后执行 go_home")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = _parse_args()

    server = FrankaInterfaceServer(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        gripper_ip=args.gripper_ip,
        gripper_port=args.gripper_port,
    )
    print(server.robot_get_ee_pose())
    print(server.robot_get_joint_positions())
    server.robot_go_home()
    if args.go_home:
        log.info("执行 go_home…")
        # server.robot_go_home()

    s = zerorpc.Server(server)
    bind_addr = f"tcp://{args.bind}:{args.port}"
    s.bind(bind_addr)
    log.info("zerorpc 服务已绑定 %s，等待客户端连接…", bind_addr)
    s.run()
