"""Franka + UMI 夹爪异步推理客户端。

本脚本专为 :class:`FrankaGenGripper` 机器人与 ACT（或其他动作分块策略）的
在线异步推理设计，适配以 ``pick_and_place_20260507`` 数据集格式训练的模型。

数据集约定
----------
- ``observation.state``：8 维 ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]``
  （相机/EE 坐标系下的绝对姿态 + 四元数 + 夹爪开度）
- ``action``：同格式 8 维 UMI ee6d 增量（相对 t₀ 的 SE(3) 变换 + 夹爪）
- ``observation.images.camera0``：480×640×3 RGB

与通用 robot_client.py 的差异
------------------------------
1. 观测格式转换：机器人返回旋转向量格式，推理前需转为四元数格式；
2. 动作执行：策略输出 8D ee6d 增量，需经 UMI 坐标变换后调用
   ``FrankaGenGripper.send_umi_action()``；
3. ``lerobot_features`` 手动对齐数据集格式，而非从机器人特征自动派生；
4. 在控制循环开始前捕获 ``T_BE_t0``（或每个动作块重新锚定）。

用法示例
--------
启动策略推理服务端（在独立进程/机器运行）::

    cd src/lerobot/async_inference
    python policy_server.py \\
        --host=0.0.0.0 \\
        --port=8080 \\
        --fps=15 \\
        --inference_latency=0.066

运行本客户端（在机器人侧运行）::

    python -m lerobot.scripts.franka_umi_async_client \\
        --pretrained-path /path/to/act_checkpoint \\
        --server-address 127.0.0.1:8080 \\
        --robot-ip 192.168.1.104 \\
        --reference ee_at_t0 \\
        --use-gripper \\
        --fps 15

空跑模式（不连接机器人，仅验证连接与配置）::

    python -m lerobot.scripts.franka_umi_async_client \\
        --pretrained-path /path/to/act_checkpoint \\
        --server-address 127.0.0.1:8080 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import pickle  # nosec
import sys
import threading
import time
from queue import Queue

import grpc
import numpy as np
import torch

from lerobot.robots.franka_gen_gripper import (
    UMI_REFERENCE_MODES,
    FrankaGenGripper,
    FrankaGenGripperConfig,
    pose_xyz_quat_to_se3,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from constants import SUPPORTED_POLICIES
from helpers import (
    FPSTracker,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    visualize_action_queue_size,
)

# -----------------------------------------------------------------------
# 数据集特征定义（与 pick_and_place_20260507 对齐）
# -----------------------------------------------------------------------

# 状态/动作空间的 8 维特征名与数据集 info.json 一致
_DATASET_STATE_NAMES = [
    "pos_x", "pos_y", "pos_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "gripper",
]

# 手动构造 lerobot_features，使策略服务端能正确解析观测
def _build_lerobot_features(
    camera_height: int = 480,
    camera_width: int = 640,
) -> dict:
    """构造与数据集对齐的 lerobot_features 字典。"""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": _DATASET_STATE_NAMES,
        },
        "observation.images.camera0": {
            "dtype": "image",
            "shape": (camera_height, camera_width, 3),
            "names": ["height", "width", "channels"],
        },
    }


logger = logging.getLogger("franka_umi_async_client")


# -----------------------------------------------------------------------
# 观测格式转换
# -----------------------------------------------------------------------

def robot_obs_to_dataset_obs(raw_obs: dict) -> dict:
    """将机器人原始观测转换为数据集格式观测字典。

    机器人返回::

        {ee_pose.x, ee_pose.y, ee_pose.z, ee_pose.rx, ee_pose.ry, ee_pose.rz,
         gripper.pos, camera0: np.ndarray(H, W, 3)}

    数据集格式（策略期望）::

        {pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper,
         camera0: np.ndarray(H, W, 3)}

    旋转向量 → 四元数通过 scipy.spatial.transform.Rotation 完成。
    """
    from scipy.spatial.transform import Rotation

    pos = np.array([raw_obs["ee_pose.x"], raw_obs["ee_pose.y"], raw_obs["ee_pose.z"]])
    rotvec = np.array([raw_obs["ee_pose.rx"], raw_obs["ee_pose.ry"], raw_obs["ee_pose.rz"]])
    # scipy 默认 xyzw 顺序
    quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()

    obs = {
        "pos_x": float(pos[0]),
        "pos_y": float(pos[1]),
        "pos_z": float(pos[2]),
        "quat_x": float(quat_xyzw[0]),
        "quat_y": float(quat_xyzw[1]),
        "quat_z": float(quat_xyzw[2]),
        "quat_w": float(quat_xyzw[3]),
        "gripper": float(raw_obs["gripper.pos"]),
        "camera0": raw_obs["camera0"],  # 直接透传图像
    }
    return obs


# -----------------------------------------------------------------------
# 动作解析与执行
# -----------------------------------------------------------------------

def parse_action_tensor(action_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """将策略输出的 8D 动作张量解析为 SE(3) 增量矩阵与夹爪宽度。

    张量布局：``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]``
    """
    a = action_tensor.cpu().numpy().astype(np.float64)
    pos = a[0:3]
    quat_xyzw = a[3:7]
    gripper = float(a[7])
    delta_k = pose_xyz_quat_to_se3(pos, quat_xyzw)
    return delta_k, gripper


# -----------------------------------------------------------------------
# 异步客户端主类
# -----------------------------------------------------------------------

class FrankaUMIAsyncClient:
    """Franka + UMI 夹爪的异步推理控制客户端。

    架构
    ----
    - 主线程（控制循环）：以目标 FPS 采集观测并发送至策略服务端；
      队列中有可用动作时立即执行动作。
    - 子线程（动作接收）：持续从服务端拉取动作块并写入本地队列。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.shutdown_event = threading.Event()

        # ---- 机器人 ----
        config = FrankaGenGripperConfig(
            id="umi_async",
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            gripper_side=args.gripper_side,
            gripper_serial_port=args.gripper_serial_port,
            gripper_camera_count=1,
            gripper_camera_width=640,
            gripper_camera_height=480,
            gripper_enable_tactile=False,
            camera_extrinsic_yaml_path=args.camera_extrinsic_yaml,
        )
        self.robot = FrankaGenGripper(config)

        # ---- gRPC ----
        self.channel = grpc.insecure_channel(
            args.server_address,
            grpc_channel_options(initial_backoff=f"{1.0 / args.fps:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # ---- 策略配置（手动构造与数据集对齐的特征）----
        lerobot_features = _build_lerobot_features(
            camera_height=480,
            camera_width=640,
        )
        self.policy_config = RemotePolicyConfig(
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=args.actions_per_chunk,
            device=args.policy_device,
            rename_map={},
        )

        # ---- 动作队列 ----
        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size_log: list[int] = []

        self.latest_action_lock = threading.Lock()
        self.latest_action: int = -1
        self.action_chunk_size: int = -1

        self.must_go = threading.Event()
        self.must_go.set()

        # ---- 同步屏障（控制循环 + 动作接收线程同步启动）----
        self.start_barrier = threading.Barrier(2)

        # ---- FPS 追踪 ----
        self.fps_tracker = FPSTracker(target_fps=args.fps)

        # ---- UMI t₀ 法兰位姿（连接后捕获）----
        self.T_BE_t0: np.ndarray | None = None
        self._t0_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def connect(self) -> None:
        """连接机器人并完成策略服务端握手。"""
        logger.info("正在连接 Franka + UMI 夹爪系统 ...")
        self.robot.connect()

        # 预加载手眼外参，确保 YAML 有效
        _ = self.robot.umi_camera_extrinsic

        # 捕获初始 t₀ 法兰位姿
        time.sleep(0.3)
        with self._t0_lock:
            self.T_BE_t0 = self.robot.get_ee_se3()
        logger.info(f"ᴮT_E(t₀): {self.T_BE_t0[:3, 3].round(4).tolist()}")

    def start_policy(self) -> bool:
        """向策略服务端发送握手和策略指令，返回是否成功。"""
        try:
            self.stub.Ready(services_pb2.Empty())
            policy_bytes = pickle.dumps(self.policy_config)  # nosec
            self.stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=policy_bytes)
            )
            self.shutdown_event.clear()
            logger.info("策略服务端握手成功，推理就绪。")
            return True
        except grpc.RpcError as e:
            logger.error(f"策略服务端连接失败：{e}")
            return False

    def stop(self) -> None:
        """停止控制循环并断开连接。"""
        self.shutdown_event.set()
        try:
            self.robot.terminate_policy()
        except Exception as e:
            logger.warning(f"terminate_policy 失败：{e}")
        try:
            self.robot.disconnect()
        except Exception as e:
            logger.warning(f"机器人断连失败：{e}")
        self.channel.close()
        logger.info("客户端已停止。")

    # ------------------------------------------------------------------
    # 观测采集与发送
    # ------------------------------------------------------------------

    def _capture_observation(self, task: str) -> tuple[TimedObservation, dict]:
        """采集当前机器人状态，转换为数据集格式并包装为 TimedObservation。

        同时返回原始观测字典，供调试使用。
        """
        raw_obs = self.robot.get_observation()
        raw_obs["task"] = task

        dataset_obs = robot_obs_to_dataset_obs(raw_obs)
        dataset_obs["task"] = task

        with self.latest_action_lock:
            timestep = max(self.latest_action, 0)

        obs = TimedObservation(
            timestamp=time.time(),
            observation=dataset_obs,
            timestep=timestep,
        )

        with self.action_queue_lock:
            obs.must_go = self.must_go.is_set() and self.action_queue.empty()

        return obs, raw_obs

    def _send_observation(self, obs: TimedObservation) -> bool:
        """通过 gRPC 将 TimedObservation 发送至策略服务端。"""
        obs_bytes = pickle.dumps(obs)  # nosec
        try:
            iterator = send_bytes_in_chunks(
                obs_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] 观测",
                silent=True,
            )
            self.stub.SendObservations(iterator)
            return True
        except grpc.RpcError as e:
            logger.error(f"发送观测失败：{e}")
            return False

    # ------------------------------------------------------------------
    # 动作接收（子线程）
    # ------------------------------------------------------------------

    def receive_actions(self) -> None:
        """持续从策略服务端拉取动作块并更新本地队列（子线程）。"""
        self.start_barrier.wait()
        logger.info("动作接收线程已启动。")

        while self.running:
            try:
                response = self.stub.GetActions(services_pb2.Empty())
                if len(response.data) == 0:
                    continue  # 服务端暂无推理结果，继续等待

                timed_actions: list[TimedAction] = pickle.loads(response.data)  # nosec

                if not timed_actions:
                    continue

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # 更新 t₀（每个动作块重新锚定当前法兰位姿）
                if self.args.update_t0_per_chunk:
                    with self._t0_lock:
                        self.T_BE_t0 = self.robot.get_ee_se3()
                    logger.debug(f"更新 T_BE_t0: {self.T_BE_t0[:3, 3].round(4).tolist()}")

                # 写入队列（丢弃已过期动作）
                with self.latest_action_lock:
                    latest = self.latest_action

                accepted = 0
                with self.action_queue_lock:
                    for ta in timed_actions:
                        if ta.get_timestep() > latest:
                            self.action_queue.put(ta)
                            accepted += 1

                logger.debug(
                    f"收到动作块：{len(timed_actions)} 个，接受 {accepted} 个，"
                    f"队列长度：{self.action_queue.qsize()}"
                )

                self.must_go.set()

            except grpc.RpcError as e:
                logger.error(f"接收动作失败：{e}")
                time.sleep(0.01)

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------

    def _execute_next_action(self) -> bool:
        """从队列弹出一个动作并发送给机器人，返回是否成功。"""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return False
            self.action_queue_size_log.append(self.action_queue.qsize())
            timed_action: TimedAction = self.action_queue.get_nowait()

        action_tensor: torch.Tensor = timed_action.get_action()
        delta_k, gripper_width = parse_action_tensor(action_tensor)

        with self._t0_lock:
            T_BE_t0 = self.T_BE_t0

        if T_BE_t0 is None:
            logger.warning("T_BE_t0 尚未捕获，跳过本次动作执行。")
            return False

        grip_cmd = gripper_width if self.args.use_gripper else None

        try:
            self.robot.send_umi_action(
                delta_k=delta_k,
                T_BE_t0=T_BE_t0,
                reference=self.args.reference,
                gripper_width=grip_cmd,
            )
        except Exception as e:
            logger.exception(f"send_umi_action 失败：{e}")
            return False

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        return True

    # ------------------------------------------------------------------
    # 控制循环（主线程）
    # ------------------------------------------------------------------

    def _actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self) -> bool:
        """当队列中剩余动作比例低于阈值时触发新的观测请求。"""
        if self.action_chunk_size <= 0:
            return True
        with self.action_queue_lock:
            ratio = self.action_queue.qsize() / self.action_chunk_size
        return ratio <= self.args.chunk_size_threshold

    def control_loop(self, task: str) -> None:
        """主控制循环：采集观测 → 发送服务端；执行队列动作。"""
        self.start_barrier.wait()
        logger.info("控制循环已启动。")

        dt = 1.0 / self.args.fps

        while self.running:
            t_start = time.perf_counter()

            # ---- 执行动作 ----
            if self._actions_available():
                self._execute_next_action()

            # ---- 发送观测 ----
            if self._ready_to_send_observation():
                obs, _ = self._capture_observation(task)
                self._send_observation(obs)

                if obs.must_go:
                    self.must_go.clear()

                fps_metrics = self.fps_tracker.calculate_fps_metrics(obs.get_timestamp())
                logger.debug(
                    f"Obs #{obs.get_timestep()} | "
                    f"均值FPS={fps_metrics['avg_fps']:.2f}/{fps_metrics['target_fps']:.2f}"
                )

            elapsed = time.perf_counter() - t_start
            sleep_for = dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Franka + UMI 夹爪异步推理控制客户端（ACT / 动作分块策略）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 策略 ----
    p.add_argument(
        "--pretrained-path", required=True,
        help="策略检查点目录或 HuggingFace Hub 路径（如 /path/to/act_ckpt）。",
    )
    p.add_argument(
        "--policy-type", default="act", choices=list(SUPPORTED_POLICIES),
        help="策略类型。",
    )
    p.add_argument(
        "--policy-device", default="cuda",
        help="策略推理设备（cuda / cpu / mps）。",
    )
    p.add_argument(
        "--actions-per-chunk", type=int, default=50,
        help="每个动作块的最大动作数（不超过策略 chunk_size）。",
    )

    # ---- 服务端 ----
    p.add_argument("--server-address", default="127.0.0.1:8080",
                   help="策略推理服务端地址 host:port。")

    # ---- 机器人 ----
    p.add_argument("--robot-ip", default="172.20.10.2")
    p.add_argument("--robot-port", type=int, default=4242)
    p.add_argument("--gripper-side", default="left", choices=["left", "right"])
    p.add_argument("--gripper-serial-port", default=None)
    p.add_argument(
        "--camera-extrinsic-yaml",
        default="/home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml",
        help="手眼标定 YAML（parent=panda_link8, child=camera）。",
    )

    # ---- UMI 坐标变换 ----
    p.add_argument(
        "--reference", default="ee_at_t0", choices=list(UMI_REFERENCE_MODES),
        help="数据集 ee6d 动作的参考坐标系约定。",
    )
    p.add_argument(
        "--update-t0-per-chunk", action="store_true",
        help="每收到一个新动作块时重新捕获 T_BE_t0（适合长任务；默认只在启动时捕获一次）。",
    )

    # ---- 夹爪 ----
    p.add_argument(
        "--use-gripper", action="store_true",
        help="执行策略输出的夹爪宽度（默认不控制夹爪）。",
    )

    # ---- 控制参数 ----
    p.add_argument("--fps", type=float, default=15.0, help="控制循环目标帧率。")
    p.add_argument(
        "--chunk-size-threshold", type=float, default=0.5,
        help="队列中剩余动作比例低于此值时触发新观测请求。",
    )
    p.add_argument("--task", default="pick and place",
                   help="传递给策略的任务描述（VLA 类策略使用）。")

    # ---- 阻抗控制增益（可选覆盖）----
    p.add_argument(
        "--Kx", type=float, nargs=6, default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="笛卡尔阻抗刚度（6 个浮点数）。",
    )
    p.add_argument(
        "--Kxd", type=float, nargs=6, default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="笛卡尔阻抗阻尼（6 个浮点数）。",
    )

    # ---- 调试 ----
    p.add_argument("--dry-run", action="store_true",
                   help="仅验证连接与策略加载，不执行运动。")
    p.add_argument("--yes", "-y", action="store_true",
                   help="跳过运动前确认提示。")
    p.add_argument("--visualize-queue", action="store_true",
                   help="退出后显示动作队列大小曲线。")
    p.add_argument("--verbose", "-v", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        logger.info("[dry-run] 仅连接验证，不执行运动。")
        client = FrankaUMIAsyncClient(args)
        client.connect()
        ok = client.start_policy()
        client.stop()
        return 0 if ok else 1

    client = FrankaUMIAsyncClient(args)

    try:
        client.connect()
    except Exception as e:
        logger.error(f"机器人连接失败：{e}")
        return 1

    if not client.start_policy():
        logger.error("无法连接策略服务端，退出。")
        client.robot.disconnect()
        return 1

    if not args.yes:
        logger.warning("即将开始运动，请确认工作空间已清空。")
        try:
            input(">>> 按 ENTER 开始（Ctrl-C 中止）...")
        except (EOFError, KeyboardInterrupt):
            logger.info("用户中止。")
            client.stop()
            return 130

    # 启动笛卡尔阻抗控制器
    Kx = np.array(args.Kx, dtype=float) if args.Kx else None
    Kxd = np.array(args.Kxd, dtype=float) if args.Kxd else None
    client.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

    # 启动动作接收子线程
    action_thread = threading.Thread(
        target=client.receive_actions, daemon=True, name="action-receiver"
    )
    action_thread.start()

    try:
        client.control_loop(task=args.task)
    except KeyboardInterrupt:
        logger.info("用户中断，停止运动。")
    finally:
        client.stop()
        action_thread.join(timeout=2.0)
        if args.visualize_queue and client.action_queue_size_log:
            visualize_action_queue_size(client.action_queue_size_log)

    return 0


if __name__ == "__main__":
    sys.exit(main())
