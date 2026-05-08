"""Franka + UMI 夹爪异步推理客户端。

继承 :class:`RobotClient` 的 gRPC 通信与线程管理，
仅覆盖观测采集（旋转向量→四元数）和动作执行（UMI 坐标变换）两处。

数据集约定
----------
**world_flange 格式（推荐，--pose-format=world_flange）**：

- ``observation.state``    : 1D ``[gripper_width]``
- ``action``               : 7D ``[pos_x_W, pos_y_W, pos_z_W, rotvec_x_W, rotvec_y_W, rotvec_z_W, gripper]``
  （sample-relative，相对推理时刻 EE 位姿；t0_mode 须为 ``per_step``）
- ``observation.umi_pose`` : 6D（仅数据集/训练期间使用，推理时不需要）
- ``observation.images.camera0`` : 480×640×3 RGB

**legacy ee_at_t0_flange 格式（向后兼容）**：

- ``observation.state`` / ``action`` : 8D ``[pos_xyz, quat_xyzw, gripper]``

与 robot_client.py 的差异
--------------------------
1. 机器人固定为 :class:`FrankaGenGripper`，通过 ``robot.get_umi_observation()`` 获取数据集格式观测；
2. lerobot_features 由 ``robot.umi_observation_features`` 提供，无需手动构造；
3. 动作执行使用 ``robot.send_umi_action()``，内含 UMI SE(3) 坐标变换；
4. 维护 T_BE_t0 状态（连接时捕获一次，可选每块更新）。

用法示例
--------
启动策略推理服务端::

    cd src/lerobot/async_inference
    python policy_server.py --host=0.0.0.0 --port=8080 --fps=15

运行本客户端::

    python franka_umi_async_client.py \\
        --pretrained_name_or_path /path/to/act_ckpt \\
        --server_address 127.0.0.1:8080 \\
        --robot.robot_ip 192.168.1.104 \\
        --reference camera_t0 \\
        --use_gripper

空跑（仅验证连接，不执行运动）::

    python franka_umi_async_client.py \\
        --pretrained_name_or_path /path/to/act_ckpt \\
        --dry_run
"""

import logging
import pickle  # nosec
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Optional

import draccus
import grpc
import numpy as np

from lerobot.robots.franka_gen_gripper import (
    FrankaGenGripper,
    FrankaGenGripperConfig,
    UMI_REFERENCE_MODES,
    pose_xyz_quat_to_se3,
    pose_xyz_rotvec_to_se3,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options

from configs import get_aggregate_function
from helpers import (
    FPSTracker,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)
from robot_client import RobotClient

logger = logging.getLogger("franka_umi_async_client")


# -----------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------

@dataclass
class FrankaUMIClientConfig:
    """Franka + UMI 异步推理客户端的完整配置。

    与 :class:`RobotClientConfig` 对齐命名，扩展 UMI 特有字段。
    """

    # ---- 策略 ----
    pretrained_name_or_path: str = field(metadata={"help": "策略检查点目录或 HF Hub 路径"})
    policy_type: str = field(default="act", metadata={"help": "策略类型"})
    policy_device: str = field(default="cuda", metadata={"help": "推理设备（cuda/cpu/mps）"})
    actions_per_chunk: int = field(default=50, metadata={"help": "每块最大动作数"})

    # ---- 服务端 ----
    server_address: str = field(default="127.0.0.1:8081", metadata={"help": "策略服务端 host:port"})

    # ---- 机器人（嵌套 dataclass，CLI 以 --robot.xxx 形式传入）----
    robot: FrankaGenGripperConfig = field(default_factory=FrankaGenGripperConfig)

    # ---- UMI 坐标变换 ----
    reference: str = field(
        default="ee_at_t0",
        metadata={"help": f"ee6d 参考坐标系约定，可选：{UMI_REFERENCE_MODES}"},
    )
    use_gripper: bool = field(default=True, metadata={"help": "是否执行策略输出的夹爪动作"})
    t0_mode: str = field(
        default="per_step",
        metadata={
            "help": (
                "T_BE_t₀ 更新策略：'per_step'（每步刷新，推荐与 world_flange 数据集配合使用）、"
                "'per_chunk'（每块刷新）、'per_episode'（仅 episode 开始时捕获一次）"
            )
        },
    )
    update_t0_per_chunk: bool = field(
        default=False,
        metadata={"help": "[已废弃] 请改用 t0_mode='per_chunk'。保留用于向后兼容。"},
    )

    # ---- 控制 ----
    fps: float = field(default=15.0, metadata={"help": "控制循环目标帧率"})
    chunk_size_threshold: float = field(
        default=0.2,
        metadata={"help": "队列剩余动作比例低于此值时触发新观测请求"},
    )
    task: str = field(default="pick and place", metadata={"help": "传递给策略的任务描述"})

    # ---- 调试 ----
    dry_run: bool = field(default=False, metadata={"help": "仅验证连接，不执行运动"})
    yes: bool = field(default=False, metadata={"help": "跳过运动前确认提示"})
    debug_visualize_queue_size: bool = field(
        default=False,
        metadata={"help": "退出后绘制动作队列大小曲线"},
    )
    verbose: bool = field(default=False)

    def __post_init__(self):
        self.environment_dt: float = 1.0 / self.fps
        # RobotClient._aggregate_action_queues 需要此字段
        self.aggregate_fn = get_aggregate_function("latest_only")
        # RobotClient.receive_actions 需要此字段
        self.client_device: str = "cpu"


# -----------------------------------------------------------------------
# 客户端
# -----------------------------------------------------------------------

class FrankaUMIClient(RobotClient):
    """Franka + UMI 的异步推理客户端，继承 RobotClient 的通信与线程逻辑。

    仅覆盖三处：
    - ``__init__``                 — 直接实例化 FrankaGenGripper 并捕获 T_BE_t0
    - ``control_loop_observation`` — 使用 get_umi_observation() 替代 get_observation()
    - ``control_loop_action``      — 应用 UMI SE(3) 变换后执行动作
    - ``receive_actions``          — 可选在收到新块时更新 T_BE_t0
    - ``stop``                     — 额外调用 terminate_policy()
    """

    prefix = "franka_umi_async_client"
    logger = get_logger("franka_umi_async_client")

    def __init__(self, cfg: FrankaUMIClientConfig):
        # 不调用 RobotClient.__init__（其内部使用 make_robot_from_config），
        # 手动复制所有线程状态初始化逻辑。
        self.config = cfg

        # 机器人连接
        self.robot: FrankaGenGripper = FrankaGenGripper(cfg.robot)
        self.robot.connect()

        # 稳定后捕获初始 t₀
        time.sleep(0.3)
        self.robot.capture_t0()

        # gRPC 通道与 stub
        self.server_address = cfg.server_address
        self.channel = grpc.insecure_channel(
            cfg.server_address,
            grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # 策略配置（特征由 robot.umi_observation_features 提供）
        self.policy_config = RemotePolicyConfig(
            policy_type=cfg.policy_type,
            pretrained_name_or_path=cfg.pretrained_name_or_path,
            lerobot_features=self.robot.umi_observation_features,
            actions_per_chunk=cfg.actions_per_chunk,
            device=cfg.policy_device,
        )

        # 线程状态（与 RobotClient 完全一致）
        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self._chunk_size_threshold = cfg.chunk_size_threshold
        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.start_barrier = threading.Barrier(2)
        self.fps_tracker = FPSTracker(target_fps=cfg.fps)
        self.must_go = threading.Event()
        self.must_go.set()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """停止控制循环：先终止控制器，再断连，再关 gRPC 通道。"""
        self.shutdown_event.set()
        try:
            self.robot.terminate_policy()
        except Exception as e:
            self.logger.warning(f"terminate_policy 失败：{e}")
        try:
            self.robot.disconnect()
        except Exception as e:
            self.logger.warning(f"机器人断连失败：{e}")
        self.channel.close()
        self.logger.info("客户端已停止。")

    # ------------------------------------------------------------------
    # 覆盖：观测采集（数据集格式）
    # ------------------------------------------------------------------

    def control_loop_observation(self, task: str, verbose: bool = False):
        """覆盖：通过 get_umi_observation() 获取四元数格式观测。"""
        try:
            raw_obs = self.robot.get_umi_observation()
            raw_obs["task"] = task

            with self.latest_action_lock:
                latest = self.latest_action

            obs = TimedObservation(
                timestamp=time.time(),
                observation=raw_obs,
                timestep=max(latest, 0),
            )

            with self.action_queue_lock:
                obs.must_go = self.must_go.is_set() and self.action_queue.empty()

            self.send_observation(obs)

            if obs.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(obs.get_timestamp())
                self.logger.info(
                    f"Obs #{obs.get_timestep()} | "
                    f"均值FPS={fps_metrics['avg_fps']:.2f}/{fps_metrics['target_fps']:.2f}"
                )

            return raw_obs

        except Exception as e:
            self.logger.error(f"观测采集失败：{e}")

    # ------------------------------------------------------------------
    # 覆盖：动作执行（UMI 坐标变换）
    # ------------------------------------------------------------------

    def control_loop_action(self, verbose: bool = False):
        """覆盖：将策略动作经 UMI SE(3) 变换后发送到 Franka。

        自动检测动作格式：
        - 7D ``[pos3, rotvec3, gripper]`` → world_flange / sample-relative 格式
        - 8D ``[pos3, quat4, gripper]``  → legacy ee_at_t0_flange 格式
        """
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action: TimedAction = self.action_queue.get_nowait()

        a = timed_action.get_action().cpu().numpy().astype(np.float64)

        if len(a) == 7:
            # world_flange / sample-relative 格式：7D pos+rotvec+gripper
            pos, rotvec, gripper_width = a[0:3], a[3:6], float(a[6])
            delta_k = pose_xyz_rotvec_to_se3(pos, rotvec)
        else:
            # legacy 格式：8D pos+quat+gripper
            pos, quat_xyzw, gripper_width = a[0:3], a[3:7], float(a[7])
            delta_k = pose_xyz_quat_to_se3(pos, quat_xyzw)

        # per_step 模式：每次动作前更新 T_BE_t0 = 当前 FK
        t0_mode = self.config.t0_mode
        if t0_mode == "per_step":
            self.robot.capture_t0()

        T_BE_t0 = self.robot.get_t0()
        if T_BE_t0 is None:
            self.logger.warning("T_BE_t0 尚未捕获，跳过本次动作。")
            return

        grip_cmd = gripper_width if self.config.use_gripper else None

        try:
            self.robot.send_umi_action(
                delta_k=delta_k,
                T_BE_t0=T_BE_t0,
                reference=self.config.reference,
                gripper_width=grip_cmd,
            )
        except Exception as e:
            self.logger.exception(f"send_umi_action 失败：{e}")
            return

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

    # ------------------------------------------------------------------
    # 覆盖：动作接收（可选更新 T_BE_t0）
    # ------------------------------------------------------------------

    def receive_actions(self, verbose: bool = False) -> None:
        """覆盖：在收到新动作块时可选更新 T_BE_t0。"""
        self.start_barrier.wait()
        self.logger.info("动作接收线程已启动。")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec
                if not timed_actions:
                    continue

                # per_chunk 模式（或 legacy update_t0_per_chunk 选项）
                t0_mode = self.config.t0_mode
                if t0_mode == "per_chunk" or self.config.update_t0_per_chunk:
                    if self.config.update_t0_per_chunk and t0_mode != "per_chunk":
                        self.logger.warning(
                            "update_t0_per_chunk 已废弃，请改用 t0_mode='per_chunk'。"
                        )
                    self.robot.capture_t0()
                    t0 = self.robot.get_t0()
                    if t0 is not None:
                        self.logger.debug(f"per_chunk 更新 T_BE_t0: {t0[:3, 3].round(4).tolist()}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                self.must_go.set()

            except grpc.RpcError as e:
                self.logger.error(f"接收动作失败：{e}")


# -----------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------

@draccus.wrap()
def main(cfg: FrankaUMIClientConfig) -> None:
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        client = FrankaUMIClient(cfg)
    except Exception as e:
        logger.error(f"初始化失败：{e}")
        return

    if cfg.dry_run:
        logger.info("[dry-run] 仅验证连接与策略加载。")
        ok = client.start()
        client.stop()
        return

    if not client.start():
        logger.error("无法连接策略服务端，退出。")
        client.stop()
        return

    if not cfg.yes:
        logger.warning("即将开始运动，请确认工作空间已清空。")
        try:
            input(">>> 按 ENTER 开始（Ctrl-C 中止）...")
        except (EOFError, KeyboardInterrupt):
            logger.info("用户中止。")
            client.stop()
            return

    Kx = np.array(cfg.robot.Kx, dtype=float) if cfg.robot.Kx else None
    Kxd = np.array(cfg.robot.Kxd, dtype=float) if cfg.robot.Kxd else None
    client.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

    action_thread = threading.Thread(
        target=client.receive_actions, daemon=True, name="action-receiver"
    )
    action_thread.start()

    try:
        client.control_loop(task=cfg.task)
    except KeyboardInterrupt:
        logger.info("用户中断，停止运动。")
    finally:
        client.stop()
        action_thread.join(timeout=2.0)
        if cfg.debug_visualize_queue_size and client.action_queue_size:
            visualize_action_queue_size(client.action_queue_size)


if __name__ == "__main__":
    main()
