#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.datasets.repair import repair_dataset_consistency
from lerobot.datasets.utils import DEFAULT_SUBTASKS_PATH
from lerobot.rlt.intervention import start_key_backend
from lerobot.rlt.piper_env import (
    JOINT_ORDER,
    build_piper_cameras,
    follower_action_to_leader,
    leader_action_to_follower,
    rate_limit_joints,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.status_view import StatusView

logger = logging.getLogger(__name__)

DOF_PER_ARM = len(JOINT_ORDER)  # 6 关节 + 夹爪
INIT_MODES = ("follower_to_leader", "home", "none")


# ===================================================================== 配置
@dataclass
class ArmSpec:
    """一条「主臂 + 从臂」。CAN 口名以 `ip -br link show | grep can` 的实际输出为准。"""

    name: str = "left"
    leader_port: str = "can_left_l"
    follower_port: str = "can_left_f"

    max_takeover_delta_rad: float = 0.15
    use_calibrated_offsets: bool = False
    require_calibration: bool = False

    gravity_comp_tx_ratio: list[float] = field(default_factory=lambda: [0.2] * 6)
    gravity_comp_base_rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gravity_comp_torque_limit: float = 8.0
    gravity_comp_mit_kp: float = 0.0
    gravity_comp_mit_kd: float = 0.0
    gravity_comp_control_hz: float = 200.0
    gravity_comp_urdf: str | None = None
    gravity_comp_payload_mass: float = 0.0
    gravity_comp_payload_com: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class CameraSpec:

    name: str = "cam"
    index_or_path: str | None = None
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 60


@dataclass
class DatasetSpec:
    repo_id: str = "user/PiperDataset"
    root: str = "outputs/record_piper"
    task: str = "default task"
    robot_type: str = "piper"
    fps: int = 30

    resume: bool = True
    use_videos: bool = True
    video_codec: str = "libsvtav1"
    streaming_encoding: bool = True
    encoder_threads: int = 4
    data_flush_every: int = 10

    # 数字键 -> 子任务描述。留空则不记录 subtask_index。
    subtasks: dict = field(default_factory=dict)


@dataclass
class CollectionSpec:
    max_timesteps: int = 10000
    # 每拍关节推进上限（rad），整条 6 维差向量等比缩放。调太大等于关掉限速。
    max_joint_step_rad: float = 0.05
    use_back_event: bool = True
    dry_run: bool = False

    # follower_to_leader（从臂去找主臂）/ home（各自回原点）/ none
    init_mode: str = "follower_to_leader"
    init_duration_s: float = 4.0
    # 初始化平滑运动的单步上限，是「手臂不猛地弹过去」的唯一保障，别调大
    init_max_step_rad: float = 0.01
    init_settle_timeout_s: float = 10.0


@dataclass
class ViewSpec:
    enabled: bool = True
    max_width: int = 1280
    # 别跟控制频率绑一起：30Hz 下每拍 imshow 会吃掉实时预算
    render_hz: float = 15.0


@dataclass
class RecordPiperConfig:
    arms: list[ArmSpec] = field(default_factory=lambda: [ArmSpec()])
    dataset: DatasetSpec = field(default_factory=DatasetSpec)
    collection: CollectionSpec = field(default_factory=CollectionSpec)
    cameras: list[CameraSpec] = field(default_factory=list)
    view: ViewSpec = field(default_factory=ViewSpec)
    # auto / termios / pynput / none
    keyboard_backend: str = "auto"


# ================================================================= 按键状态机
class CollectKeys:
    """采集的按键状态机，键位沿用 ROS 版 data_to_lerobot3_node.py。
    不复用 `KeyboardEventListener`：它的内建绑定（s=成功 / r=交接 / space=接管）与
    采集键位语义冲突。共用的只有底层后端，见 :func:`start_key_backend`。
    """

    def __init__(self, backend: str = "auto", subtask_keys=()):
        self._impl = start_key_backend(backend)
        self._subtask_keys = set(subtask_keys)
        self.collecting = False
        self.error_mode = False
        self.quit = False
        self._save = False
        self._discard = False
        self.subtask_key = None
        self.messages: list[str] = []

    @property
    def available(self) -> bool:
        return self._impl is not None

    def poll(self) -> None:
        if self._impl is None:
            return
        for key in self._impl.poll():
            self._dispatch(key)

    def _dispatch(self, key: str) -> None:
        if key == "c":
            if not self.collecting:
                self.collecting = True
                self.messages.append("▶ 开始采集")
        elif key == "space":
            if self.collecting:
                self.collecting = False
                # error mode 不允许跨暂停泄漏：重新开始时操作员默认自己处于正常状态
                self.error_mode = False
                self.messages.append("⏸ 暂停采集")
        elif key == "e":
            self.error_mode = not self.error_mode
            self.messages.append("🔴 错误模式 开 (back_event=1)" if self.error_mode else "🟢 错误模式 关")
        elif key == "s":
            self._save = True
        elif key == "r":
            self._discard = True
        elif key in ("q", "esc"):
            self.quit = True
        elif key in self._subtask_keys:
            if self.collecting:
                self.messages.append("⚠ 采集中不能切子任务，先按 space 暂停")
            else:
                self.subtask_key = key

    def poll_save(self) -> bool:
        val, self._save = self._save, False
        return val

    def poll_discard(self) -> bool:
        val, self._discard = self._discard, False
        return val

    def poll_subtask(self):
        val, self.subtask_key = self.subtask_key, None
        return val

    def drain_messages(self) -> list[str]:
        out, self.messages = self.messages, []
        return out

    def stop(self) -> None:
        if self._impl is not None:
            self._impl.stop()
            self._impl = None


# =================================================================== 一条主从臂
class ArmPair:
    """「主臂 + 从臂」。"""

    def __init__(self, spec: ArmSpec):
        from lerobot.robots.piper.config_piper import PIPERConfig
        from lerobot.robots.piper.piper import Piper
        from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

        self.spec = spec

        self.robot_cfg = PIPERConfig(can_port=spec.follower_port)
        self.robot_cfg.cameras = {}
        self.robot = Piper(self.robot_cfg)

        self.leader = PiperLeader(
            PiperLeaderConfig(
                port=spec.leader_port,
                id=f"piper_leader_{spec.name}",
                require_calibration=spec.require_calibration,
                gravity_comp_tx_ratio=tuple(spec.gravity_comp_tx_ratio),
                gravity_comp_base_rpy_deg=tuple(spec.gravity_comp_base_rpy_deg),
                gravity_comp_torque_limit=spec.gravity_comp_torque_limit,
                gravity_comp_mit_kp=spec.gravity_comp_mit_kp,
                gravity_comp_mit_kd=spec.gravity_comp_mit_kd,
                gravity_comp_control_hz=spec.gravity_comp_control_hz,
                gravity_comp_urdf=spec.gravity_comp_urdf,
                gravity_comp_payload_mass=spec.gravity_comp_payload_mass,
                gravity_comp_payload_com=tuple(spec.gravity_comp_payload_com),
            )
        )

    def connect(self) -> None:
        if not self.robot.is_connected:
            self.robot.connect(calibrate=False)
        self.leader.connect()

    def disconnect(self) -> None:
        if self.leader is not None and getattr(self.leader, "is_connected", False):
            try:
                self.leader.set_manual_control(False)
            except Exception:
                logger.exception("[%s] 主臂交回位置环失败", self.spec.name)
            try:
                self.leader.disconnect()
            except Exception:
                logger.exception("[%s] 主臂断开失败", self.spec.name)
        if self.robot is not None:
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            except Exception:
                logger.exception("[%s] 从臂断开失败", self.spec.name)

    def read_leader(self) -> dict:
        raw = self.leader.get_action() if self.spec.use_calibrated_offsets else self.leader.get_raw_action()
        return leader_action_to_follower(raw)

    def read_follower(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[k] for k in JOINT_ORDER], dtype=np.float32)

    def send(self, joints: np.ndarray) -> None:
        self.robot.send_action(dict(zip(JOINT_ORDER, joints.tolist(), strict=True)))

    def takeover_delta(self) -> float:
        """主从最大关节差。夹爪是开度不是角度，不参与。"""
        leader = np.array([self.read_leader()[k] for k in JOINT_ORDER], dtype=np.float32)
        follower = self.read_follower()
        return float(np.abs(leader[:6] - follower[:6]).max())

    def initialize(self, cfg: CollectionSpec) -> None:
        """对齐主从并释放主臂。
        方向和 `teleop_check.py` 相反：那里是主臂去找从臂（因为要反复交还），这里是
        从臂去找主臂 —— 人已经握着主臂摆好了起始姿态，不该让主臂自己跑掉。
        """
        mode = cfg.init_mode
        if mode not in INIT_MODES:
            raise ValueError(f"未知的 init_mode {mode!r}；可选 {INIT_MODES}")

        if mode == "none":
            self.leader.set_manual_control(True)
            return

        if mode == "home":
            home = list(self.robot_cfg.home_position)
            self.leader.send_feedback(follower_action_to_leader(dict(zip(JOINT_ORDER, home, strict=True))))
            self._wait_leader_settle(np.array(home, dtype=np.float32), cfg)
            self.leader.set_manual_control(True)
            target = home
            logger.info("[%s] 初始化：主从各自回原点", self.spec.name)
        else:
            self.leader.set_manual_control(True)
            leader_action = self.read_leader()
            target = [float(leader_action[k]) for k in JOINT_ORDER]
            logger.info("[%s] 初始化：从臂运动到主臂当前位姿", self.spec.name)

        self.robot.bus.move_to_joint_smoothly(
            target,
            duration_s=cfg.init_duration_s,
            max_joint_step_rad=cfg.init_max_step_rad,
        )
        limit = self.spec.max_takeover_delta_rad
        if limit > 0:
            delta = self.takeover_delta()
            if delta > limit:
                raise RuntimeError(
                    f"[{self.spec.name}] 初始化后主从仍相差 {delta:.4f} rad（上限 {limit:.3f}）。"
                    "多半是运动途中主臂被挪了，或某个关节被限位卡住。"
                )
            logger.info("[%s] 初始化完成，主从差 %.4f rad", self.spec.name, delta)

    def _wait_leader_settle(self, target: np.ndarray, cfg: CollectionSpec) -> None:
        limit = max(self.spec.max_takeover_delta_rad, 1e-3) * 0.5
        deadline = time.perf_counter() + cfg.init_settle_timeout_s
        while time.perf_counter() < deadline:
            current = np.array([self.read_leader()[k] for k in JOINT_ORDER], dtype=np.float32)
            if float(np.abs(current[:6] - target[:6]).max()) <= limit:
                return
            time.sleep(0.02)
        logger.warning(
            "[%s] 主臂在 %.1fs 内没走到目标位姿；检查主臂是否使能，或调大 init_settle_timeout_s。",
            self.spec.name,
            cfg.init_settle_timeout_s,
        )


# ================================================================= 采集主流程
def _confirm_overwrite(target: str) -> bool:
    print("\n" + "=" * 60)
    print("  警告：数据集目录已存在：")
    print(f"    {target}")
    print("  而 dataset.resume = false。")
    print("=" * 60)
    try:
        answer = input("  覆盖（删除）已有数据集？[y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


def _local_codebase_version(root: Path) -> str:
    """从本地 info.json 读，避免续采时去访问 HuggingFace Hub。"""
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        return json.loads(info_path.read_text()).get("codebase_version", "v3.0")
    return "v3.0"


def _local_video_codec(root: Path, default: str) -> str:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return default
    info = json.loads(info_path.read_text())
    codec_to_encoder = {"av1": "libsvtav1", "h264": "h264_nvenc", "hevc": "hevc_nvenc"}
    for feat in info.get("features", {}).values():
        codec = (feat.get("info") or {}).get("video.codec")
        if codec:
            return codec_to_encoder.get(codec, codec)
    return default


class PiperRecorder:
    def __init__(self, cfg: RecordPiperConfig):
        self.cfg = cfg
        if not cfg.arms:
            raise ValueError("`arms` 不能为空：至少要配一条「主臂 + 从臂」。")
        self.dt = 1.0 / cfg.dataset.fps
        self.n_arms = len(cfg.arms)
        self.dof = DOF_PER_ARM * self.n_arms

        self.pairs = [ArmPair(spec) for spec in cfg.arms]
        self.cameras: dict = {}
        self.dataset = None
        self.view = None
        self.keys = None
        # gs_usb 写一帧 ≈ 2.3ms，一次 JointCtrl 是 3 帧 ≈ 7ms；双臂串行下发 14ms，
        # 30Hz 的 33ms 预算里去掉快一半，所以双臂并行。
        self._pool = ThreadPoolExecutor(max_workers=self.n_arms) if self.n_arms > 1 else None
        self.subtasks = {str(k): v for k, v in cfg.dataset.subtasks.items()}
        names = list(self.subtasks.values())
        if len(names) != len(set(names)):
            raise ValueError(f"子任务描述有重复：{names}")
        self.subtask_name_to_idx = {name: i for i, name in enumerate(names)}
        self.use_subtask = bool(self.subtasks)
        self.use_back_event = cfg.collection.use_back_event
        self.current_subtask = names[0] if names else None

        self._episodes_since_flush = 0
        self._saturated: list[bool] = []
        self._loop_periods: list[float] = []
        self._dropped_frames = 0

    def joint_names(self) -> list[str]:
        """单臂 joint_1..7；双臂左 joint_1..7、右 joint_8..14（与 DualPIPERConfig 一致）。"""
        return [f"joint_{i + 1}.pos" for i in range(self.dof)]

    def build_features(self) -> dict:
        cfg = self.cfg
        names = self.joint_names()
        features = {
            "observation.state": {"dtype": "float32", "shape": (self.dof,), "names": names},
            "action": {"dtype": "float32", "shape": (self.dof,), "names": names},
        }
        for cam in cfg.cameras:
            features[f"observation.images.{cam.name}"] = {
                "dtype": "video" if cfg.dataset.use_videos else "image",
                "shape": (cam.height, cam.width, 3),
                "names": ["height", "width", "channels"],
            }
        if self.use_subtask:
            features["subtask_index"] = {"dtype": "int64", "shape": (1,), "names": None}
        if self.use_back_event:
            features["back_event"] = {"dtype": "int64", "shape": (1,), "names": None}
        return features

    def connect(self) -> None:
        cfg = self.cfg
        for pair in self.pairs:
            pair.connect()
            print(
                f"[record] 已连接臂 {pair.spec.name}：主臂 {pair.spec.leader_port} / "
                f"从臂 {pair.spec.follower_port}"
            )
            if max(pair.spec.gravity_comp_tx_ratio) <= 0.25:
                print(
                    f"[record] 提示：{pair.spec.name} 的 tx_ratio 很低，只补偿约两成自重，"
                    "主臂拖起来会明显发沉。"
                )
            if pair.spec.gravity_comp_payload_mass <= 0 and not pair.spec.gravity_comp_urdf:
                print(
                    f"[record] 提示：{pair.spec.name} 的重力模型没有末端负载。主臂若装了夹爪 / "
                    "示教手柄，腕部重力矩会被系统性低估，且调 tx_ratio 补不回来。"
                )

        if cfg.cameras:
            from lerobot.cameras.utils import make_cameras_from_configs

            self.cameras = make_cameras_from_configs(build_piper_cameras(cfg.cameras, cfg.dataset.fps))
            for name, cam in self.cameras.items():
                cam.connect()
                print(f"[record] 相机 {name} 已连接")

    def initialize_arms(self) -> None:
        for pair in self.pairs:
            pair.initialize(self.cfg.collection)

    # ------------------------------------------------------------- 数据集
    def setup_dataset(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        cfg = self.cfg.dataset
        root = Path(cfg.root).expanduser()
        features = self.build_features()

        if root.exists():
            if cfg.resume:
                logger.info("续采已有数据集：%s", root)
                # 必须在构造 LeRobotDataset 之前修：数据与元数据不一致时 __init__ 会
                # 回退去访问 HuggingFace Hub，本地 repo_id 会直接挂住。
                repair_dataset_consistency(root)
                codec = _local_video_codec(root, cfg.video_codec)
                if codec != cfg.video_codec:
                    logger.warning(
                        "已有数据集用的是 %s 编码，续采时沿用它（配置写的是 %s）。要换编码器请新建数据集。",
                        codec,
                        cfg.video_codec,
                    )
                self.dataset = LeRobotDataset(
                    repo_id=cfg.repo_id,
                    root=root,
                    tolerance_s=1e-4,
                    revision=_local_codebase_version(root),
                    vcodec=codec,
                    streaming_encoding=cfg.streaming_encoding,
                    encoder_threads=cfg.encoder_threads,
                )
                # __init__ 之后 episode_buffer 是 None；add_frame 会自建，但
                # clear_episode_buffer 不会 —— 一帧还没加就按 r 丢弃会 TypeError。
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                if not cfg.streaming_encoding:
                    self.dataset.start_image_writer(
                        num_processes=0, num_threads=4 * max(1, len(self.cameras))
                    )
                self._check_resume_compatibility(features)
                print(
                    f"[record] 续采：已有 {self.dataset.meta.total_episodes} 集 / "
                    f"{self.dataset.meta.total_frames} 帧"
                )
                return

            if not _confirm_overwrite(str(root)):
                print("[record] 用户取消覆盖，退出。")
                sys.exit(0)
            logger.warning("覆盖已有数据集：%s", root)
            shutil.rmtree(root)

        self.dataset = LeRobotDataset.create(
            repo_id=cfg.repo_id,
            fps=cfg.fps,
            features=features,
            root=root,
            robot_type=cfg.robot_type,
            use_videos=cfg.use_videos,
            tolerance_s=1e-4,
            image_writer_processes=0,
            image_writer_threads=4,
            vcodec=cfg.video_codec,
            streaming_encoding=cfg.streaming_encoding,
            encoder_threads=cfg.encoder_threads,
        )
        if self.use_subtask:
            self._write_subtasks_parquet(root)
        print(f"[record] 已新建数据集：{root}（编码 {cfg.video_codec}，流式={cfg.streaming_encoding}）")

    def _check_resume_compatibility(self, features: dict) -> None:
        """续采不能凭空长出新列，已有数据集没有的字段必须关掉，否则 add_frame 会拒收。"""
        existing = self.dataset.meta.features
        if self.use_back_event and "back_event" not in existing:
            logger.warning("已有数据集没有 back_event 列，本次运行自动关闭它。")
            self.use_back_event = False
        if self.use_subtask and "subtask_index" not in existing:
            logger.warning("已有数据集没有 subtask_index 列，本次运行自动关闭它。")
            self.use_subtask = False
        extra = sorted(set(features) - set(existing) - {"subtask_index", "back_event"})
        if extra:
            logger.warning(
                "本次配置比已有数据集多出这些字段：%s。续采无法新增列 —— "
                "请改回与数据集一致的相机 / 臂数配置，或新建数据集。",
                extra,
            )

    def _write_subtasks_parquet(self, root: Path) -> None:
        """仓库的 dataset 代码只在 `__getitem__` 里读这个文件，没有任何写入方。"""
        import pandas as pd

        names = list(self.subtasks.values())
        df = pd.DataFrame(
            {"subtask_index": range(len(names))},
            index=pd.Index(names, name="subtask"),
        )
        path = root / DEFAULT_SUBTASKS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("已写入 %d 条子任务到 %s", len(df), path)

    # ------------------------------------------------------------- 每拍
    def _map(self, fn, items):
        if self._pool is None:
            return [fn(x) for x in items]
        return list(self._pool.map(fn, items))

    def step_arms(self) -> tuple[np.ndarray, np.ndarray]:
        """读主臂 + 读从臂 + 限速 + 下发，返回 (从臂实测, 实际下发的目标)。"""
        cfg = self.cfg.collection
        leader_actions = self._map(lambda p: p.read_leader(), self.pairs)
        measured_list = self._map(lambda p: p.read_follower(), self.pairs)

        limited_list = []
        for leader_action, measured in zip(leader_actions, measured_list, strict=True):
            target = np.array([leader_action[k] for k in JOINT_ORDER], dtype=np.float32)
            limited, saturated = rate_limit_joints(target, measured, cfg.max_joint_step_rad)
            limited_list.append(limited)
            self._saturated.append(bool(saturated))

        if not cfg.dry_run:
            self._map(lambda iv: iv[0].send(iv[1]), list(zip(self.pairs, limited_list, strict=True)))

        return np.concatenate(measured_list), np.concatenate(limited_list)

    def read_cameras(self) -> dict:
        images = {}
        for name, cam in self.cameras.items():
            try:
                images[name] = cam.async_read()
            except Exception as exc:
                logger.warning("相机 %s 读取失败：%s", name, exc)
        return images

    def build_frame(self, state: np.ndarray, action: np.ndarray, images: dict) -> dict:
        """拼一帧。

        `add_frame` 要求键集合恰好等于 schema，数值特征 dtype / shape 精确匹配 ——
        所以 subtask_index / back_event 是 (1,) 的 int64 数组而不是裸 int，
        `task` 是 frame 里的一个键而不是单独参数。
        """
        frame = {
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
            "task": self.cfg.dataset.task,
        }
        for cam in self.cfg.cameras:
            img = images.get(cam.name)
            if img is None:
                continue
            img = np.asarray(img)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            # lerobot 相机默认输出 RGB，正是数据集要的，不需要翻通道
            frame[f"observation.images.{cam.name}"] = img
        if self.use_subtask:
            idx = self.subtask_name_to_idx.get(self.current_subtask, 0)
            frame["subtask_index"] = np.array([idx], dtype=np.int64)
        if self.use_back_event:
            frame["back_event"] = np.array(
                [1 if self.keys is not None and self.keys.error_mode else 0], dtype=np.int64
            )
        return frame

    def buffered_frames(self) -> int:
        """`save_episode()` 会先 pop 'size'，两集之间缓冲也可能是 None，所以要防御式读。"""
        if self.dataset is None:
            return 0
        buf = getattr(self.dataset, "episode_buffer", None)
        if not buf:
            return 0
        return buf.get("size", 0)

    # ------------------------------------------------------------- episode
    def save_episode(self) -> None:
        n = self.buffered_frames()
        if n == 0:
            print("[record] 缓冲区是空的，没有东西可存。")
            return
        if n < 10:
            print(f"[record] 这一集只有 {n} 帧，确认不是误按？（按 r 可丢弃）")

        print(f"[record] 💾 正在保存 {n} 帧 …")
        self.dataset.save_episode()
        # 元数据默认攒够 10 集才落盘，不主动 flush 一次硬杀最多丢 10 集
        self.dataset.meta._flush_metadata_buffer()

        self._episodes_since_flush += 1
        if self._episodes_since_flush >= self.cfg.dataset.data_flush_every:
            self._checkpoint_data_writer()

        buf = getattr(self.dataset, "episode_buffer", None)
        if not buf or "size" not in buf:
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()

        self.reset_episode_state()
        print(
            f"[record] ✅ 已保存 episode {self.dataset.meta.total_episodes - 1}"
            f"（累计 {self.dataset.meta.total_episodes} 集 / "
            f"{self.dataset.meta.total_frames} 帧）"
        )

    def _checkpoint_data_writer(self) -> None:
        """让 parquet footer 落盘。

        LeRobot 跨集复用同一个 ParquetWriter，footer 只有 close() 时才写，在那之前
        文件读不了，进程被杀会丢掉整个文件里的所有 episode。`_writer_closed_for_reading`
        必须置上，下一集才会滚到新文件 —— 不置的话 LeRobot 会在同一路径重开 writer，
        把已写内容整个截断。
        """
        ds = self.dataset
        if ds is None or getattr(ds, "writer", None) is None:
            return
        try:
            ds._close_writer()
            ds._writer_closed_for_reading = True
            self._episodes_since_flush = 0
            print("[record] 💾 数据已落盘，下一集换新文件。")
        except Exception as exc:
            logger.warning("落盘 data writer 失败：%s", exc)

    def discard_episode(self) -> None:
        if self.buffered_frames() > 0:
            self.dataset.clear_episode_buffer()
        self.reset_episode_state()
        print("[record] 🗑 已丢弃缓冲区，子任务与错误模式已复位。")

    def reset_episode_state(self) -> None:
        if self.subtasks:
            self.current_subtask = next(iter(self.subtasks.values()))
        if self.keys is not None:
            self.keys.error_mode = False
            self.keys.collecting = False

    # ------------------------------------------------------------- 主循环
    def print_usage(self) -> None:
        lines = [
            "═══════════════════════════════════════════",
            "  Piper 主从遥操作数据采集",
            "═══════════════════════════════════════════",
            "  c     → 开始 / 继续采集",
            "  space → 暂停采集",
        ]
        if self.use_back_event:
            lines.append("  e     → 切换错误模式（back_event=1）")
        for key, desc in self.subtasks.items():
            lines.append(f"  {key}     → 子任务：{desc}")
        lines += [
            "  s     → 保存当前 episode",
            "  r     → 丢弃缓冲区",
            "  q/Esc → 收尾数据集并退出",
            "═══════════════════════════════════════════",
        ]
        print("\033[32m" + "\n".join(lines) + "\033[0m")

    def run(self) -> None:
        cfg = self.cfg
        self.keys = CollectKeys(cfg.keyboard_backend, subtask_keys=self.subtasks.keys())
        if not self.keys.available:
            logger.warning("没有可用的按键接口：所有操作员按键都失效， Ctrl-C 退出。")

        if cfg.view.enabled:
            self.view = StatusView(
                camera_names=[c.name for c in cfg.cameras],
                bgr=True,
                max_width=cfg.view.max_width,
                window_name="Piper Collect",
            ).start()

        self.print_usage()
        if cfg.collection.dry_run:
            print("[record] ⚠ DRY-RUN：只读不下发，从臂不会动。")

        render_interval = 1.0 / max(1e-3, cfg.view.render_hz)
        last_render = 0.0
        collect_start = time.perf_counter()
        meas_fps = float(cfg.dataset.fps)
        prev_loop_t = None

        try:
            while True:
                loop_t0 = time.perf_counter()
                if prev_loop_t is not None and loop_t0 > prev_loop_t:
                    meas_fps = 0.9 * meas_fps + 0.1 / (loop_t0 - prev_loop_t)
                prev_loop_t = loop_t0

                self.keys.poll()
                for msg in self.keys.drain_messages():
                    print(f"[record] {msg}")
                if self.keys.quit:
                    print("[record] 操作员请求退出")
                    break
                sub_key = self.keys.poll_subtask()
                if sub_key is not None:
                    self.current_subtask = self.subtasks[sub_key]
                    print(f"[record] 🔄 子任务 → {self.current_subtask}")
                if self.keys.poll_save():
                    if self.keys.collecting:
                        print("[record] ⚠ 采集中不能保存，先按 space 暂停")
                    else:
                        self.save_episode()
                if self.keys.poll_discard():
                    self.discard_episode()
                state, action = self.step_arms()
                images = self.read_cameras()
                if self.keys.collecting:
                    if self.buffered_frames() >= cfg.collection.max_timesteps:
                        print(f"[record] ⏹ 达到单集上限 {cfg.collection.max_timesteps}，自动暂停")
                        self.keys.collecting = False
                    elif len(images) < len(cfg.cameras):
                        # 少一路相机就整帧丢掉：add_frame 要求键集合精确匹配，
                        # 补一张假图会把坏数据写进数据集，比丢一帧糟得多。
                        self._dropped_frames += 1
                    else:
                        self.dataset.add_frame(self.build_frame(state, action, images))

                if self.view is not None and self.view.enabled:
                    self.view.update(images, self._status(meas_fps, collect_start))
                    now = time.perf_counter()
                    if now - last_render >= render_interval:
                        last_render = now
                        if not self.view.render_once():
                            print("[record] 操作员在窗口里按了 q/ESC")
                            break

                busy = time.perf_counter() - loop_t0
                precise_sleep(max(self.dt - busy, 0.0))
                self._loop_periods.append(time.perf_counter() - loop_t0)
        finally:
            self.keys.stop()

    def _status(self, meas_fps: float, collect_start: float) -> dict:
        total_eps = self.dataset.meta.total_episodes if self.dataset is not None else 0
        total_frames = self.dataset.meta.total_frames if self.dataset is not None else 0
        buffered = self.buffered_frames()
        return {
            "task": self.cfg.dataset.task,
            "skill_text": self.current_subtask,
            "recording": self.keys.collecting,
            "back": self.keys.error_mode,
            "step": buffered,
            "elapsed": time.perf_counter() - collect_start,
            "fps": meas_fps,
            "episode_index": total_eps,
            "buffered_frames": buffered,
            "saved_episodes": total_eps,
            "saved_frames": total_frames,
        }

    # ------------------------------------------------------------- 收尾
    def print_summary(self) -> None:
        n = len(self._loop_periods)
        if n == 0:
            return
        achieved = n / sum(self._loop_periods)
        sat = float(np.mean(self._saturated)) if self._saturated else 0.0
        print("\n" + "=" * 66)
        print(f"目标 {self.cfg.dataset.fps:.1f} Hz -> 实测 {achieved:.1f} Hz（共 {n} 拍）")
        print(f"限速饱和率 {sat:.1%}（max_joint_step_rad={self.cfg.collection.max_joint_step_rad}）")
        if sat > 0.3:
            print("  ⚠ 饱和率偏高：记录的 action 已经不等于人手的意图，请调大 max_joint_step_rad 后重采。")
        if achieved < 0.9 * self.cfg.dataset.fps:
            print("  ⚠ 实测频率明显低于目标：减少相机 / 降低分辨率 / 降低 dataset.fps。")
        if self._dropped_frames:
            print(f"  ⚠ 有 {self._dropped_frames} 帧因为相机读数缺失被丢弃。")
        print("=" * 66 + "\n")

    def close(self) -> None:
        if self.view is not None:
            try:
                self.view.stop()
            except Exception:
                logger.exception("关闭实时视图失败")
        for pair in self.pairs:
            pair.disconnect()
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception:
                logger.exception("相机 %s 断开失败", name)
        if self._pool is not None:
            self._pool.shutdown(wait=False)


def record(cfg: RecordPiperConfig) -> None:
    from lerobot.datasets.video_utils import VideoEncodingManager

    recorder = PiperRecorder(cfg)
    try:
        recorder.connect()
        recorder.setup_dataset()
        recorder.initialize_arms()
        with VideoEncodingManager(recorder.dataset):
            recorder.run()
    finally:
        recorder.print_summary()
        recorder.close()
        if recorder.dataset is not None:
            try:
                recorder.dataset.finalize()
            except Exception:
                logger.exception("数据集 finalize 失败")
        print(f"[record] 数据集已收尾：{cfg.dataset.root}")


@parser.wrap()
def main(cfg: RecordPiperConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    record(cfg)


if __name__ == "__main__":
    main()
