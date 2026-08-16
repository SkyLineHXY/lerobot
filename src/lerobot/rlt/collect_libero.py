"""用键盘/手柄在 LIBERO 里人工示教，采集成 LeRobotDataset。

阶段 1 的 `train_rl_token` 需要任务演示数据。官方 `lerobot/libero_*` 数据集只覆盖
固定任务集，自建任务（或换了初始摆放、换了相机）就没有对应的演示可用，这个脚本
补的是这一段。

产出的列与 `lerobot/libero_10` 完全一致，所以可以直接喂给
`lerobot-rlt-train-token`：

    observation.images.image        (H, W, 3) uint8，已按 LIBERO 约定旋转 180 度
    observation.images.wrist_image  同上
    observation.state               8 维 = eef 位置(3) + eef 轴角(3) + 夹爪 qpos(2)
    action                          7 维 delta EE = [dx,dy,dz,droll,dpitch,dyaw,gripper]

用法：

    python -m lerobot.rlt.collect_libero --config_path examples/rlt/collect_libero.yaml
    python -m lerobot.rlt.collect_libero --config_path examples/rlt/collect_libero.yaml \\
        --teleop.device=gamepad --dataset.repo_id=me/libero_pick_bowl

算子键（与 train_online 一致）：
    space   开始/暂停记录（松开时机械臂不动，也不写帧）
    s       本集成功，保存
    f       本集失败，保存（失败演示对 RL 也有用，但不该混进 BC 数据里，
            所以默认 keep_failures=false 直接丢弃）
    退格    丢弃本集
    Esc     结束采集

注意：键盘遥操作用方向键，与算子键默认的 left=丢弃冲突，所以这里的丢弃键是退格。
"""

# 这里不能加 `from __future__ import annotations`：parser.wrap() 直接读
# inspect.getfullargspec(fn).annotations，注解变成字符串后 draccus 取不到配置类。
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.policies.rlt import SimTeleopConfig
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

from .teleop.device import make_teleop_device
from .teleop.keys import KeyboardEventListener

logger = logging.getLogger(__name__)

# robosuite OSC_POSE 的通道顺序，也是 lerobot/libero_* 数据集 action 列的顺序。
ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]
ROTATION_CHANNELS = frozenset({"delta_roll", "delta_pitch", "delta_yaw"})
STATE_DIM = 8

GRIPPER_CLOSE, GRIPPER_STAY, GRIPPER_OPEN = 0.0, 1.0, 2.0


@dataclass
class CollectDatasetConfig:
    repo_id: str = "local/libero_teleop"
    root: str = "outputs/datasets/libero_teleop"
    fps: int = 20
    resume: bool = False
    video_codec: str = "libsvtav1"
    # 相机键名必须和 lerobot/libero_* 一致，否则阶段 1 训出来的处理器对不上。
    image_keys: list[str] = field(default_factory=lambda: ["image", "wrist_image"])


@dataclass
class CollectLiberoConfig:
    suite: str = "libero_10"
    task_id: int = 0
    observation_size: int = 256
    camera_name: str = "agentview_image,robot0_eye_in_hand_image"
    max_episode_steps: int = 600
    n_episodes: int = 20
    seed: int = 0

    teleop: SimTeleopConfig = field(default_factory=SimTeleopConfig)
    dataset: CollectDatasetConfig = field(default_factory=CollectDatasetConfig)

    keyboard_backend: str = "auto"
    # 方向键被键盘遥操作占用了，所以丢弃键默认改成退格。
    discard_key: str = "backspace"
    # 失败演示会把 BC 目标往错误方向拉，默认不写进数据集。
    keep_failures: bool = False


def build_features(cfg: CollectLiberoConfig) -> dict:
    size = cfg.observation_size
    features = {
        f"{OBS_IMAGES}.{key}": {
            "dtype": "video",
            "shape": (size, size, 3),
            "names": ["height", "width", "channels"],
        }
        for key in cfg.dataset.image_keys
    }
    features[OBS_STATE] = {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": [f"state_{i}" for i in range(STATE_DIM)],
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (len(ACTION_NAMES),),
        "names": list(ACTION_NAMES),
    }
    return features


class LiberoTeleopCollector:
    """LIBERO + 手持设备 -> LeRobotDataset。"""

    def __init__(self, cfg: CollectLiberoConfig):
        self.cfg = cfg
        self.keys = KeyboardEventListener(
            backend=cfg.keyboard_backend, discard_key=cfg.discard_key
        )
        self.env = None
        self.teleop = None
        self.dataset = None
        self._gripper = -1.0  # robosuite: -1 张开 / +1 闭合，stay 时保持不变
        self._channel_index = None

    # --------------------------------------------------------------- 构建
    def setup_env(self) -> None:
        from lerobot.envs.libero import LiberoEnv, _get_suite, _parse_camera_names

        cameras = _parse_camera_names(self.cfg.camera_name)
        keys = self.cfg.dataset.image_keys
        if len(keys) != len(cameras):
            raise ValueError(
                f"dataset.image_keys 有 {len(keys)} 个 {keys}，而 camera_name 是 "
                f"{len(cameras)} 个 {cameras}；两者必须一一对应。"
            )
        self.env = LiberoEnv(
            task_suite=_get_suite(self.cfg.suite),
            task_id=self.cfg.task_id,
            task_suite_name=self.cfg.suite,
            obs_type="pixels_agent_pos",
            observation_width=self.cfg.observation_size,
            observation_height=self.cfg.observation_size,
            camera_name=self.cfg.camera_name,
            camera_name_mapping=dict(zip(cameras, keys, strict=True)),
            control_mode="relative",
            episode_length=self.cfg.max_episode_steps,
        )

    def setup_teleop(self) -> None:
        self.teleop = make_teleop_device(self.cfg.teleop)
        names = self.teleop.action_features.get("names", {})
        self._channel_index = {name: i for i, name in enumerate(ACTION_NAMES)}
        missing = [n for n in ACTION_NAMES if n not in names]
        if missing:
            logger.warning(
                "遥操作设备没有 %s 通道，这些维度将恒为 0；需要转手腕的任务请开 "
                "teleop.use_rotation 或换设备。",
                missing,
            )

    def setup_dataset(self) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.repair import repair_dataset_consistency

        cfg = self.cfg.dataset
        root = Path(cfg.root).expanduser()
        features = build_features(self.cfg)

        if root.exists():
            if not cfg.resume:
                raise FileExistsError(
                    f"{root} 已存在。加 --dataset.resume=true 续采，或换一个 root。"
                    "（LeRobotDatasetMetadata.create 用的是 mkdir(exist_ok=False)）"
                )
            # 上次运行如果是被 kill -9 掐掉的，元数据与 data 的 episode 数会对不上，
            # 此时 LeRobotDataset.__init__ 会退回去请求 HF Hub 并卡死在本地 repo_id 上。
            repair_dataset_consistency(root)
            self.dataset = LeRobotDataset(repo_id=cfg.repo_id, root=root, tolerance_s=1e-4)
            # 续采得到的 episode_buffer 是 None，不先建好，丢弃空集时会解引用 None。
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
            print(f"[collect] 续采：已有 {self.dataset.meta.total_episodes} 集")
            return

        self.dataset = LeRobotDataset.create(
            repo_id=cfg.repo_id,
            fps=cfg.fps,
            features=features,
            root=root,
            robot_type="libero_panda",
            use_videos=True,
            tolerance_s=1e-4,
            vcodec=cfg.video_codec,
        )
        print(f"[collect] 已新建数据集：{root}")

    # --------------------------------------------------------------- 采集
    def _env_action(self) -> np.ndarray:
        raw = self.teleop.get_action()
        vec = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        for name, i in self._channel_index.items():
            if name == "gripper":
                continue
            scale = (
                self.cfg.teleop.rotation_scale
                if name in ROTATION_CHANNELS
                else self.cfg.teleop.position_scale
            )
            vec[i] = float(raw.get(name, 0.0)) * scale

        # stay 必须重复上一次命令：发 0 会让夹爪在抓取途中自己松开。
        command = float(raw.get("gripper", GRIPPER_STAY))
        if command == GRIPPER_OPEN:
            self._gripper = -1.0
        elif command == GRIPPER_CLOSE:
            self._gripper = 1.0
        vec[self._channel_index["gripper"]] = self._gripper
        return vec

    def _frame(self, obs: dict, action: np.ndarray, task: str) -> dict:
        frame = {"task": task}
        for key in self.cfg.dataset.image_keys:
            img = obs["pixels"][key]
            # LiberoProcessorStep 在推理时把图像旋转 180 度；数据集必须以同样的
            # 朝向落盘，否则训练和评估看到的是上下颠倒的两个世界。
            frame[f"{OBS_IMAGES}.{key}"] = np.ascontiguousarray(img[::-1, ::-1])
        frame[OBS_STATE] = self._flat_state(obs)
        frame["action"] = action.astype(np.float32)
        return frame

    @staticmethod
    def _flat_state(obs: dict) -> np.ndarray:
        """与 LiberoProcessorStep 完全一致的 8 维扁平状态。"""
        from lerobot.processor.env_processor import LiberoProcessorStep

        state = obs["robot_state"]
        eef = state["eef"]
        quat = torch.as_tensor(eef["quat"], dtype=torch.float32).reshape(1, 4)
        axisangle = LiberoProcessorStep()._quat2axisangle(quat).reshape(-1)
        flat = np.concatenate(
            [
                np.asarray(eef["pos"], dtype=np.float32).reshape(-1),
                axisangle.numpy().astype(np.float32),
                np.asarray(state["gripper"]["qpos"], dtype=np.float32).reshape(-1),
            ]
        )
        return flat.astype(np.float32)

    def record_episode(self, index: int) -> str:
        """采一集；返回 'success' / 'failure' / 'discard' / 'quit'。"""
        obs, _info = self.env.reset(seed=self.cfg.seed + index)
        self.keys.reset_episode_flags()
        self.keys.clear_intervention()
        self._gripper = -1.0
        task = self.env.task
        dt = 1.0 / self.cfg.dataset.fps
        n_frames = 0

        print(
            f"[collect] 第 {index + 1} 集 | 任务：{task}\n"
            "          空格开始记录，s=成功 f=失败 退格=丢弃 Esc=退出"
        )
        for _ in range(self.cfg.max_episode_steps):
            t0 = time.perf_counter()
            if self.keys.should_quit():
                return "quit"
            if self.keys.poll_discard():
                return "discard"
            success_key, failure_key = self.keys.poll_outcome()
            if success_key:
                return "success"
            if failure_key:
                return "failure"

            # 没按空格时既不推机械臂也不写帧：操作员需要时间摆位和看清场景，
            # 把这段静止画面写进数据集只会教策略"原地不动"。
            if not self.keys.intervening:
                precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
                continue

            action = self._env_action()
            frame = self._frame(obs, action, task)
            obs, _reward, terminated, _truncated, info = self.env.step(action)
            self.dataset.add_frame(frame)
            n_frames += 1

            if bool(info.get("is_success", False)) or terminated:
                print(f"[collect] 仿真判定成功（{n_frames} 帧）")
                return "success"
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

        print(f"[collect] 到达步数上限（{n_frames} 帧）")
        return "failure"

    def _finish_episode(self, outcome: str) -> bool:
        """落盘或丢弃当前 episode；返回是否真的写入了。"""
        buffer = self.dataset.episode_buffer
        empty = buffer is None or buffer.get("size", 0) == 0
        keep = outcome == "success" or (outcome == "failure" and self.cfg.keep_failures)

        if empty or not keep:
            self.dataset.clear_episode_buffer()
            self.dataset.episode_buffer = self.dataset.create_episode_buffer()
            print(f"[collect] 丢弃本集（{outcome}）")
            return False

        self.dataset.save_episode()
        # 元数据每集都刷盘；数据侧仍在一个长生命周期的 ParquetWriter 后面，
        # 中途被 kill -9 会让两边的 episode 数对不上。
        self.dataset.meta._flush_metadata_buffer()
        print(f"[collect] 已保存第 {self.dataset.meta.total_episodes} 集（{outcome}）")
        return True

    def run(self) -> None:
        from lerobot.datasets.video_utils import VideoEncodingManager

        saved = 0
        with VideoEncodingManager(self.dataset):
            for index in range(self.cfg.n_episodes):
                outcome = self.record_episode(index)
                if outcome == "quit":
                    self._finish_episode("discard")
                    print("[collect] 操作员结束采集。")
                    break
                if self._finish_episode(outcome):
                    saved += 1
                # 每 10 集合上 parquet writer：footer 只在 close() 时写，
                # 不定期落盘的话崩溃会丢掉这中间所有集的数据。
                if saved and saved % 10 == 0:
                    self.dataset._close_writer()
                    self.dataset._writer_closed_for_reading = True
        # VideoEncodingManager 只收编码器，不收 parquet writer。少了这一步，
        # data 和 meta 的 parquet 都没有 footer，数据集根本读不回来。
        self.dataset.finalize()
        print(f"[collect] 完成：共保存 {saved} 集 -> {self.cfg.dataset.root}")

    def close(self) -> None:
        for label, teardown in (
            ("keyboard", self.keys.stop),
            ("teleop", (self.teleop.disconnect if self.teleop is not None else lambda: None)),
            ("env", (self.env.close if self.env is not None else lambda: None)),
        ):
            try:
                teardown()
            except Exception:
                logger.exception("[collect] %s 收尾失败", label)


def collect(cfg: CollectLiberoConfig) -> None:
    collector = LiberoTeleopCollector(cfg)
    try:
        collector.setup_env()
        collector.setup_dataset()
        # 遥操作设备和键盘后端都放在最后：连接设备可能要打开显示/输入子系统，
        # 而 termios 后端一旦把 stdin 切进 cbreak，交互式提示就用不了了。
        collector.setup_teleop()
        collector.keys.start()
        if not collector.keys.available:
            raise RuntimeError(
                "算子键不可用，无法标注成功/失败，也没法开始记录。"
                "请在真实终端里运行，或设置 DISPLAY 让 pynput 可用。"
            )
        collector.run()
    finally:
        collector.close()


@parser.wrap()
def main(cfg: CollectLiberoConfig):
    collect(cfg)


if __name__ == "__main__":
    main()
