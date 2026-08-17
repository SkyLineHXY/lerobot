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

"""回放 lerobot-record-piper 采到的数据集：画曲线、拼视频、可选在真机上重放。

三件事互相独立，按需开关：

* **画曲线**（`--plot=true`）：observation.state 与 action 随时间的曲线，用来一眼看出
  从臂跟没跟上、有没有跳变。
* **拼视频**（`--stitch_video=true`）：多路相机横向拼接导出 mp4。
* **真机回放**（`--replay_on_robot=true`）：把 action 序列发回 Piper。会先从当前位姿
  平滑插值到第一帧再开始，绝不让手臂猛地弹过去。

`--chunk_size` / `--inference_delay_s` 用来模拟策略部署时的推理停顿：每 chunk_size 帧
停 inference_delay_s 秒并保持住最后一个目标，用于判断某个 chunk 长度在真机上是否可接受。

用法::

    # 只看曲线和视频，不碰机械臂
    python -m lerobot.scripts.lerobot_replay_piper \
        --root=/data/PiperDemo --repo_id=user/PiperDemo --episode=0

    # 真机回放
    python -m lerobot.scripts.lerobot_replay_piper \
        --root=/data/PiperDemo --repo_id=user/PiperDemo --episode=0 \
        --replay_on_robot=true --arms='[can_left_f]'
"""

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

DOF_PER_ARM = 7  # 6 关节 + 夹爪
RAD_TO_MILLIDEG = 57324.840764  # 与 PiperMotorsBus.joint_factor 一致
GRIPPER_SCALE = 1e6


@dataclass
class ReplayPiperConfig:
    repo_id: str = "user/PiperDataset"
    root: str = "outputs/record_piper"
    episode: int = 0
    out: str = "outputs/replay_piper"

    plot: bool = True
    stitch_video: bool = True
    # 导出视频的帧率，留空则用数据集自己的 fps
    video_fps: float | None = None

    # ---- 真机回放 ----
    replay_on_robot: bool = False
    # 从臂的 CAN 口，顺序必须与采集时的 arms 一致（左在前、右在后），
    # 因为 action 是按这个顺序拼起来的：action[i*7:(i+1)*7] -> arms[i]
    arms: list[str] = field(default_factory=list)
    # 从当前位姿插值到第一帧的参数。别调大 max_joint_step_rad。
    init_duration_s: float = 4.0
    init_hz: float = 100.0
    init_max_step_rad: float = 0.01
    # 到达起始位姿后的倒计时，给人挪开手的时间
    start_delay_s: float = 3.0
    move_speed_ratio: int = 60

    # 模拟策略部署的推理停顿：每 chunk_size 帧停 inference_delay_s 秒
    chunk_size: int = 0
    inference_delay_s: float = 0.0


# ------------------------------------------------------------------ 数据加载
def load_episode(cfg: ReplayPiperConfig):
    """读出一集的 state / action / 图像。

    直接用 `LeRobotDataset(episodes=[i])`，视频解码、时间戳对齐这些都由它负责 ——
    比自己去 parquet 和 mp4 里按 chunk/file 索引拼要稳得多。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(cfg.root).expanduser()
    dataset = LeRobotDataset(
        repo_id=cfg.repo_id,
        root=root,
        episodes=[cfg.episode],
        tolerance_s=1e-4,
        revision=_local_codebase_version(root),
    )
    if len(dataset) == 0:
        raise ValueError(f"episode {cfg.episode} 是空的（数据集共 {dataset.meta.total_episodes} 集）")

    image_keys = [k for k in dataset.meta.features if k.startswith("observation.images.")]
    states, actions, subtasks = [], [], []
    images: dict[str, list] = {k: [] for k in image_keys}
    task = ""

    for i in range(len(dataset)):
        item = dataset[i]
        states.append(np.asarray(item["observation.state"], dtype=np.float32))
        actions.append(np.asarray(item["action"], dtype=np.float32))
        task = item.get("task", task)
        if "subtask" in item:
            subtasks.append(item["subtask"])
        for key in image_keys:
            # LeRobotDataset 给的是 (C,H,W) 的 float32 [0,1] 张量
            frame = np.asarray(item[key])
            if frame.ndim == 3 and frame.shape[0] == 3:
                frame = frame.transpose(1, 2, 0)
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            images[key].append(frame)

    return {
        "meta": dataset.meta,
        "fps": dataset.meta.fps,
        "state": np.stack(states),
        "action": np.stack(actions),
        "images": images,
        "subtasks": subtasks,
        "task": task,
    }


def _local_codebase_version(root: Path) -> str:
    import json

    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        return json.loads(info_path.read_text()).get("codebase_version", "v3.0")
    return "v3.0"


def _joint_labels(meta, key: str, dim: int) -> list[str]:
    names = (meta.features.get(key) or {}).get("names")
    if isinstance(names, list) and len(names) == dim:
        return names
    return [f"j{i}" for i in range(dim)]


# ------------------------------------------------------------------ 可视化
def plot_state_action(episode: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state, action = episode["state"], episode["action"]
    t = np.arange(len(state)) / float(episode["fps"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, data, key, title in (
        (axes[0], state, "observation.state", "Observation State (qpos)"),
        (axes[1], action, "action", "Action"),
    ):
        labels = _joint_labels(episode["meta"], key, data.shape[1])
        for i in range(data.shape[1]):
            ax.plot(t, data[:, i], label=labels[i], linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel("Position (rad / m)")
        ax.legend(fontsize=6, ncol=4)
        ax.grid(True, alpha=0.4)
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"[replay] 已保存关节曲线 → {output_path}")


def stitch_video(episode: dict, output_path: Path, fps: float) -> None:
    """把多路相机横向拼成一个 mp4。"""
    import cv2

    lists = [(k, v) for k, v in episode["images"].items() if v]
    if not lists:
        print("[replay] 数据集里没有图像，跳过拼视频。")
        return

    n_frames = min(len(v) for _k, v in lists)
    height = max(v[0].shape[0] for _k, v in lists)
    width = sum(v[0].shape[1] for _k, v in lists)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for i in range(n_frames):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            x = 0
            for _key, frames in lists:
                frame = frames[i]
                h, w = frame.shape[:2]
                # 数据集里存的是 RGB，cv2 写文件要 BGR
                canvas[:h, x : x + w] = frame[:, :, ::-1]
                x += w
            writer.write(canvas)
    finally:
        writer.release()
    print(f"[replay] 已保存拼接视频 → {output_path}（{n_frames} 帧 @ {fps:g} Hz）")


# ------------------------------------------------------------------ 真机回放
def _read_arm_state(arm) -> np.ndarray:
    js = arm.GetArmJointMsgs().joint_state
    gs = arm.GetArmGripperMsgs().gripper_state
    return np.array(
        [
            js.joint_1 / RAD_TO_MILLIDEG,
            js.joint_2 / RAD_TO_MILLIDEG,
            js.joint_3 / RAD_TO_MILLIDEG,
            js.joint_4 / RAD_TO_MILLIDEG,
            js.joint_5 / RAD_TO_MILLIDEG,
            js.joint_6 / RAD_TO_MILLIDEG,
            gs.grippers_angle / GRIPPER_SCALE,
        ],
        dtype=np.float64,
    )


def _send_arm_command(arm, cmd: np.ndarray, speed: int) -> None:
    arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    arm.JointCtrl(*(int(v * RAD_TO_MILLIDEG) for v in cmd[:6]))
    arm.GripperCtrl(int(abs(cmd[6] * GRIPPER_SCALE)), 1000, 0x01, 0)


def _move_to_start(arms: list, targets: list, cfg: ReplayPiperConfig) -> bool:
    """所有臂同时从当前位姿平滑插值到第一帧。

    步数取「总时长」与「单步最大关节增量」两个约束里的较大者 —— 起始位姿离得远就
    多花点时间，而不是全速冲过去。smoothstep 缓入缓出去掉两端的冲击。
    """
    time.sleep(0.1)  # 刚 EnableArm 时关节反馈可能还是旧的
    try:
        starts = [_read_arm_state(a) for a in arms]
    except Exception as exc:
        print(f"[replay] 读不到当前关节位姿（{exc}），拒绝盲动。")
        return False

    max_delta = max(float(np.abs(t[:6] - s[:6]).max()) for s, t in zip(starts, targets, strict=True))
    steps = max(
        int(round(cfg.init_duration_s * cfg.init_hz)),
        int(math.ceil(max_delta / cfg.init_max_step_rad)) if cfg.init_max_step_rad > 0 else 1,
        1,
    )
    print(f"[replay] 移动到第一帧：最大关节差 {max_delta:.3f} rad，{steps} 步 @ {cfg.init_hz:g} Hz")

    dt = 1.0 / cfg.init_hz
    for i in range(1, steps + 1):
        alpha = i / steps
        s = alpha * alpha * (3.0 - 2.0 * alpha)
        for arm, start, target in zip(arms, starts, targets, strict=True):
            _send_arm_command(arm, start + (target - start) * s, cfg.move_speed_ratio)
        precise_sleep(dt)
    return True


def _hold_pose(arms: list, action: np.ndarray, cfg: ReplayPiperConfig, duration_s: float) -> None:
    """反复重发同一个目标，模拟等待推理时手臂被主动保持住的状态。"""
    if duration_s <= 0:
        return
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        for i, arm in enumerate(arms):
            _send_arm_command(arm, action[i * DOF_PER_ARM : (i + 1) * DOF_PER_ARM], cfg.move_speed_ratio)
        precise_sleep(min(0.02, max(0.0, deadline - time.perf_counter())))


def replay_on_robot(episode: dict, cfg: ReplayPiperConfig) -> None:
    from piper_sdk import C_PiperInterface_V2

    actions = episode["action"].astype(np.float64)
    expected = len(cfg.arms) * DOF_PER_ARM
    if actions.shape[1] != expected:
        raise ValueError(
            f"action 宽度 {actions.shape[1]} 与 arms×7 ({expected}) 对不上 —— "
            f"`arms` 配了 {len(cfg.arms)} 条臂，顺序也必须和采集时一致。"
        )

    arms = []
    for port in cfg.arms:
        print(f"[replay] 连接从臂 {port}")
        arm = C_PiperInterface_V2(port)
        arm.ConnectPort()
        arm.EnableArm(7)
        arms.append(arm)
    time.sleep(1.0)

    first = actions[0]
    targets = [first[i * DOF_PER_ARM : (i + 1) * DOF_PER_ARM] for i in range(len(arms))]
    if not _move_to_start(arms, targets, cfg):
        return

    if cfg.start_delay_s > 0:
        print(f"[replay] 已到起始位姿，{cfg.start_delay_s:g}s 后开始回放，请把手挪开 …")
        time.sleep(cfg.start_delay_s)

    fps = episode["fps"]
    dt = 1.0 / fps
    simulate = cfg.chunk_size > 0 and cfg.inference_delay_s > 0
    if simulate:
        stalls = max(0, math.ceil(len(actions) / cfg.chunk_size) - 1)
        print(
            f"[replay] 模拟推理延迟：每 {cfg.chunk_size} 帧停 "
            f"{cfg.inference_delay_s * 1e3:.0f}ms（共 {stalls} 次，总计 "
            f"+{stalls * cfg.inference_delay_s:.1f}s）"
        )

    print(f"[replay] 回放 {len(actions)} 帧 @ {fps:g} Hz，臂：{cfg.arms}")
    for k, action in enumerate(actions):
        if simulate and k > 0 and k % cfg.chunk_size == 0:
            _hold_pose(arms, actions[k - 1], cfg, cfg.inference_delay_s)
        loop_t0 = time.perf_counter()
        for i, arm in enumerate(arms):
            _send_arm_command(arm, action[i * DOF_PER_ARM : (i + 1) * DOF_PER_ARM], cfg.move_speed_ratio)
        precise_sleep(max(0.0, dt - (time.perf_counter() - loop_t0)))
    print("[replay] 回放结束。")


# ------------------------------------------------------------------ 入口
def replay(cfg: ReplayPiperConfig) -> None:
    episode = load_episode(cfg)
    state, action = episode["state"], episode["action"]
    print(
        f"[replay] episode {cfg.episode}: {len(state)} 帧 @ {episode['fps']} Hz，"
        f"{state.shape[1]} 维，任务「{episode['task']}」"
    )
    # 采集时 action 记的是从臂下一拍的位姿，所以这个差值就是"从臂落后主臂多少"
    lag = float(np.abs(action[:, :6] - state[:, :6]).max())
    print(f"[replay] |action - state| 最大 {lag:.4f} rad")
    if episode["subtasks"]:
        seen = list(dict.fromkeys(episode["subtasks"]))
        print(f"[replay] 子任务序列：{seen}")

    out = Path(cfg.out).expanduser() / f"episode_{cfg.episode:06d}"
    if cfg.plot:
        plot_state_action(episode, out / "state_action.png")
    if cfg.stitch_video:
        stitch_video(episode, out / "cameras.mp4", cfg.video_fps or episode["fps"])
    if cfg.replay_on_robot:
        if not cfg.arms:
            raise ValueError("replay_on_robot=true 时必须用 `arms` 给出从臂 CAN 口，顺序与采集时一致。")
        replay_on_robot(episode, cfg)


@parser.wrap()
def main(cfg: ReplayPiperConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    replay(cfg)


if __name__ == "__main__":
    main()
