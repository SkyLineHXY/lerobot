"""采集 → 数据集 → 回放的往返测试（不碰硬件）。

单测各自把 `build_frame` 和 `load_episode` 钉住还不够：真正会出事的是两者对不上 ——
schema 写进去能过、读出来维度或通道错了。所以这里用真的 `LeRobotDataset` 走一遍，
用采集脚本的 features 建库、写帧、存集，再用回放脚本读回来比对。
"""

from pathlib import Path

import numpy as np
import pytest

from lerobot.scripts import lerobot_record_piper as rp
from lerobot.scripts.lerobot_replay_piper import ReplayPiperConfig, load_episode, plot_state_action

CAM = "cam_top"
H, W = 8, 12
N_FRAMES = 6
FPS = 10


@pytest.fixture
def recorded(tmp_path):
    """用采集脚本的 schema 造一个真实的单臂数据集。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cfg = rp.RecordPiperConfig(
        arms=[rp.ArmSpec(name="left")],
        dataset=rp.DatasetSpec(
            repo_id="test/piper",
            root=str(tmp_path / "ds"),
            task="pick the cube",
            fps=FPS,
            # 图像模式，免得单测依赖具体的视频编码器
            use_videos=False,
            subtasks={"1": "reach", "2": "grasp"},
        ),
        cameras=[rp.CameraSpec(name=CAM, width=W, height=H)],
    )

    recorder = rp.PiperRecorder.__new__(rp.PiperRecorder)
    recorder.cfg = cfg
    recorder.dof = 7
    recorder.use_subtask = True
    recorder.use_back_event = True
    recorder.subtasks = dict(cfg.dataset.subtasks)
    recorder.subtask_name_to_idx = {"reach": 0, "grasp": 1}
    recorder.current_subtask = "reach"
    recorder.keys = None

    dataset = LeRobotDataset.create(
        repo_id=cfg.dataset.repo_id,
        fps=FPS,
        features=recorder.build_features(),
        root=cfg.dataset.root,
        robot_type="piper",
        use_videos=False,
        tolerance_s=1e-4,
        image_writer_processes=0,
        image_writer_threads=2,
    )
    # 光有 subtask_index 列没用：__getitem__ 要能读到 meta/subtasks.parquet 才会
    # 把索引翻译回描述，所以这里走采集脚本自己的写入方法。
    recorder._write_subtasks_parquet(Path(cfg.dataset.root))

    rng = np.random.default_rng(0)
    states, actions, images = [], [], []
    for i in range(N_FRAMES):
        state = np.full(7, i * 0.1, dtype=np.float32)
        action = np.full(7, (i + 1) * 0.1, dtype=np.float32)
        img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        states.append(state)
        actions.append(action)
        images.append(img)
        dataset.add_frame(recorder.build_frame(state, action, {CAM: img}))
    dataset.save_episode()
    dataset.finalize()

    return {
        "root": cfg.dataset.root,
        "repo_id": cfg.dataset.repo_id,
        "state": np.stack(states),
        "action": np.stack(actions),
        "images": images,
    }


def _load(recorded):
    return load_episode(ReplayPiperConfig(repo_id=recorded["repo_id"], root=recorded["root"], episode=0))


def test_state_and_action_survive_the_round_trip(recorded):
    ep = _load(recorded)
    assert ep["fps"] == FPS
    assert ep["state"].shape == (N_FRAMES, 7)
    np.testing.assert_allclose(ep["state"], recorded["state"], atol=1e-5)
    np.testing.assert_allclose(ep["action"], recorded["action"], atol=1e-5)


def test_images_come_back_as_uint8_hwc(recorded):
    """回放要拼视频，读回来必须是 (H,W,3) uint8，不是 (C,H,W) 的 float 张量。"""
    ep = _load(recorded)
    frames = ep["images"][f"observation.images.{CAM}"]
    assert len(frames) == N_FRAMES
    for frame in frames:
        assert frame.dtype == np.uint8
        assert frame.shape == (H, W, 3)


def test_images_keep_their_content(recorded):
    """通道顺序错了这里就会炸 —— 拼出来的视频会红蓝互换。"""
    ep = _load(recorded)
    frames = ep["images"][f"observation.images.{CAM}"]
    np.testing.assert_allclose(frames[0], recorded["images"][0], atol=1)


def test_task_and_subtask_are_readable(recorded):
    ep = _load(recorded)
    assert ep["task"] == "pick the cube"
    assert ep["subtasks"] == ["reach"] * N_FRAMES


def test_plot_writes_a_file(recorded, tmp_path):
    out = tmp_path / "plots" / "state_action.png"
    plot_state_action(_load(recorded), out)
    assert out.is_file() and out.stat().st_size > 0


def test_missing_episode_is_rejected(recorded):
    """越界的 episode 必须直接报错，不能悄悄回放成空的。"""
    with pytest.raises((ValueError, IndexError, KeyError)):
        load_episode(ReplayPiperConfig(repo_id=recorded["repo_id"], root=recorded["root"], episode=99))
