"""LIBERO 示教采集：帧组装与数据集读写往返。

不启动 LIBERO（robosuite 太重且要显示），只喂一个形状正确的假观测，但**数据集
是真的**——按 CLAUDE.md 的要求，写侧和读侧分开钉住会漏掉 schema 不匹配，
往返测试才抓得到。
"""

from types import SimpleNamespace

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.rlt.collect_libero import (
    ACTION_NAMES,
    STATE_DIM,
    CollectLiberoConfig,
    LiberoTeleopCollector,
    build_features,
)

SIZE = 32


def _cfg(root, **kw):
    cfg = CollectLiberoConfig(observation_size=SIZE, **kw)
    cfg.dataset.root = str(root)
    cfg.dataset.repo_id = "test/libero_teleop"
    cfg.dataset.fps = 10
    # 这台机器的解码器读不了 libsvtav1 的 32x32 流，往返测试改用 h264
    cfg.dataset.video_codec = "h264"
    return cfg


def _obs(rng):
    """与 LiberoEnv 的 pixels_agent_pos 观测同构的假观测。"""
    return {
        "pixels": {
            key: rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8) for key in ("image", "wrist_image")
        },
        "robot_state": {
            "eef": {
                "pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            },
            "gripper": {"qpos": np.array([0.02, -0.02], dtype=np.float32)},
        },
    }


def test_flat_state_matches_the_processor_used_at_inference():
    """采集侧的 8 维状态必须和 LiberoProcessorStep 逐位一致。"""
    from lerobot.processor.env_processor import LiberoProcessorStep

    obs = _obs(np.random.default_rng(0))
    flat = LiberoTeleopCollector._flat_state(obs)

    assert flat.shape == (STATE_DIM,)
    assert flat.dtype == np.float32
    eef = obs["robot_state"]["eef"]
    expected_axisangle = (
        LiberoProcessorStep()._quat2axisangle(torch.as_tensor(eef["quat"]).reshape(1, 4)).reshape(-1)
    )
    assert np.allclose(flat[:3], eef["pos"])
    assert np.allclose(flat[3:6], expected_axisangle.numpy(), atol=1e-6)
    assert np.allclose(flat[6:], obs["robot_state"]["gripper"]["qpos"])


def test_frame_keys_match_the_declared_features(tmp_path):
    """add_frame 要求 frame 的键恰好等于 features 减去 DEFAULT_FEATURES。"""
    cfg = _cfg(tmp_path / "ds")
    collector = LiberoTeleopCollector(cfg)
    frame = collector._frame(
        _obs(np.random.default_rng(0)),
        np.zeros(len(ACTION_NAMES), dtype=np.float32),
        task="pick up the bowl",
    )

    expected = set(build_features(cfg)) - set(DEFAULT_FEATURES)
    assert set(frame) - {"task"} == expected


def test_images_are_rotated_to_the_libero_dataset_convention(tmp_path):
    """LiberoProcessorStep 推理时把图旋转 180 度，落盘必须同朝向。"""
    cfg = _cfg(tmp_path / "ds")
    collector = LiberoTeleopCollector(cfg)
    obs = _obs(np.random.default_rng(1))
    frame = collector._frame(obs, np.zeros(len(ACTION_NAMES), dtype=np.float32), task="t")

    raw = obs["pixels"]["image"]
    stored = frame["observation.images.image"]
    assert stored.dtype == np.uint8
    assert stored.shape == (SIZE, SIZE, 3)
    assert np.array_equal(stored, raw[::-1, ::-1])
    # 通道顺序不能翻：只有 cv2 显示路径才要 BGR
    assert np.array_equal(stored[0, 0], raw[-1, -1])


def test_status_view_receives_upright_rgb_frames_and_collection_status(tmp_path):
    cfg = _cfg(tmp_path / "ds")
    collector = LiberoTeleopCollector(cfg)
    obs = _obs(np.random.default_rng(3))

    class FakeView:
        enabled = True

        def update(self, images, status):
            self.images = images
            self.status = status

        def render_once(self):
            return False

    collector.view = FakeView()
    collector.env = SimpleNamespace(task_description="pick up the bowl")
    collector.dataset = SimpleNamespace(meta=SimpleNamespace(total_episodes=2, total_frames=42))

    keep_running = collector._render_view(
        obs,
        task="pick_up_the_bowl",
        episode_index=3,
        n_frames=7,
        recording=True,
        elapsed=1.5,
        fps=19.5,
    )

    assert keep_running is False
    assert np.array_equal(collector.view.images["image"], obs["pixels"]["image"][::-1, ::-1])
    assert collector.view.images["image"].flags.c_contiguous
    assert collector.view.status["task"] == "pick up the bowl"
    assert collector.view.status["recording"] is True
    assert collector.view.status["buffered_frames"] == 7
    assert collector.view.status["saved_episodes"] == 2
    assert collector.view.status["saved_frames"] == 42


def test_dataset_roundtrip(tmp_path):
    cfg = _cfg(tmp_path / "ds")
    collector = LiberoTeleopCollector(cfg)
    collector.setup_dataset()

    from lerobot.datasets.video_utils import VideoEncodingManager

    rng = np.random.default_rng(2)
    actions = []
    with VideoEncodingManager(collector.dataset):
        for _ in range(6):
            action = rng.random(len(ACTION_NAMES)).astype(np.float32)
            actions.append(action)
            collector.dataset.add_frame(collector._frame(_obs(rng), action, task="pick up the bowl"))
        collector.dataset.save_episode()
        collector.dataset.meta._flush_metadata_buffer()
    collector.dataset.finalize()

    reloaded = LeRobotDataset(repo_id=cfg.dataset.repo_id, root=cfg.dataset.root)
    assert reloaded.meta.total_episodes == 1
    assert reloaded.meta.total_frames == 6

    item = reloaded[0]
    assert item["observation.state"].shape == (STATE_DIM,)
    assert item["action"].shape == (len(ACTION_NAMES),)
    assert np.allclose(item["action"].numpy(), actions[0], atol=1e-5)
    assert reloaded.meta.features["action"]["names"] == list(ACTION_NAMES)
