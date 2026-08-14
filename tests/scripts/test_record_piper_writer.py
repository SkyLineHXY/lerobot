"""采集脚本写帧线程 / drain 的回归测试。

真实崩溃现场：按 s 保存时抛
`AttributeError: 'numpy.ndarray' object has no attribute 'append'`。
`save_episode` 会把 episode_buffer 里每个 list 原地换成 np.ndarray，而旧版
`drain_writer` 只等队列 empty —— 队列空了不代表写帧线程写完了，它可能刚 get()
出最后一帧、正卡在 add_frame 里。于是 save 和 add_frame 撞车。

所以这里用真实的 LeRobotDataset 跑真实的写帧线程，不 mock add_frame：
只有真数据集才会在 save_episode 时真的把 list 换成 ndarray。
"""

import threading
import time

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record_piper import PiperRecorder


def _make_dataset(root):
    return LeRobotDataset.create(
        repo_id="test/piper_writer",
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
            "action": {"dtype": "float32", "shape": (7,), "names": None},
        },
        root=root,
        robot_type="piper",
        use_videos=False,
        tolerance_s=1e-4,
    )


def _make_recorder(dataset, queue_maxsize=90):
    """绕开 __init__：它会去构造 ArmPair（要 CAN 口）。这里只测写帧/drain 这一段。"""
    import queue as queue_mod

    rec = PiperRecorder.__new__(PiperRecorder)
    rec.dataset = dataset
    rec._frame_queue = queue_mod.Queue(maxsize=queue_maxsize)
    rec._writer_thread = None
    rec._writer_stop = threading.Event()
    rec._writer_error = []
    rec._inflight = 0
    rec._inflight_cv = threading.Condition()
    rec._bar = None
    rec._bar_frames = 0
    rec._dropped_frames = 0
    rec._t_add = []
    return rec


def _frame(i):
    return {
        "observation.state": np.full(7, i, dtype=np.float32),
        "action": np.full(7, i, dtype=np.float32),
        "task": "t",
    }


def test_drain_waits_for_inflight_add_frame(tmp_path):
    """队列已空但写帧线程还在 add_frame 里时，drain 必须继续等。"""
    dataset = _make_dataset(tmp_path / "ds")
    rec = _make_recorder(dataset)

    slow_started = threading.Event()
    release = threading.Event()
    real_add = dataset.add_frame

    def slow_add(frame, *a, **kw):
        slow_started.set()
        release.wait(timeout=5.0)  # 模拟编码器队列满导致 add_frame 卡住
        return real_add(frame, *a, **kw)

    dataset.add_frame = slow_add
    rec.start_writer()
    try:
        rec._enqueue_frame(_frame(0))
        assert slow_started.wait(timeout=5.0)
        assert rec._frame_queue.empty()  # 已 get 出来，队列确实空了

        drained = threading.Event()
        threading.Thread(target=lambda: (rec.drain_writer(), drained.set()), daemon=True).start()

        # 帧还没写完，drain 不许返回
        assert not drained.wait(timeout=0.5)
        release.set()
        assert drained.wait(timeout=5.0)
        assert rec.buffered_frames() == 1
    finally:
        dataset.add_frame = real_add
        rec.stop_writer()


def test_save_after_drain_does_not_race_add_frame(tmp_path):
    """完整复现崩溃场景：连续入队 + drain + save_episode，反复多轮不许抛。"""
    dataset = _make_dataset(tmp_path / "ds")
    rec = _make_recorder(dataset)
    rec.start_writer()
    try:
        for _ in range(5):
            for i in range(20):
                rec._enqueue_frame(_frame(i))
                time.sleep(0.001)
            rec.drain_writer(timeout_s=30.0)
            # drain 返回后写帧线程必须是空闲的，否则 save_episode 会撞上 add_frame
            assert rec._inflight == 0
            assert rec.buffered_frames() == 20
            dataset.save_episode()
        assert dataset.meta.total_episodes == 5
        assert dataset.meta.total_frames == 100
    finally:
        rec.stop_writer()


def test_drain_reraises_writer_error_instead_of_hanging(tmp_path):
    """写帧线程挂掉后 inflight 永远归不了零，drain 必须靠错误短路而不是等满超时。"""
    dataset = _make_dataset(tmp_path / "ds")
    rec = _make_recorder(dataset)

    def boom(frame, *a, **kw):
        raise RuntimeError("编码器炸了")

    dataset.add_frame = boom
    rec.start_writer()
    try:
        for i in range(3):
            rec._enqueue_frame(_frame(i))
        t0 = time.perf_counter()
        with pytest.raises(RuntimeError, match="编码器炸了"):
            rec.drain_writer(timeout_s=30.0)
        assert time.perf_counter() - t0 < 10.0
    finally:
        rec.stop_writer()


def test_enqueue_drops_when_queue_full_without_leaking_inflight(tmp_path):
    """队列满了要丢帧，但不能把丢掉的帧算进 inflight —— 否则 drain 永远等不到 0。"""
    dataset = _make_dataset(tmp_path / "ds")
    rec = _make_recorder(dataset, queue_maxsize=4)
    # 不启动写帧线程，队列只进不出
    for i in range(10):
        rec._enqueue_frame(_frame(i))

    assert rec._frame_queue.qsize() == 4
    assert rec._inflight == 4
    assert rec._dropped_frames == 6
