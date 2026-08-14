"""遥操作线程与采集线程解耦的回归测试。

改造前 `dataset.fps` 同时是数据集帧率和从臂指令频率，双臂为了扛写盘负载把 fps 降到
25，指令周期反而从 33ms 涨到 40ms —— 位置模式下阶跃幅度正比于周期，所以双臂比单臂
更顿。这里钉死：指令频率由主臂新样本驱动，与 dataset.fps 无关。

硬件全部用假件，遵循 tests/rlt/test_piper_ctrl_mode_guard.py 的模式。
"""

import threading
import time

import numpy as np
import pytest

from lerobot.scripts.lerobot_record_piper import (
    ArmSpec,
    PiperRecorder,
    RecordPiperConfig,
    TeleopSpec,
)


class FakeLimiterPair:
    """假的 ArmPair：主臂按固定节拍产生新样本，从臂记录收到的每一条指令。"""

    def __init__(self, name="left", leader_hz=200.0):
        self.spec = ArmSpec(name=name)
        self.sent: list[np.ndarray] = []
        self._leader_hz = leader_hz
        self._t0 = time.perf_counter()
        self._last_leader_ts = -1.0
        from lerobot.rlt.piper_env import JointSlewLimiter

        self.limiter = JointSlewLimiter(max_vel_rad_s=100.0, max_lead_rad=0.0)

    def leader_timestamp(self) -> float:
        # 用"经过了多少个主臂周期"当时间戳，等价于反馈帧的 time_stamp。
        # 从 1 起步：真实时间戳是纪元秒，0 被用作"SDK 没给时间戳"的哨兵值。
        return 1.0 + float(int((time.perf_counter() - self._t0) * self._leader_hz))

    def read_leader(self) -> dict:
        t = time.perf_counter() - self._t0
        return {f"joint_{i}.pos": 0.1 * t for i in range(1, 8)}

    def read_follower(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def send(self, joints: np.ndarray) -> None:
        self.sent.append(np.asarray(joints).copy())

    def configure_limiter(self, teleop):
        pass


def _recorder(n_arms=1, fps=30, leader_hz=200.0) -> PiperRecorder:
    cfg = RecordPiperConfig(
        arms=[ArmSpec(name=f"arm{i}") for i in range(n_arms)],
        teleop=TeleopSpec(max_hz=500.0, idle_poll_s=0.0005),
    )
    cfg.dataset.fps = fps
    rec = PiperRecorder.__new__(PiperRecorder)
    rec.cfg = cfg
    rec.dt = 1.0 / fps
    rec.n_arms = n_arms
    rec.pairs = [FakeLimiterPair(f"arm{i}", leader_hz) for i in range(n_arms)]
    rec._arm_snapshots = [None] * n_arms
    rec._snapshot_lock = threading.Lock()
    rec._teleop_threads = []
    rec._teleop_stop = threading.Event()
    rec._teleop_error = []
    rec._teleop_periods = [[] for _ in range(n_arms)]
    from collections import deque

    rec._traj_cmd = [deque(maxlen=20000) for _ in range(n_arms)]
    rec._traj_meas = [deque(maxlen=20000) for _ in range(n_arms)]
    rec._saturated = []
    return rec


def test_command_rate_tracks_leader_not_dataset_fps():
    """dataset.fps=5 也不该把指令频率拉低到 5Hz。"""
    rec = _recorder(fps=5, leader_hz=200.0)
    rec.start_teleop()
    try:
        time.sleep(1.0)
    finally:
        rec.stop_teleop()
    assert not rec._teleop_error

    n = len(rec.pairs[0].sent)
    # 主臂 200Hz，允许调度抖动，但必须远高于 dataset.fps=5
    assert n > 100, f"指令只发了 {n} 次，说明还是被采集帧率拖着走"


def test_no_duplicate_sends_without_a_fresh_leader_sample():
    """主臂没有新样本时不该重复下发同一个目标，白占 CAN 帧。"""
    rec = _recorder(fps=30, leader_hz=20.0)  # 主臂只有 20Hz
    rec.start_teleop()
    try:
        time.sleep(1.0)
    finally:
        rec.stop_teleop()

    n = len(rec.pairs[0].sent)
    assert 10 <= n <= 40, f"主臂 20Hz 却发了 {n} 次，说明没有按新样本驱动"


def test_snapshot_requires_every_arm_to_have_published():
    """缺一条臂的帧写进数据集就是错的。"""
    rec = _recorder(n_arms=2)
    assert rec.latest_snapshot() is None

    rec._arm_snapshots[0] = (np.zeros(7, np.float32), np.ones(7, np.float32))
    assert rec.latest_snapshot() is None

    rec._arm_snapshots[1] = (np.zeros(7, np.float32), np.ones(7, np.float32))
    state, cmd = rec.latest_snapshot()
    assert state.shape == (14,)
    assert cmd.shape == (14,)


def test_each_arm_is_driven_by_its_own_leader():
    """左臂来了新样本不该顺带把右臂也重发一遍。"""
    rec = _recorder(n_arms=2)
    rec.pairs[0]._leader_hz = 200.0
    rec.pairs[1]._leader_hz = 25.0
    rec.start_teleop()
    try:
        time.sleep(1.0)
    finally:
        rec.stop_teleop()

    fast, slow = len(rec.pairs[0].sent), len(rec.pairs[1].sent)
    assert fast > 3 * slow, f"两条臂被耦合了：fast={fast} slow={slow}"


def test_teleop_thread_failure_surfaces_instead_of_being_swallowed():
    rec = _recorder()

    def boom(*a, **kw):
        raise RuntimeError("CAN 掉了")

    rec.pairs[0].read_follower = boom
    rec.start_teleop()
    try:
        time.sleep(0.3)
    finally:
        rec.stop_teleop()

    assert rec._teleop_error
    assert isinstance(rec._teleop_error[0], RuntimeError)


def test_legacy_max_joint_step_rad_is_converted_to_velocity():
    """旧键含义随频率漂移，必须折算成 rad/s 并告警。"""
    cfg = RecordPiperConfig()
    cfg.dataset.fps = 30
    cfg.collection.max_joint_step_rad = 0.05
    rec = PiperRecorder.__new__(PiperRecorder)
    rec.cfg = cfg
    rec._resolve_rate_limit()
    assert cfg.teleop.max_joint_vel_rad_s == pytest.approx(1.5)


def test_absent_legacy_key_leaves_velocity_limit_alone():
    cfg = RecordPiperConfig()
    cfg.teleop.max_joint_vel_rad_s = 6.0
    rec = PiperRecorder.__new__(PiperRecorder)
    rec.cfg = cfg
    rec._resolve_rate_limit()
    assert cfg.teleop.max_joint_vel_rad_s == pytest.approx(6.0)
