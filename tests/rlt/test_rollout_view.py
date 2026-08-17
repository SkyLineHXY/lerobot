"""Rollout 视图与 warmup 期间的按需接管。

不开窗口：面板直接渲染到 numpy 画布上做断言（`_render` 不需要 `start()`）。接管
用假的 InterventionManager，验证两件真会咬人的事——中途接管必须当帧生效（等到
chunk 边界就是最多 C 步的延迟），以及仿真里操作员按成功必须真的变成 reward。
"""

import numpy as np
import pytest
import torch

from lerobot.rlt.teleop.base import InterventionManager, InterventionResult
from lerobot.rlt.view import RolloutStatusView, RolloutView

cv2 = pytest.importorskip("cv2")


def render(status, images=None):
    view = RolloutStatusView(camera_names=None, bgr=True, max_width=640, panel_height=200)
    view._cv2 = cv2
    return view._render(images, status)


def test_panel_renders_without_any_camera():
    """MockManipEnv 没有图像；视图必须退化成纯面板而不是崩掉。"""
    canvas = render({"phase": "RL", "env_steps": 10})
    assert canvas is not None
    assert canvas.shape[0] == 200
    assert canvas.any(), "面板整块是黑的，说明什么都没画上去"


def test_panel_renders_with_cameras_stacked_on_top():
    images = {"image": np.full((64, 64, 3), 128, np.uint8)}
    canvas = render({"phase": "WARMUP"}, images)
    # 图像条在上、面板在下
    assert canvas.shape[0] > 200


def test_panel_survives_a_status_with_nothing_in_it():
    assert render({}) is not None


def test_panel_survives_partial_status():
    """每个字段都是可选的：日志还没攒够时不能因为缺 key 就炸。"""
    assert render({"task": "pick up the bowl", "buffer": 10}) is not None
    assert render({"metrics": {}, "success_rate": None}) is not None


class FakeView:
    def __init__(self):
        self.steps = []
        self.quit_requested = False

    def set(self, **fields):
        pass

    def on_step(self, obs=None, **fields):
        self.steps.append(fields)
        return True


class OnDemandIntervention(InterventionManager):
    """第 `engage_at` 次询问之后开始要求接管，接管一次就交还。"""

    def __init__(self, engage_at):
        self.engage_at = engage_at
        self.checks = 0
        self.ran = 0

    def check(self):
        self.checks += 1
        return self.checks > self.engage_at

    def run_chunk(self, chunk_len):
        if not self.check():
            return None
        self.ran += 1
        return InterventionResult(
            action_chunk=torch.zeros(chunk_len, 7),
            obs_list=[{"t": i} for i in range(chunk_len)],
            rewards=torch.zeros(chunk_len),
            n_steps=chunk_len,
        )


def test_intervention_manager_notifies_each_step():
    seen = []
    manager = InterventionManager()
    manager.on_step = seen.append
    manager.notify_step({"t": 1})
    manager.notify_step({"t": 2})
    assert seen == [{"t": 1}, {"t": 2}]


def test_notify_step_is_a_noop_without_a_view():
    InterventionManager().notify_step({"t": 1})  # 不设 on_step 也不能抛


# ------------------------------------------------- 中途接管（本次改动的重点）
class CountingEnv:
    """记录跑了多少步、并在指定步数后开始要求接管的最小 env。"""

    max_episode_steps = 1000

    def __init__(self, engage_after):
        self.engage_after = engage_after
        self.steps = 0

    def step(self, action):
        self.steps += 1
        return {"t": self.steps}, 0.0, False

    def intervention_pending(self):
        return self.steps >= self.engage_after


def run_chunk_loop(env, chunk_len):
    """复刻 rollout.run_chunk 的执行循环，只保留退出条件。"""
    n_exec = 0
    for _ in range(chunk_len):
        env.step(None)
        n_exec += 1
        if env.intervention_pending():
            break
    return n_exec


def test_takeover_breaks_the_chunk_immediately():
    env = CountingEnv(engage_after=3)
    assert run_chunk_loop(env, 10) == 3, "接管请求必须当帧结束这个 chunk"


def test_chunk_runs_to_the_end_when_nobody_intervenes():
    env = CountingEnv(engage_after=10_000)
    assert run_chunk_loop(env, 10) == 10


def test_rollout_view_disabled_is_a_noop():
    view = RolloutView(env=None, enabled=False)
    view.start()
    view.set(phase="RL")
    assert view.on_step({"pixels": {}}) is True
    assert view.enabled is False
    view.stop()


class RecordingStatusView:
    """StatusView 的替身：只记录 start/stop 的调用序列，不碰 cv2。"""

    def __init__(self):
        self.calls = []
        self.enabled = False

    def start(self):
        self.calls.append("start")
        self.enabled = True

    def stop(self):
        self.calls.append("stop")
        self.enabled = False


def test_set_active_only_acts_on_transitions():
    """每个 chunk 都会调一次；`start()` 会重新 import cv2 并探测 DISPLAY，不能白付。"""
    view = RolloutView(env=None, enabled=True)
    inner = RecordingStatusView()
    view._view = inner

    view.set_active(False)  # warmup 起步就是关的，无事发生
    assert inner.calls == []
    view.set_active(True)
    view.set_active(True)
    assert inner.calls == ["start"]
    view.set_active(False)
    assert inner.calls == ["start", "stop"]
    view.set_active(True)
    assert inner.calls == ["start", "stop", "start"], "关掉之后必须还能再开"


def test_set_active_is_a_noop_when_the_view_was_never_enabled():
    view = RolloutView(env=None, enabled=False)
    view.set_active(True)
    assert view.enabled is False
    assert view.on_step({"pixels": {}}) is True


def test_rollout_view_tolerates_an_env_without_render_frames():
    class NoRender:
        pass

    view = RolloutView(env=NoRender(), enabled=False)
    assert view._frames({"anything": 1}) is None


def test_rollout_view_swallows_a_broken_render_frames():
    """渲染失败不能打断训练——机械臂还在动。"""

    class Broken:
        def render_frames(self, obs):
            raise RuntimeError("camera gone")

    assert RolloutView(env=Broken(), enabled=False)._frames({}) is None
