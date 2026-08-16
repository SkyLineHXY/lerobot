"""并发回归：rollout 写入与 learner 采样同时进行时的不变量。

这三条都对应真实存在过的缺陷：buffer 无锁时 discard_episode 会在 learner
采样途中回退写指针；learner 线程异常被静默吞掉后训练永久空转；保存
checkpoint 时 learner 仍在做 optimizer step。
"""

import threading
import time

import pytest
import torch

from lerobot.policies.rlt import ActorCriticConfig, OnlineRLConfig, RLTAgent
from lerobot.rlt.learner import LearnerThread
from lerobot.rlt.replay_buffer import ChunkRecord, ChunkReplayBuffer

C, D, X_DIM = 4, 6, 12


def _cfg(**kw):
    ac = ActorCriticConfig(
        chunk_len=C, action_dim=D, proprio_dim=6, rl_token_dim=6, hidden_dim=16
    )
    base = {"ac": ac, "device": "cpu", "batch_size": 8, "subsample_stride": 2, "utd": 2}
    base.update(kw)
    return OnlineRLConfig(**base)


def _buffer(capacity=512):
    return ChunkReplayBuffer(
        capacity=capacity,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=0.97,
        stride=2,
        device="cpu",
    )


def _record(done=False):
    return ChunkRecord(
        xs=torch.randn(C // 2, X_DIM),
        actions=torch.randn(C, D),
        rewards=torch.zeros(C),
        ref_full=torch.randn(2 * C, D),
        done=done,
    )


def test_buffer_survives_concurrent_writes_samples_and_discards():
    """写线程 + 采样线程 + 丢弃同时跑，size/total_added 不能越界。"""
    buffer = _buffer()
    stop = threading.Event()
    errors = []

    def writer():
        try:
            while not stop.is_set():
                buffer.start_episode()
                for _ in range(4):
                    buffer.add_chunk(_record())
                buffer.add_chunk(_record(done=True))
                # 一半的 episode 被操作员丢弃，正是会回退写指针的那条路径
                if buffer.total_added % 2 == 0:
                    buffer.discard_episode()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def sampler():
        try:
            while not stop.is_set():
                batch = buffer.sample(8)
                # discard_episode 可能在 len() 与 sample() 之间清空 buffer，
                # 所以 sample 返回 None 是合法结果，不是错误
                if batch is not None:
                    assert batch["x"].shape == (8, X_DIM)
                    assert torch.isfinite(batch["x"]).all()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=sampler)]
    for t in threads:
        t.start()
    time.sleep(2.0)
    stop.set()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, errors
    assert 0 <= len(buffer) <= buffer.capacity
    assert buffer.total_added >= 0


def test_learner_exception_is_raised_on_the_main_thread():
    """learner 死掉必须能被主循环发现，而不是静默空转。"""
    buffer = _buffer()
    cfg = _cfg()
    buffer.start_episode()
    for _ in range(8):  # 要够 batch_size，否则 learner 一直在空转不会触发
        buffer.add_chunk(_record())

    class ExplodingAgent(RLTAgent):
        def update(self, batch, allow_actor=True):
            raise ValueError("boom")

    learner = LearnerThread(ExplodingAgent(cfg, device="cpu"), buffer, cfg, idle_sleep_s=0.001)
    learner.start()
    try:
        deadline = time.time() + 5.0
        while learner.is_alive() and time.time() < deadline:
            time.sleep(0.02)
        with pytest.raises(RuntimeError, match="learner thread died"):
            learner.raise_if_failed()
    finally:
        learner.stop()
        learner.join(timeout=5.0)


def test_pause_blocks_until_the_learner_is_between_updates():
    buffer = _buffer()
    cfg = _cfg()
    buffer.start_episode()
    for _ in range(6):
        buffer.add_chunk(_record())

    learner = LearnerThread(RLTAgent(cfg, device="cpu"), buffer, cfg, idle_sleep_s=0.001)
    learner.start()
    try:
        assert learner.pause(timeout=5.0)
        # 暂停期间 learner 不得再前进，否则 checkpoint 仍可能是撕裂的
        frozen = learner.updates
        time.sleep(0.2)
        assert learner.updates == frozen
        learner.resume()
    finally:
        learner.stop()
        learner.join(timeout=5.0)


def test_burst_cap_preserves_the_utd_invariant():
    """突发上限不能改变 '总更新数 == utd * 总 transition 数' 的不变量。"""
    buffer = _buffer()
    cfg = _cfg(utd=3)
    buffer.start_episode()
    for _ in range(10):
        buffer.add_chunk(_record())

    expected = cfg.utd * buffer.total_added
    learner = LearnerThread(
        RLTAgent(cfg, device="cpu"),
        buffer,
        cfg,
        idle_sleep_s=0.001,
        max_updates_per_burst=4,  # 远小于一次可用的更新量，强制多次突发
    )
    learner.start()
    try:
        deadline = time.time() + 20.0
        while learner.updates < expected and time.time() < deadline:
            time.sleep(0.02)
    finally:
        learner.stop()
        learner.join(timeout=5.0)

    assert learner.updates == expected, f"{learner.updates} != {expected}"
