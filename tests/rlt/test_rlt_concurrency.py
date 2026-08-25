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
from lerobot.rlt.learner import LearnerThread, burst_size
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


def _record(base=0, done=False):
    offsets = [o for o in range(C) if (base + o) % 2 == 0]
    return ChunkRecord(
        xs=torch.randn(len(offsets), X_DIM),
        x_offsets=torch.tensor(offsets, dtype=torch.long),
        refs=torch.randn(len(offsets), C, D),
        aligned=torch.tensor([o == 0 for o in offsets], dtype=torch.bool),
        actions=torch.randn(C, D),
        rewards=torch.zeros(C),
        done=done,
    )


def _fill(buffer, chunks=8):
    """一整集：只有 end_episode 才会真正落库。"""
    buffer.start_episode()
    for i in range(chunks):
        buffer.add_chunk(_record(base=i * C))
    buffer.end_episode(torch.randn(X_DIM), torch.randn(C, D))


def test_buffer_survives_concurrent_writes_samples_and_discards():
    """写线程 + 采样线程 + 丢弃同时跑，size/total_added 不能越界。"""
    buffer = _buffer()
    stop = threading.Event()
    errors = []

    def writer():
        try:
            while not stop.is_set():
                buffer.start_episode()
                for i in range(4):
                    buffer.add_chunk(_record(base=i * C))
                buffer.add_chunk(_record(base=4 * C, done=True))
                # 一半的 episode 被操作员丢弃，正是和采样竞争的那条路径
                if buffer.total_added % 2 == 0:
                    buffer.discard_episode()
                else:
                    buffer.end_episode(torch.randn(X_DIM), torch.randn(C, D))
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
    _fill(buffer, chunks=8)  # 要够 batch_size，否则 learner 一直在空转不会触发

    class ExplodingAgent(RLTAgent):
        def update(self, batch, warmup=False):
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
    _fill(buffer, chunks=6)

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
    _fill(buffer, chunks=10)

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


# ------------------------------------------------------ warmup update budget
def test_burst_size_paces_by_utd_when_data_is_flowing():
    assert burst_size(pending=3, utd=5, max_updates_per_burst=0) == (15, 3)


def test_burst_size_caps_in_transitions_not_updates():
    """按 update 截断会记掉半条 transition，`updates == utd * total_added` 就破了。"""
    n_updates, consumed = burst_size(pending=100, utd=5, max_updates_per_burst=32)
    assert n_updates == consumed * 5
    assert n_updates <= 32


def test_warmup_keeps_grinding_on_the_data_it_already_has():
    """openpi-RLT 的 warmup_post_collect_updates：UTD 额度耗尽后仍要磨够步数。"""
    n_updates, consumed = burst_size(
        pending=0, utd=5, max_updates_per_burst=32, warmup_deficit=20_000
    )
    assert n_updates == 32
    assert consumed == 0, "补足的步数不能记账，否则 online 阶段的 UTD 会被提前扣掉"


def test_no_warmup_deficit_and_no_data_means_no_updates():
    assert burst_size(pending=0, utd=5, max_updates_per_burst=32) == (0, 0)


def test_warmup_deficit_does_not_displace_real_data():
    """两者同时存在时先走 UTD，补足只在没有新数据时兜底。"""
    n_updates, consumed = burst_size(
        pending=2, utd=5, max_updates_per_burst=0, warmup_deficit=20_000
    )
    assert (n_updates, consumed) == (10, 2)
