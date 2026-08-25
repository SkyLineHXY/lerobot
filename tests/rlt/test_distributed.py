"""多进程后端：wire 协议与 buffer 语义等价性。

最重要的一条是 `test_op_stream_matches_the_in_process_buffer`：进程版把 buffer
搬到了另一侧，如果 op 流重放出来的 buffer 和本地直接写出来的不一致，症状会是
"RL 就是训不动"，而不是任何一条报错。所以这里逐字段对比两条路径。

不起 gRPC 服务，只测序列化 + 重放；网络那一层是 lerobot.rl 已有的通用代码。
"""

import queue

import pytest
import torch

from lerobot.policies.rlt import ActorCriticConfig, OnlineRLConfig, TransitionSource
from lerobot.rlt.distributed.client import RemoteBufferSink
from lerobot.rlt.distributed.learner_proc import RemoteLearner, apply_op
from lerobot.rlt.distributed.messages import (
    bytes_to_ops,
    bytes_to_params,
    ops_to_bytes,
    params_to_bytes,
    record_from_op,
)
from lerobot.rlt.replay_buffer import ChunkRecord, ChunkReplayBuffer

C, D, X_DIM = 4, 6, 12
FIELDS = (
    "x", "action", "ref", "rewards", "x_next", "ref_next", "done", "mc_return",
    "intervention",
)
INT_FIELDS = (
    "source_chunk", "source", "phase", "success", "episode_id", "mc_valid", "aligned",
)


def _buffer(capacity=256):
    return ChunkReplayBuffer(
        capacity=capacity,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=0.97,
        stride=2,
        device="cpu",
    )


def test_remote_learner_initialisation_uses_the_configured_seed(tmp_path):
    ac = ActorCriticConfig(
        chunk_len=C,
        action_dim=D,
        proprio_dim=4,
        rl_token_dim=X_DIM - 4,
        hidden_dim=16,
    )
    cfg = OnlineRLConfig(ac=ac, device="cpu", buffer_capacity=16, batch_size=4)
    a = RemoteLearner(cfg, tmp_path / "a", X_DIM, "cpu", seed=7)
    b = RemoteLearner(cfg, tmp_path / "b", X_DIM, "cpu", seed=7)
    c = RemoteLearner(cfg, tmp_path / "c", X_DIM, "cpu", seed=8)

    sa, sb, sc = a.agent.actor.state_dict(), b.agent.actor.state_dict(), c.agent.actor.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


def _record(seed: int, base=0, done=False, source=int(TransitionSource.RL)):
    g = torch.Generator().manual_seed(seed)
    offsets = [o for o in range(C) if (base + o) % 2 == 0]
    return ChunkRecord(
        xs=torch.randn(len(offsets), X_DIM, generator=g),
        x_offsets=torch.tensor(offsets, dtype=torch.long),
        refs=torch.randn(len(offsets), C, D, generator=g),
        aligned=torch.tensor([o == 0 for o in offsets], dtype=torch.bool),
        actions=torch.randn(C, D, generator=g),
        rewards=torch.zeros(C),
        source=source,
        done=done,
    )


def _drain(op_queue: queue.Queue) -> list[dict]:
    """把队列走一遍真实的序列化往返，模拟过网络。"""
    ops = []
    while not op_queue.empty():
        ops.append(op_queue.get_nowait())
    return bytes_to_ops(ops_to_bytes(ops))


def _assert_buffers_equal(a: ChunkReplayBuffer, b: ChunkReplayBuffer):
    assert len(a) == len(b)
    assert a.total_added == b.total_added
    for name in FIELDS:
        left, right = getattr(a, name)[: len(a)], getattr(b, name)[: len(b)]
        assert torch.allclose(left, right), f"字段 {name} 不一致"
    for name in INT_FIELDS:
        left, right = getattr(a, name)[: len(a)], getattr(b, name)[: len(b)]
        assert torch.equal(left, right), f"字段 {name} 不一致"


@pytest.mark.parametrize("with_discard", [False, True])
def test_op_stream_matches_the_in_process_buffer(with_discard):
    """同一串操作，本地直写 vs 经 op 流重放，必须逐字段相同。"""
    local = _buffer()
    remote = _buffer()
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)

    def both(fn_name, *args):
        getattr(local, fn_name)(*args)
        getattr(sink, fn_name)(*args)

    x_last, ref_last = torch.randn(X_DIM), torch.randn(C, D)

    both("start_episode", True)
    for i in range(4):
        both("add_chunk", _record(i, base=i * C))
    if with_discard:
        both("discard_episode")
    else:
        both("end_episode", x_last, ref_last, False)
    both("start_episode", False)
    for i in range(4, 8):
        both("add_chunk", _record(i, base=(i - 4) * C, source=int(TransitionSource.HUMAN)))
    both("add_chunk", _record(99, base=4 * C, done=True))
    both("end_episode", x_last, ref_last, True)

    for op in _drain(op_queue):
        apply_op(remote, op)

    _assert_buffers_equal(local, remote)
    assert len(remote) > 0, "测试本身要有数据才有意义"


def test_truncation_carries_the_real_next_state():
    """end_episode 的 x_last 必须过得去：丢了它这条 transition 会被整条丢弃。"""
    local, remote = _buffer(), _buffer()
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    x_last, ref_last = torch.randn(X_DIM), torch.randn(C, D)

    for buf in (local, sink):
        buf.start_episode()
        buf.add_chunk(_record(1, base=0))
        buf.add_chunk(_record(2, base=C))
        buf.end_episode(x_last, ref_last)

    for op in _drain(op_queue):
        apply_op(remote, op)

    _assert_buffers_equal(local, remote)
    assert len(remote) > 0


def test_record_survives_the_roundtrip():
    rec = _record(7, done=True, source=int(TransitionSource.HUMAN))
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    sink.start_episode()
    sink.add_chunk(rec)

    ops = _drain(op_queue)
    back = record_from_op(ops[1])
    assert torch.allclose(back.xs, rec.xs)
    assert torch.equal(back.x_offsets, rec.x_offsets)
    assert torch.allclose(back.actions, rec.actions)
    assert torch.allclose(back.refs, rec.refs)
    assert torch.equal(back.aligned, rec.aligned)
    assert back.done is True
    assert back.source == int(TransitionSource.HUMAN)


def test_final_anchor_survives_the_wire():
    """每个越界的 partial window 都拿它当 bootstrap，丢了整条尾巴就静默消失。"""
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    x_last, ref_last = torch.randn(X_DIM), torch.randn(C, D)
    sink.start_episode()
    sink.add_chunk(_record(0))
    sink.end_episode(x_last, ref_last)

    end = [op for op in _drain(op_queue) if op["kind"] == "end"][0]
    assert torch.allclose(end["x_last"], x_last)
    assert torch.allclose(end["ref_last"], ref_last)


def test_episode_end_carries_the_success_label():
    """success 只进采样分层与统计，但丢了它 stratified 的 warmup/human 池会错位。"""
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    sink.start_episode()
    sink.add_chunk(_record(3))
    sink.end_episode(torch.randn(X_DIM), torch.randn(C, D), success=True)
    end = [op for op in _drain(op_queue) if op["kind"] == "end"][0]
    assert end["success"] is True
    assert "ref_last" in end


def test_ops_decode_without_pickle():
    """op 流必须能用 weights_only=True 读回来——它是从 socket 上来的。"""
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    sink.start_episode()
    sink.add_chunk(_record(0))
    payload = ops_to_bytes([op_queue.get_nowait(), op_queue.get_nowait()])
    decoded = bytes_to_ops(payload)
    assert [op["kind"] for op in decoded] == ["start", "chunk"]


def test_warmup_flag_rides_along_with_each_chunk():
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    sink.start_episode()
    sink.warmup = True
    sink.add_chunk(_record(0))
    sink.warmup = False
    sink.add_chunk(_record(1))

    ops = _drain(op_queue)
    chunks = [op for op in ops if op["kind"] == "chunk"]
    assert [op["warmup"] for op in chunks] == [True, False]


def test_params_roundtrip_carries_stats():
    state = {"net.0.weight": torch.randn(4, 3), "net.0.bias": torch.randn(4)}
    stats = {"critic_loss": 0.5, "buffer": 128.0, "updates": 42.0}
    back_state, back_stats = bytes_to_params(params_to_bytes(state, stats))

    assert set(back_state) == set(state)
    assert torch.allclose(back_state["net.0.weight"], state["net.0.weight"])
    assert back_stats == stats


def test_sink_blocks_rather_than_dropping_when_full():
    """队列满了要阻塞（丢数据会表现成 'RL 就是不 work'），这里只验证不静默丢。"""
    op_queue: queue.Queue = queue.Queue(maxsize=2)
    sink = RemoteBufferSink(op_queue, max_pending=2)
    sink.start_episode()
    sink.add_chunk(_record(0))
    assert op_queue.full()
    # 第三条会阻塞；先腾出一格再写，确认它确实入队而不是被丢弃
    op_queue.get_nowait()
    sink.add_chunk(_record(1))
    assert op_queue.qsize() == 2
