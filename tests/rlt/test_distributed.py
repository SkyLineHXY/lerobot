"""多进程后端：wire 协议与 buffer 语义等价性。

最重要的一条是 `test_op_stream_matches_the_in_process_buffer`：进程版把 buffer
搬到了另一侧，如果 op 流重放出来的 buffer 和本地直接写出来的不一致，症状会是
"RL 就是训不动"，而不是任何一条报错。所以这里逐字段对比两条路径。

不起 gRPC 服务，只测序列化 + 重放；网络那一层是 lerobot.rl 已有的通用代码。
"""

import queue

import pytest
import torch

from lerobot.rlt.distributed.client import RemoteBufferSink
from lerobot.rlt.distributed.learner_proc import apply_op
from lerobot.rlt.distributed.messages import (
    bytes_to_ops,
    bytes_to_params,
    ops_to_bytes,
    params_to_bytes,
    record_from_op,
)
from lerobot.rlt.replay_buffer import ChunkRecord, ChunkReplayBuffer

C, D, X_DIM = 4, 6, 12
FIELDS = ("x", "action", "ref", "reward_disc", "x_next", "ref_next", "done", "actual_steps")


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


def _record(seed: int, done=False):
    g = torch.Generator().manual_seed(seed)
    return ChunkRecord(
        xs=torch.randn(C // 2, X_DIM, generator=g),
        actions=torch.randn(C, D, generator=g),
        rewards=torch.zeros(C),
        ref_full=torch.randn(2 * C, D, generator=g),
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

    both("start_episode")
    for i in range(4):
        both("add_chunk", _record(i))
    if with_discard:
        both("discard_episode")
    both("start_episode")
    for i in range(4, 8):
        both("add_chunk", _record(i))
    both("add_chunk", _record(99, done=True))

    for op in _drain(op_queue):
        apply_op(remote, op)

    _assert_buffers_equal(local, remote)
    assert len(remote) > 0, "测试本身要有数据才有意义"


def test_truncation_carries_the_real_next_state():
    """end_episode 的 x_last 必须过得去：丢了它这条 transition 会被整条丢弃。"""
    local, remote = _buffer(), _buffer()
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    x_last = torch.randn(X_DIM)

    for buf in (local, sink):
        buf.start_episode()
        buf.add_chunk(_record(1))
        buf.add_chunk(_record(2))
        buf.end_episode(x_last)

    for op in _drain(op_queue):
        apply_op(remote, op)

    _assert_buffers_equal(local, remote)
    assert len(remote) > 0


def test_record_survives_the_roundtrip():
    rec = _record(7, done=True)
    rec.done_step = 3
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    sink.start_episode()
    sink.add_chunk(rec)

    ops = _drain(op_queue)
    back = record_from_op(ops[1])
    assert torch.allclose(back.xs, rec.xs)
    assert torch.allclose(back.actions, rec.actions)
    assert torch.allclose(back.ref_full, rec.ref_full)
    assert back.done is True
    assert back.done_step == 3
    assert back.n_executed == 3


def test_done_step_none_roundtrips_as_none():
    """done_step 在线上编码成 -1；解回来必须还是 None，否则 n_executed 会算错。"""
    op_queue: queue.Queue = queue.Queue()
    sink = RemoteBufferSink(op_queue)
    sink.start_episode()
    sink.add_chunk(_record(3))
    back = record_from_op(_drain(op_queue)[1])
    assert back.done_step is None
    assert back.n_executed == C


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
