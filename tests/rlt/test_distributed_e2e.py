"""真的起一个 gRPC learner，跑通 op 上行 + 权重下行。

用线程而不是进程跑 learner：要验的是 wire 层（服务端能收到 op、能把权重推回来、
学习循环真的在动），进程隔离本身是 torch.multiprocessing 的事。用线程能让断言
失败时直接看到 traceback，不用去翻子进程的 stderr。
"""

import queue
import socket
import threading
import time

import torch

from lerobot.policies.rlt import ActorCriticConfig, OnlineRLConfig
from lerobot.rlt.distributed.client import (
    RemoteActorMirror,
    RemoteBufferSink,
    start_client_threads,
    stop_client,
)
from lerobot.rlt.distributed.learner_proc import serve
from lerobot.rlt.replay_buffer import ChunkRecord

C, D, PROPRIO, TOKEN = 4, 6, 6, 8
X_DIM = TOKEN + PROPRIO


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg():
    ac = ActorCriticConfig(
        chunk_len=C, action_dim=D, proprio_dim=PROPRIO, rl_token_dim=TOKEN, hidden_dim=16
    )
    return OnlineRLConfig(
        ac=ac, device="cpu", batch_size=8, subsample_stride=2, utd=2, buffer_capacity=512
    )


def _record(seed: int, base=0, done=False):
    g = torch.Generator().manual_seed(seed)
    offsets = [o for o in range(C) if (base + o) % 2 == 0]
    return ChunkRecord(
        xs=torch.randn(len(offsets), X_DIM, generator=g),
        x_offsets=torch.tensor(offsets, dtype=torch.long),
        refs=torch.randn(len(offsets), C, D, generator=g),
        aligned=torch.tensor([o == 0 for o in offsets], dtype=torch.bool),
        actions=torch.randn(C, D, generator=g),
        rewards=torch.zeros(C),
        done=done,
    )


def test_ops_reach_the_learner_and_weights_come_back(tmp_path):
    # 这台机器上 torch 默认开 48 个线程跑这个小 MLP，只有 ~7 updates/s；
    # 进程后端会把它设成 1（快 24 倍），线程内测试得自己设。
    torch.set_num_threads(1)
    cfg = _cfg()
    port = _free_port()
    shutdown = threading.Event()
    ready = threading.Event()

    server = threading.Thread(
        target=serve,
        kwargs={
            "cfg": cfg,
            "out_dir": str(tmp_path),
            "x_dim": X_DIM,
            "host": "127.0.0.1",
            "port": port,
            "device": "cpu",
            "parameters_push_hz": 50.0,
            "shutdown_event": shutdown,
            "ready_event": ready,
        },
        daemon=True,
    )
    server.start()
    assert ready.wait(30.0), "learner 没起来"

    op_queue: queue.Queue = queue.Queue()
    params_queue: queue.Queue = queue.Queue()
    client_shutdown = threading.Event()
    threads, channel = start_client_threads(
        "127.0.0.1", port, op_queue, params_queue, client_shutdown
    )

    sink = RemoteBufferSink(op_queue)
    mirror = RemoteActorMirror(cfg, "cpu", params_queue)
    try:
        sink.warmup = False
        sink.start_episode()
        for i in range(12):
            sink.add_chunk(_record(i, base=i * C))
        sink.add_chunk(_record(99, base=12 * C, done=True))
        sink.end_episode(torch.randn(X_DIM), torch.randn(C, D), success=True)

        # 等 learner 收到数据并真的做了梯度步
        deadline = time.time() + 60.0
        while time.time() < deadline:
            mirror.sync()
            if mirror.stats.get("updates", 0) > 0 and mirror.stats.get("buffer", 0) > 0:
                break
            time.sleep(0.1)

        assert mirror.stats.get("buffer", 0) > 0, f"learner 没收到 op：{mirror.stats}"
        assert mirror.stats.get("updates", 0) > 0, f"learner 没做梯度步：{mirror.stats}"
        assert mirror.version > 0, "权重没推回来"

        # 权重确实在变，而不是一直发同一份初始快照
        before = mirror.actor.state_dict()["trunk.0.weight"].clone()
        changed_deadline = time.time() + 30.0
        while time.time() < changed_deadline:
            mirror.sync()
            if not torch.allclose(before, mirror.actor.state_dict()["trunk.0.weight"]):
                break
            time.sleep(0.1)
        after = mirror.actor.state_dict()["trunk.0.weight"]
        assert not torch.allclose(before, after), "actor 权重一直没更新"
    finally:
        stop_client(threads, channel, client_shutdown)
        shutdown.set()
        server.join(timeout=30.0)

    assert (tmp_path / "rlt_agent.pt").exists(), "learner 退出时应写下 checkpoint"
