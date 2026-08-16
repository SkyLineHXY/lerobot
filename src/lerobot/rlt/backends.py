"""The two ways rollout and learning can be separated, behind one interface.

`train_online.py` drives a backend, not a `LearnerThread` — otherwise every
call site would have to know whether the buffer is a local object or a socket.
The five things the training loop actually needs are: somewhere to put chunks,
an actor to plan with, a way to say "still warming up", metrics to log, and a
health check.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import torch

from lerobot.policies.rlt import OnlineRLConfig, RLTAgent

from .learner import ActorMirror, LearnerThread
from .replay_buffer import ChunkReplayBuffer

logger = logging.getLogger(__name__)


class ThreadBackend:
    """Learner in a thread, buffer in this process. The reference implementation."""

    mode = "threads"

    def __init__(self, cfg, agent: RLTAgent, x_dim: int):
        rl: OnlineRLConfig = cfg.rl
        self.agent = agent
        self.buffer = ChunkReplayBuffer(
            capacity=rl.buffer_capacity,
            x_dim=x_dim,
            chunk_len=rl.ac.chunk_len,
            action_dim=rl.ac.action_dim,
            discount=rl.discount,
            stride=rl.subsample_stride,
            device=cfg.device,
            seed=cfg.seed,
        )
        if cfg.resume_buffer:
            self.buffer.load(cfg.resume_buffer)
            print(f"[stage2] resumed buffer with {len(self.buffer)} transitions")
        self.mirror = ActorMirror(rl, cfg.device)
        self.learner = LearnerThread(agent, self.buffer, rl)

    def start(self) -> None:
        self.mirror.sync(self.learner)
        self.learner.start()

    def sync_mirror(self) -> None:
        self.mirror.sync(self.learner)

    def set_warmup(self, warmup: bool) -> None:
        self.learner.allow_actor_updates = not warmup

    def metrics(self) -> dict[str, float]:
        return self.learner.metrics()

    def check_health(self) -> None:
        self.learner.raise_if_failed()

    def checkpoint(self, out_dir: Path) -> None:
        # The learner is mid-optimizer-step at arbitrary moments, so pause it or
        # the saved actor and critic can disagree.
        if not self.learner.pause():
            logger.warning("[stage2] learner did not pause in time; checkpoint may be torn")
        try:
            torch.save(self.agent.state_dict(), out_dir / "rlt_agent.pt")
            self.buffer.save(out_dir / "replay_buffer.pt")
        finally:
            self.learner.resume()

    def stop(self) -> None:
        self.learner.stop()
        self.learner.join(timeout=5.0)


class ProcessBackend:
    """Learner in its own process behind gRPC; the buffer goes with it.

    The rollout keeps only a queue. Checkpointing moves to the learner, which
    owns both the agent and the buffer — so unlike the threaded backend there is
    no window in which a checkpoint can catch an optimizer mid-step.
    """

    mode = "processes"

    def __init__(self, cfg, agent: RLTAgent, x_dim: int):
        import queue as _queue

        import torch.multiprocessing as mp

        from .distributed import RemoteActorMirror, RemoteBufferSink, start_client_threads
        from .distributed.learner_proc import serve

        conc = cfg.concurrency
        self._shutdown = threading.Event()
        self._op_queue: _queue.Queue = _queue.Queue(maxsize=conc.max_pending_ops)
        self._params_queue: _queue.Queue = _queue.Queue()

        ctx = mp.get_context("spawn")
        self._proc_shutdown = ctx.Event()
        self._proc = ctx.Process(
            target=_learner_entry,
            args=(serve, cfg, x_dim, self._proc_shutdown),
            name="rlt-learner",
            daemon=False,
        )
        self._proc.start()

        self.buffer = RemoteBufferSink(self._op_queue, max_pending=conc.max_pending_ops)
        self.mirror = RemoteActorMirror(cfg.rl, cfg.device, self._params_queue)
        self._threads, self._channel = start_client_threads(
            conc.learner_host,
            conc.learner_port,
            self._op_queue,
            self._params_queue,
            self._shutdown,
        )

    def start(self) -> None:
        self.mirror.sync()

    def sync_mirror(self) -> None:
        if self.mirror.sync():
            # The learner reports its own buffer size; without this the training
            # log would show 0 transitions for the whole run.
            self.buffer.remote_size = int(self.mirror.stats.get("buffer", 0))
            self.buffer.total_added = int(self.mirror.stats.get("total_added", 0))

    def set_warmup(self, warmup: bool) -> None:
        self.buffer.warmup = warmup

    def metrics(self) -> dict[str, float]:
        return dict(self.mirror.stats)

    def check_health(self) -> None:
        if not self._proc.is_alive():
            raise RuntimeError(
                f"RLT learner process exited with code {self._proc.exitcode}. "
                "Its traceback is in that process's stderr."
            )
        if self._shutdown.is_set():
            raise RuntimeError("RLT learner stream died; see the log above.")

    def checkpoint(self, out_dir: Path) -> None:
        """No-op: the learner process owns the agent and the buffer."""

    def stop(self) -> None:
        from .distributed.client import stop_client

        # Streams first, then the channel, then the process: the reverse order
        # aborts in-flight RPCs inside the gRPC C core.
        stop_client(self._threads, self._channel, self._shutdown)
        self._proc_shutdown.set()
        self._proc.join(timeout=30.0)
        if self._proc.is_alive():
            logger.warning("[stage2] learner process did not exit; terminating")
            self._proc.terminate()


def _learner_entry(serve, cfg, x_dim: int, shutdown_event) -> None:
    """Child-process entry point. Must be importable at module level for spawn."""
    from lerobot.utils.utils import init_logging

    init_logging()
    threads = cfg.concurrency.learner_torch_threads
    if threads > 0:
        torch.set_num_threads(threads)
    serve(
        cfg=cfg.rl,
        out_dir=cfg.out,
        x_dim=x_dim,
        host=cfg.concurrency.learner_host,
        port=cfg.concurrency.learner_port,
        device=cfg.device,
        seed=cfg.seed,
        parameters_push_hz=cfg.concurrency.parameters_push_hz,
        shutdown_event=shutdown_event,
        resume_buffer=cfg.resume_buffer,
    )


def make_backend(cfg, agent: RLTAgent, x_dim: int):
    mode = cfg.concurrency.mode
    if mode == "threads":
        return ThreadBackend(cfg, agent, x_dim)
    if mode == "processes":
        return ProcessBackend(cfg, agent, x_dim)
    raise ValueError(f"unknown concurrency mode {mode!r}; use threads/processes")
