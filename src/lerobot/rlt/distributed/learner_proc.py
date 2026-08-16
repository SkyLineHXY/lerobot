"""Learner process: owns the replay buffer, the agent, and the checkpoints.

Runs `LearnerService` (reused unchanged from `lerobot.rl` — it is generic byte
plumbing) on a gRPC server, applies incoming buffer ops, and pushes actor
weights back.

Two things move here that the threaded backend keeps on the rollout side:

* **The buffer.** So `discard_episode` still operates on the real buffer, in
  stream order — see `messages.py`.
* **Checkpointing.** This process owns both the agent and the buffer, so there
  is no cross-thread window in which `state_dict()` can catch an optimizer
  mid-step. The `pause()` / `resume()` dance the threaded backend needs simply
  does not apply.

What this backend does *not* fix: the two processes still share the GPU. It
removes GIL contention and the shared-CUDA-stream drain, not compute
contention, so `max_updates_per_burst` still earns its keep.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import torch

from lerobot.policies.rlt import OnlineRLConfig, RLTAgent
from lerobot.rl.learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import MAX_MESSAGE_SIZE

from ..replay_buffer import ChunkReplayBuffer
from .messages import CHUNK, DISCARD, END, START, bytes_to_ops, params_to_bytes, record_from_op

logger = logging.getLogger(__name__)


def apply_op(buffer: ChunkReplayBuffer, op: dict) -> int:
    """Replay one buffer op. Returns transitions dropped (only ever from discard)."""
    kind = op["kind"]
    if kind == START:
        buffer.start_episode()
    elif kind == CHUNK:
        buffer.add_chunk(record_from_op(op))
    elif kind == END:
        buffer.end_episode(op.get("x_last"))
    elif kind == DISCARD:
        return buffer.discard_episode()
    else:
        raise ValueError(f"unknown buffer op {kind!r}")
    return 0


class RemoteLearner:
    """Consumes buffer ops, runs gradient steps, publishes actor weights."""

    def __init__(
        self,
        cfg: OnlineRLConfig,
        out_dir: Path,
        x_dim: int,
        device: str,
        seed: int = 0,
        publish_every: int = 10,
        max_updates_per_burst: int = 32,
        save_every_updates: int = 2000,
        resume_buffer: str | None = None,
    ):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.publish_every = publish_every
        self.max_updates_per_burst = max_updates_per_burst
        self.save_every_updates = save_every_updates

        self.agent = RLTAgent(cfg, device=device)
        self.buffer = ChunkReplayBuffer(
            capacity=cfg.buffer_capacity,
            x_dim=x_dim,
            chunk_len=cfg.ac.chunk_len,
            action_dim=cfg.ac.action_dim,
            discount=cfg.discount,
            stride=cfg.subsample_stride,
            device=device,
            seed=seed,
        )
        if resume_buffer:
            self.buffer.load(resume_buffer)
            logger.info("[learner] resumed buffer with %d transitions", len(self.buffer))

        self.updates = 0
        self._consumed = 0
        self._last_metrics: dict[str, float] = {}
        # Warmup collects with the base VLA; the critic learns from it, the
        # actor waits. The rollout owns that decision, so it rides along with
        # the ops rather than being recomputed here.
        self.allow_actor_updates = False

    # ----------------------------------------------------------------- ops
    def drain_ops(self, transition_queue: queue.Queue, budget: int = 64) -> None:
        for _ in range(budget):
            try:
                payload = transition_queue.get_nowait()
            except queue.Empty:
                return
            for op in bytes_to_ops(payload):
                if "warmup" in op:
                    self.allow_actor_updates = not op["warmup"]
                dropped = apply_op(self.buffer, op)
                if dropped:
                    logger.info("[learner] operator discarded %d transitions", dropped)

    # ------------------------------------------------------------ updates
    def train_burst(self) -> int:
        cfg = self.cfg
        pending = self.buffer.total_added - self._consumed
        if len(self.buffer) < cfg.batch_size or pending <= 0:
            return 0
        if self.max_updates_per_burst > 0 and cfg.utd > 0:
            pending = min(pending, max(1, self.max_updates_per_burst // cfg.utd))
        n_updates = int(cfg.utd * pending)
        self._consumed += pending

        for _ in range(n_updates):
            batch = self.buffer.sample(cfg.batch_size)
            if batch is None:  # operator discarded the episode mid-burst
                break
            self._last_metrics = self.agent.update(
                batch, allow_actor=self.allow_actor_updates
            )
            self.updates += 1
        return n_updates

    def stats(self) -> dict[str, float]:
        out = dict(self._last_metrics)
        out["updates"] = float(self.updates)
        out["buffer"] = float(len(self.buffer))
        out["total_added"] = float(self.buffer.total_added)
        return out

    def publish(self, parameters_queue: queue.Queue) -> None:
        state = {k: v.detach().cpu().clone() for k, v in self.agent.actor.state_dict().items()}
        parameters_queue.put(params_to_bytes(state, self.stats()))

    def save(self) -> None:
        torch.save(self.agent.state_dict(), self.out_dir / "rlt_agent.pt")
        self.buffer.save(self.out_dir / "replay_buffer.pt")


def serve(
    cfg: OnlineRLConfig,
    out_dir: str,
    x_dim: int,
    host: str,
    port: int,
    device: str,
    seed: int = 0,
    parameters_push_hz: float = 10.0,
    shutdown_event: threading.Event | None = None,
    resume_buffer: str | None = None,
    ready_event: threading.Event | None = None,
) -> None:
    """Run the learner until `shutdown_event` is set (blocking)."""
    shutdown_event = shutdown_event or threading.Event()
    parameters_queue: queue.Queue = queue.Queue()
    transition_queue: queue.Queue = queue.Queue()
    interaction_queue: queue.Queue = queue.Queue()

    learner = RemoteLearner(
        cfg, Path(out_dir), x_dim, device, seed=seed, resume_buffer=resume_buffer
    )
    # Publish once before serving so the rollout's first chunk is planned with
    # real weights instead of waiting a full push interval for the actor.
    learner.publish(parameters_queue)

    push_period = 1.0 / max(parameters_push_hz, 1e-6)
    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=push_period,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_queue,
        # LearnerService busy-spins on an empty parameters queue, logging once
        # per pass. Its 1 ms default turns that into ~1000 log lines a second
        # for a whole training run; pacing it to the push period does not.
        queue_get_timeout=max(push_period / 2, 0.05),
    )
    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )
    services_pb2_grpc.add_LearnerServiceServicer_to_server(service, server)
    # Returns 0 on failure rather than raising; without the check a busy port
    # shows up much later as the rollout timing out against nothing.
    if server.add_insecure_port(f"{host}:{port}") == 0:
        raise RuntimeError(f"[learner] could not bind {host}:{port} — is it already in use?")
    server.start()
    logger.info("[learner] gRPC server listening on %s:%d", host, port)
    if ready_event is not None:
        ready_event.set()

    last_saved = 0
    last_published = 0
    try:
        while not shutdown_event.is_set():
            learner.drain_ops(transition_queue)
            did = learner.train_burst()
            if did == 0:
                # Nothing to train on. Sleeping here is what keeps this process
                # off the GPU entirely while the rollout is mid-chunk.
                time.sleep(0.002)
            # Counted, not modulo: updates advance in bursts, so `updates % N`
            # skips straight past the multiples and would never publish again.
            if learner.updates - last_published >= learner.publish_every:
                learner.publish(parameters_queue)
                last_published = learner.updates
            if learner.updates - last_saved >= learner.save_every_updates:
                learner.save()
                last_saved = learner.updates
    finally:
        for label, teardown in (
            ("checkpoint", learner.save),
            ("grpc", lambda: server.stop(SHUTDOWN_TIMEOUT)),
        ):
            try:
                teardown()
            except Exception:
                logger.exception("[learner] %s teardown failed", label)
    logger.info("[learner] stopped after %d updates", learner.updates)
