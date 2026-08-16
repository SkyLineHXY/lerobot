"""Rollout-process side of the split: stream buffer ops out, pull weights in.

`RemoteBufferSink` is a drop-in for `ChunkReplayBuffer` as far as
`RolloutWorker` is concerned — same four mutating methods — but it enqueues the
calls instead of applying them. `RemoteActorMirror` is likewise a drop-in for
`ActorMirror`, reading published weights off the parameters queue rather than
out of a `LearnerThread`.

Keeping both duck-typed means `rollout.py` and the training loop are identical
in both concurrency modes; only the wiring in `train_online.py` differs.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import grpc
import torch

from lerobot.policies.rlt import ChunkActor, OnlineRLConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    grpc_channel_options,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
)

from ..replay_buffer import ChunkRecord
from .messages import (
    bytes_to_params,
    chunk_op,
    discard_op,
    end_op,
    ops_to_bytes,
    start_op,
)

logger = logging.getLogger(__name__)


class RemoteBufferSink:
    """Looks like a ChunkReplayBuffer to the rollout worker; ships ops instead.

    `total_added` and `__len__` report what the *learner* last told us, so the
    training log still shows a real buffer size. They lag by one parameters
    push, which is fine for logging and is never used for control decisions on
    this side — the UTD pacer lives with the buffer, in the learner.
    """

    def __init__(self, out_queue: queue.Queue, max_pending: int = 4096):
        self._queue = out_queue
        self._max_pending = max_pending
        self._episode_id = 0
        self._sent_this_episode = 0
        self.remote_size = 0
        self.total_added = 0
        # Set by the training loop each iteration, exactly as it sets
        # `LearnerThread.allow_actor_updates` in the threaded backend.
        self.warmup = True

    def __len__(self) -> int:
        return self.remote_size

    def _emit(self, op: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(op)
        except queue.Full:
            # Dropping rollout data silently would look like "RL just doesn't
            # work"; a blocked arm is the lesser evil and the operator can see it.
            logger.warning("Transition queue is full (%d); blocking the rollout.", self._max_pending)
            self._queue.put(op)

    def start_episode(self) -> None:
        self._episode_id += 1
        self._sent_this_episode = 0
        self._emit(start_op(self._episode_id))

    def add_chunk(self, rec: ChunkRecord) -> None:
        self._sent_this_episode += 1
        self._emit(chunk_op(self._episode_id, rec, warmup=self.warmup))

    def end_episode(self, x_last: torch.Tensor | None = None) -> None:
        self._emit(end_op(self._episode_id, x_last))

    def discard_episode(self) -> int:
        """Ask the learner to rewind. Returns *chunks* sent, not transitions.

        The transition count is only knowable where the buffer lives, and the
        stream is ordered, so the learner applies this rewind against exactly
        the state the in-process buffer would have had.
        """
        self._emit(discard_op(self._episode_id))
        sent, self._sent_this_episode = self._sent_this_episode, 0
        return sent

    def save(self, path) -> None:
        """No-op: in process mode the learner owns the buffer and checkpoints it."""


class RemoteActorMirror:
    """Read-only actor copy fed by the learner's parameters stream."""

    def __init__(self, cfg: OnlineRLConfig, device: str | torch.device, params_queue: queue.Queue):
        self.actor = ChunkActor(cfg.ac).to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self._queue = params_queue
        self.version = 0
        self.stats: dict[str, float] = {}

    @torch.no_grad()
    def act(self, x: torch.Tensor, ref_chunk: torch.Tensor, deterministic: bool = False):
        return self.actor.sample(x, ref_chunk, deterministic=deterministic)

    def sync(self, _learner=None) -> bool:
        """Load the most recent published weights; drop any older backlog.

        Signature matches `ActorMirror.sync(learner)` so the training loop does
        not have to know which mode it is in.
        """
        payload = None
        while True:
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                break
        if payload is None:
            return False
        state, stats = bytes_to_params(payload)
        self.actor.load_state_dict({k: v.to(self.device) for k, v in state.items()})
        self.stats = stats
        self.version += 1
        return True


def learner_stub(host: str, port: int):
    # gRPC honours `http_proxy` by default, so with one exported the rollout
    # dials the proxy instead of the learner and every RPC comes back
    # UNAVAILABLE naming a port nobody configured. Learner and rollout are on
    # the same host (or an explicitly given one), so a proxy is never wanted.
    options = [*grpc_channel_options(), ("grpc.enable_http_proxy", 0)]
    channel = grpc.insecure_channel(f"{host}:{port}", options)
    return services_pb2_grpc.LearnerServiceStub(channel), channel


def wait_for_learner(stub, shutdown_event: threading.Event, attempts: int = 60) -> bool:
    """Block until the learner answers `Ready`, or we give up / are shut down."""
    for _ in range(attempts):
        if shutdown_event.is_set():
            return False
        try:
            stub.Ready(services_pb2.Empty())
            return True
        except grpc.RpcError as exc:
            logger.info("[rollout] waiting for the learner to come up... (%s)", exc.code())
            shutdown_event.wait(1.0)
    return False


def _op_stream(out_queue: queue.Queue, shutdown_event: threading.Event, batch_max: int = 32):
    """Drain the op queue into chunked gRPC messages, batching what is ready.

    Batching matters because ops arrive in bursts: one `add_chunk` emits up to
    `chunk_len / stride` transitions' worth of work on the learner side, and a
    round trip per op would put gRPC latency inside the control loop.
    """
    while not shutdown_event.is_set():
        try:
            ops = [out_queue.get(timeout=0.1)]
        except queue.Empty:
            continue
        while len(ops) < batch_max:
            try:
                ops.append(out_queue.get_nowait())
            except queue.Empty:
                break
        yield from send_bytes_in_chunks(
            ops_to_bytes(ops), services_pb2.Transition, log_prefix="[rollout] ops"
        )


def start_client_threads(
    host: str,
    port: int,
    op_queue: queue.Queue,
    params_queue: queue.Queue,
    shutdown_event: threading.Event,
) -> tuple[list[threading.Thread], Any]:
    """Start the send/receive threads against a learner that is already up."""
    stub, channel = learner_stub(host, port)
    if not wait_for_learner(stub, shutdown_event):
        channel.close()
        raise RuntimeError(
            f"Learner at {host}:{port} never became ready. Check its log — in "
            "process mode it starts before the rollout and exits on its own errors."
        )

    def send_ops():
        try:
            stub.SendTransitions(_op_stream(op_queue, shutdown_event))
        except grpc.RpcError:
            if not shutdown_event.is_set():
                logger.exception("[rollout] transition stream died")
                shutdown_event.set()

    def receive_params():
        try:
            receive_bytes_in_chunks(
                stub.StreamParameters(services_pb2.Empty()),
                params_queue,
                shutdown_event,
                log_prefix="[rollout] params",
            )
        except grpc.RpcError:
            if not shutdown_event.is_set():
                logger.exception("[rollout] parameter stream died")
                shutdown_event.set()

    threads = [
        threading.Thread(target=send_ops, name="rlt-send-ops", daemon=True),
        threading.Thread(target=receive_params, name="rlt-recv-params", daemon=True),
    ]
    for t in threads:
        t.start()
    return threads, channel


def stop_client(threads: list[threading.Thread], channel, shutdown_event: threading.Event) -> None:
    """Wind the streams down, *then* drop the channel.

    Closing the channel while an RPC is still in flight aborts it inside the
    gRPC C core, which surfaces as `terminate called without an active
    exception` at interpreter exit — a SIGABRT with no Python traceback.
    """
    shutdown_event.set()
    for t in threads:
        t.join(timeout=5.0)
    channel.close()
