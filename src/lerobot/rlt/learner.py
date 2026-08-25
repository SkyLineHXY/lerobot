"""Background learner thread for RLT online RL (paper Sec. V, "Update").

The paper performs rollouts and learning *asynchronously*: on a real robot a
synchronous loop would freeze the arm at every chunk boundary for the duration
of `utd * transitions` gradient steps. Here the learner owns the agent and runs
in its own thread, pacing itself by the update-to-data ratio against the number
of transitions the rollout thread has actually produced.

The rollout thread never touches the training weights directly. It queries an
:class:`ActorMirror` — a private copy of the actor refreshed at chunk
boundaries under a lock — so a chunk is always planned with one coherent set of
parameters rather than with weights being mutated mid-forward.

What this concurrency does *not* buy: the two threads share one CUDA context and
one default stream, and `RLTAgent.update` calls `.item()` several times per
gradient step, each of which drains that stream. So a long gradient burst still
sits in front of the rollout's next VLA forward. `max_updates_per_burst` bounds
how long that can be — at `utd=5` and stride-2 subsampling an unbounded burst is
~25 updates per chunk boundary, which on a 30 Hz arm is exactly the stall this
design exists to avoid.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import torch

from lerobot.policies.rlt import ChunkActor, OnlineRLConfig, RLTAgent

from .replay_buffer import ChunkReplayBuffer


def burst_size(
    pending: int, utd: int, max_updates_per_burst: int, warmup_deficit: int = 0
) -> tuple[int, int]:
    """How many gradient steps to run now, and how many transitions that spends.

    Two regimes share this. Normally the learner is paced by UTD against what
    the rollout has actually stored. During warmup it may additionally owe
    `warmup_deficit` steps (openpi-RLT `warmup_post_collect_updates`), and those
    are *not* charged against a transition: the point of that phase is to fit
    the critic on the data already collected, so once the UTD budget runs dry
    the learner keeps grinding on the same rows. Leaving `_consumed` untouched
    there is what preserves the `updates == utd * total_added` invariant for the
    online phase that follows.
    """
    if pending > 0:
        # Cap in *transitions*, not updates: crediting a partial transition
        # would break that invariant, and the leftover is picked up next round.
        if max_updates_per_burst > 0 and utd > 0:
            pending = min(pending, max(1, max_updates_per_burst // utd))
        return int(utd * pending), pending
    if warmup_deficit > 0:
        capped = min(warmup_deficit, max_updates_per_burst) if max_updates_per_burst > 0 else warmup_deficit
        return int(capped), 0
    return 0, 0


class ActorMirror:
    """Read-only actor copy used by the rollout thread."""

    def __init__(self, cfg: OnlineRLConfig, device: str | torch.device):
        self.actor = ChunkActor(cfg.ac).to(device).eval().requires_grad_(False)
        self.version = -1

    @torch.no_grad()
    def act(self, x: torch.Tensor, ref_chunk: torch.Tensor, deterministic: bool = False):
        """Inference: the reference is always provided (paper App. B)."""
        return self.actor.sample(x, ref_chunk, deterministic=deterministic)

    def sync(self, learner: LearnerThread) -> bool:
        """Pull the latest published weights. Returns True if they changed."""
        state, version = learner.published_actor()
        if version == self.version:
            return False
        self.actor.load_state_dict(state)
        self.version = version
        return True


class LearnerThread(threading.Thread):
    """Runs actor-critic updates off the shared replay buffer."""

    def __init__(
        self,
        agent: RLTAgent,
        buffer: ChunkReplayBuffer,
        cfg: OnlineRLConfig,
        publish_every: int = 10,
        idle_sleep_s: float = 0.002,
        max_updates_per_burst: int = 32,
    ):
        super().__init__(daemon=True, name="rlt-learner")
        self.agent = agent
        self.buffer = buffer
        self.cfg = cfg
        # Publishing clones the whole actor state_dict. The mirror only syncs at
        # chunk boundaries, so doing it every gradient step was pure overhead.
        self.publish_every = publish_every
        self.idle_sleep_s = idle_sleep_s
        self.max_updates_per_burst = max_updates_per_burst

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._paused = threading.Event()
        self._lock = threading.Lock()
        self._published: dict[str, torch.Tensor] = {}
        self._version = 0
        self._error: BaseException | None = None
        # Warmup collects with the base VLA. Both networks keep training on it —
        # the actor is not frozen, it just runs on the warmup BC/Q weight pair,
        # which leans on imitation while the critic is still worthless
        # (openpi-RLT `trainer.py::_resolve_actor_loss_weights`).
        self.warmup = True

        self._consumed = 0  # transitions already accounted for by the UTD pacer
        self.updates = 0
        self._recent: deque[dict[str, float]] = deque(maxlen=200)
        self._publish(force=True)

    # ------------------------------------------------------------- publish
    def _publish(self, force: bool = False) -> None:
        if not force and self.updates % self.publish_every:
            return
        state = {k: v.detach().clone() for k, v in self.agent.actor.state_dict().items()}
        with self._lock:
            self._published = state
            self._version += 1

    def published_actor(self) -> tuple[dict[str, torch.Tensor], int]:
        with self._lock:
            return self._published, self._version

    def warmup_deficit(self) -> int:
        if not self.warmup:
            return 0
        return max(self.cfg.warmup_post_collect_updates - self.updates, 0)

    # --------------------------------------------------------------- stats
    def metrics(self) -> dict[str, float]:
        """Window-averaged metrics; a single update's loss is mostly noise."""
        # Snapshot first: the learner appends to `_recent` while the main thread
        # reads it, and iterating a deque being mutated raises.
        recent = list(self._recent)
        if not recent:
            return {}
        keys = {k for m in recent for k in m}
        out = {}
        for k in keys:
            vals = [m[k] for m in recent if k in m]
            out[k] = sum(vals) / len(vals)
        out["updates"] = float(self.updates)
        return out

    # ---------------------------------------------------------------- loop
    def stop(self) -> None:
        self._stop_event.set()

    def raise_if_failed(self) -> None:
        """Re-raise a learner-thread exception on the caller's thread.

        Without this the thread dies silently and the run continues forever with
        a frozen actor and no gradient steps, while the log keeps printing the
        last window's averages as if nothing happened.
        """
        if self._error is not None:
            raise RuntimeError("RLT learner thread died") from self._error

    def pause(self, timeout: float = 5.0) -> bool:
        """Block until the learner is between gradient steps.

        Checkpointing calls `agent.state_dict()` from the main thread; without
        this it can run halfway through an optimizer step and write a state
        whose actor and critic disagree.
        """
        if not self.is_alive():
            return True
        self._pause_event.set()
        return self._paused.wait(timeout)

    def resume(self) -> None:
        self._pause_event.clear()
        self._paused.clear()

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            self._error = exc
            self._paused.set()  # never leave a `pause()` caller blocked

    def _run(self) -> None:
        cfg = self.cfg
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                self._paused.set()
                time.sleep(self.idle_sleep_s)
                continue

            # UTD is defined per *transition*, so pace against what the rollout
            # thread actually stored rather than against a nominal chunk length
            # (chunks end early on success or truncation).
            n_updates, consumed = burst_size(
                self.buffer.total_added - self._consumed,
                cfg.utd,
                self.max_updates_per_burst,
                self.warmup_deficit(),
            )
            if len(self.buffer) < cfg.batch_size or n_updates <= 0:
                time.sleep(self.idle_sleep_s)
                continue

            self._consumed += consumed
            for _ in range(n_updates):
                if self._stop_event.is_set() or self._pause_event.is_set():
                    break
                batch = self.buffer.sample(cfg.batch_size)
                if batch is None:  # operator discarded the episode mid-burst
                    break
                metrics = self.agent.update(batch, warmup=self.warmup)
                if cfg.diagnostics_every > 0 and self.updates % cfg.diagnostics_every == 0:
                    metrics |= self.agent.diagnostics(batch)
                self._recent.append(metrics)
                self.updates += 1
                self._publish()
