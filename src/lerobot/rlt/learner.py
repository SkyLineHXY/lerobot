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
"""
from __future__ import annotations

import threading
import time
from collections import deque

import torch

from lerobot.policies.rlt import ChunkActor, OnlineRLConfig, RLTAgent

from .replay_buffer import ChunkReplayBuffer


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
        publish_every: int = 1,
        idle_sleep_s: float = 0.002,
    ):
        super().__init__(daemon=True, name="rlt-learner")
        self.agent = agent
        self.buffer = buffer
        self.cfg = cfg
        self.publish_every = publish_every
        self.idle_sleep_s = idle_sleep_s

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._published: dict[str, torch.Tensor] = {}
        self._version = 0
        # Warmup collects with the base VLA. The critic should still learn from
        # that data (it is exactly the "initial learning signal" the paper wants
        # from warmup); only the actor waits until there is a critic worth
        # maximising.
        self.allow_actor_updates = False

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

    # --------------------------------------------------------------- stats
    def metrics(self) -> dict[str, float]:
        """Window-averaged metrics; a single update's loss is mostly noise."""
        if not self._recent:
            return {}
        keys = {k for m in self._recent for k in m}
        out = {}
        for k in keys:
            vals = [m[k] for m in self._recent if k in m]
            out[k] = sum(vals) / len(vals)
        out["updates"] = float(self.updates)
        return out

    # ---------------------------------------------------------------- loop
    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cfg = self.cfg
        while not self._stop_event.is_set():
            # UTD is defined per *transition*, so pace against what the rollout
            # thread actually stored rather than against a nominal chunk length
            # (chunks end early on success or truncation).
            pending = self.buffer.total_added - self._consumed
            if len(self.buffer) < cfg.batch_size or pending <= 0:
                time.sleep(self.idle_sleep_s)
                continue

            n_updates = int(cfg.utd * pending)
            self._consumed = self.buffer.total_added
            for _ in range(n_updates):
                if self._stop_event.is_set():
                    break
                batch = self.buffer.sample(cfg.batch_size)
                metrics = self.agent.update(batch, allow_actor=self.allow_actor_updates)
                self._recent.append(metrics)
                self.updates += 1
                self._publish()
