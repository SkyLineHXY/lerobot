"""Replay buffer for chunk-level transitions, built at episode end.

Ported from openpi-RLT `rlt_online_rl/src/rlt_online_rl/replay.py` and the
window construction in `inference.py::_build_episode_replay`. Nothing is stored
while an episode runs: the rollout appends executed steps and per-anchor RL
states to a trace, and `end_episode` turns that trace into transitions.

The paper (Sec. V "Subsampling Action Chunks") stores, for an episode, the
transitions <x_{t}, a_{t:t+C}> for t = 0, stride, 2*stride, ... Windows are
built from the flat step trace, so a window freely spans the boundary between
two executed chunks, and they run all the way to the last step: a window that
overruns the episode is zero-padded (openpi-RLT `allow_partial=True`). Refusing
partial windows instead leaves exactly *one* reward-carrying transition per
episode — measured at 0.45% of a real LIBERO buffer, which is far too thin for
a terminal reward to bootstrap back through 24-60 chunk hops.

**Each anchor carries its own freshly sampled reference**, as openpi-RLT does by
re-querying its feature server per anchor. Slicing the chunk boundary's plan
forward instead is wrong whenever the execution horizon equals the chunk length:
the sliced row is the VLA's open-loop continuation while `action` holds what the
*next* re-plan did, and the two are far apart (measured 0.41 per element against
a reference std of 0.9, with the re-plan systematically 0.39-0.92x smaller). A
buffer built that way trains the actor to shrink the VLA's actions.

A fresh reference fixes the actor's *input*. Actions in a window that starts
mid-chunk can still straddle two planning decisions; `aligned` marks rows that
came from one decision. This matters only for optional behaviour imitation
(AWR). Eq. (5)'s reference BC and the off-policy critic can use every anchor.

Stored per transition:
  x             RL state (z_rl ++ proprio)                (D_x,)
  action        executed action chunk, zero-padded        (C, d)
  ref           VLA reference chunk at x                  (C, d)
  rewards       per-step reward, zero-padded              (C,)
  x_next        RL state C control steps later            (D_x,)
  ref_next      reference chunk at x_next                 (C, d)
  done          episode terminated within the window      scalar 0/1
  mc_return     discounted return-to-go from x            scalar
  mc_valid      whether `mc_return` is unbiased           scalar 0/1
  aligned       window lies inside one planning decision  scalar 0/1
  source_chunk  per-step control source                   (C,) uint8
  source        window-level control source               scalar uint8
  phase         warmup / online                           scalar uint8
  episode_id / success / intervention                     scalars

An operator-driven step preserves the VLA proposal in `ref`, because that is
what conditions the actor at deployment. The actor loss selects the executed
operator action as the BC target for those steps. `human_ref_override=True` is
the paper Sec. V prose variant; the current openpi-RLT code instead uses the
default here, which avoids a train/deploy input shift.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from lerobot.policies.rlt import TransitionSource

COLLECTION_PHASE_UNKNOWN = 0
COLLECTION_PHASE_WARMUP = 1
COLLECTION_PHASE_ONLINE = 2


@dataclass
class ChunkRecord:
    """Data gathered while executing one chunk."""

    xs: Tensor  # (n_anchor, D_x) RL states at `x_offsets` within this chunk
    x_offsets: Tensor  # (n_anchor,) long, offsets into the chunk
    refs: Tensor  # (n_anchor, >= C, d) VLA reference sampled *at* each anchor
    aligned: Tensor  # (n_anchor,) bool, see `_EpisodeTrace.anchors`
    actions: Tensor  # (n_exec, d) executed actions
    rewards: Tensor  # (n_exec,) reward after each executed step
    source: int = int(TransitionSource.RL)
    done: bool = False  # the episode terminated on the last executed step


@dataclass
class _EpisodeTrace:
    """Flat per-step log plus the RL states computed at window anchors."""

    phase: int
    actions: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    # position -> (x, ref, aligned). `aligned` marks the anchors whose window
    # lies inside a single planning decision, i.e. the ones whose `action` is a
    # target the deployed policy could actually have produced at `x`. It gates
    # optional behaviour imitation, not fresh-reference BC or critic updates.
    anchors: dict[int, tuple[Tensor, Tensor, bool]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.actions)


def ref_slice(ref_full: Tensor, offset: int, chunk_len: int) -> Tensor:
    """`ref_full[offset : offset + C]`, repeating the last row if it runs out.

    Real references are long enough (the VLA chunk has horizon H = 50), but
    hand-built records in tests are not, and a short slice would be a shape
    error inside `_store`.
    """
    out = ref_full[offset : offset + chunk_len].cpu()
    pad = chunk_len - out.shape[0]
    if pad <= 0:
        return out
    tail = out[-1:] if out.shape[0] else ref_full[-1:].cpu()
    return torch.cat([out, tail.expand(pad, -1)], dim=0)


def resolve_window_source(sources: list[int]) -> int:
    """Window-level label from per-step labels (openpi `_resolve_chunk_source`)."""
    values = set(sources)
    has_human = int(TransitionSource.HUMAN) in values
    has_policy = bool(
        values & {int(TransitionSource.BASE), int(TransitionSource.RL), int(TransitionSource.MIXED)}
    )
    if int(TransitionSource.MIXED) in values or (has_human and has_policy):
        return int(TransitionSource.MIXED)
    if has_human:
        return int(TransitionSource.HUMAN)
    return sources[0]


class ChunkReplayBuffer:
    def __init__(
        self,
        capacity: int,
        x_dim: int,
        chunk_len: int,
        action_dim: int,
        discount: float,
        stride: int = 2,
        device: str | torch.device = "cuda",
        seed: int = 0,
        sample_strategy: str = "uniform",
        human_ref_override: bool = False,
        recent_episode_window: int = 20,
        recent_online_ratio: float = 0.4,
        warmup_demo_ratio: float = 0.3,
        human_intervention_ratio: float = 0.2,
        aligned_ratio: float = 0.0,
    ):
        if sample_strategy not in ("uniform", "stratified"):
            raise ValueError(f"sample_strategy must be uniform or stratified, got {sample_strategy!r}")
        # Windows start at multiples of `stride`, so `start + C` is an anchor
        # only if C is a multiple of it too. Without this every second window
        # would silently be dropped for want of a next state.
        if chunk_len % stride:
            raise ValueError(f"chunk_len ({chunk_len}) must be a multiple of stride ({stride})")
        self.sample_strategy = sample_strategy
        self.human_ref_override = human_ref_override
        self.recent_episode_window = max(int(recent_episode_window), 1)
        self.recent_online_ratio = float(recent_online_ratio)
        self.warmup_demo_ratio = float(warmup_demo_ratio)
        self.human_intervention_ratio = float(human_intervention_ratio)
        self.aligned_ratio = float(aligned_ratio)
        self.capacity = capacity
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.discount = discount
        self.stride = stride
        self.device = torch.device(device)

        self.x = torch.zeros(capacity, x_dim)
        self.action = torch.zeros(capacity, chunk_len, action_dim)
        self.ref = torch.zeros(capacity, chunk_len, action_dim)
        self.rewards = torch.zeros(capacity, chunk_len)
        self.x_next = torch.zeros(capacity, x_dim)
        self.ref_next = torch.zeros(capacity, chunk_len, action_dim)
        self.done = torch.zeros(capacity)
        self.mc_return = torch.zeros(capacity)
        self.mc_valid = torch.zeros(capacity, dtype=torch.uint8)
        self.source_chunk = torch.zeros(capacity, chunk_len, dtype=torch.uint8)
        self.source = torch.zeros(capacity, dtype=torch.uint8)
        self.phase = torch.zeros(capacity, dtype=torch.uint8)
        self.success = torch.zeros(capacity, dtype=torch.uint8)
        self.aligned = torch.zeros(capacity, dtype=torch.uint8)
        self.intervention = torch.zeros(capacity)
        self.episode_id = torch.full((capacity,), -1, dtype=torch.long)

        self.size = 0
        self.total_added = 0  # monotone counter, used to pace the learner
        self._ptr = 0
        self._episode_id = -1  # monotone; `episode_id` column tags each stored row
        self._trace: _EpisodeTrace | None = None
        self._last_idx: Tensor | None = None
        self._gen = torch.Generator().manual_seed(seed)
        self._lock = threading.Lock()

    # ------------------------------------------------------------- assembly
    def start_episode(self, warmup: bool = False) -> None:
        with self._lock:
            self._episode_id += 1
            self._trace = _EpisodeTrace(
                phase=COLLECTION_PHASE_WARMUP if warmup else COLLECTION_PHASE_ONLINE
            )

    def discard_episode(self) -> int:
        """Drop the episode in progress. Returns the number of steps dropped.

        The operator's discard key means "that episode was garbage" (a bad
        reset, a mislabelled outcome, an object knocked over). Nothing has been
        written to the ring buffer yet, so this is just dropping the trace.
        """
        with self._lock:
            n = len(self._trace) if self._trace is not None else 0
            if self._trace is not None:
                self._trace = _EpisodeTrace(phase=self._trace.phase)
            return n

    def add_chunk(self, rec: ChunkRecord) -> None:
        """Append one executed chunk to the episode trace."""
        with self._lock:
            if self._trace is None:
                self._trace = _EpisodeTrace(phase=COLLECTION_PHASE_UNKNOWN)
            trace = self._trace
            base = len(trace)
            n_exec = rec.actions.shape[0]
            for i, offset in enumerate(rec.x_offsets.tolist()):
                if offset >= n_exec:
                    continue
                trace.anchors[base + int(offset)] = (
                    rec.xs[i].cpu(),
                    self._ref_slice(rec.refs[i], 0),
                    bool(rec.aligned[i]),
                )
            for j in range(n_exec):
                trace.actions.append(rec.actions[j].cpu())
                trace.rewards.append(float(rec.rewards[j]))
                trace.sources.append(int(rec.source))
                trace.dones.append(bool(rec.done) and j == n_exec - 1)

    def end_episode(
        self,
        x_last: Tensor | None = None,
        ref_last: Tensor | None = None,
        success: bool = False,
        truncation_is_failure: bool = False,
    ) -> int:
        """Build and store this episode's windows. Returns the count.

        `x_last` / `ref_last` are the RL state and reference observed after the
        final executed step. A terminated window masks its bootstrap so they do
        not matter there, but a truncated one does not, and pointing `x_next`
        back at `x` would make the Bellman backup a self-loop that silently pins
        those states to the wrong value. Every window that overruns the episode
        bootstraps off this one state, so without it none of them are built.
        """
        with self._lock:
            trace = self._trace
            self._trace = None
            if trace is None or not len(trace):
                return 0
            if x_last is not None and ref_last is not None:
                trace.anchors[len(trace)] = (x_last.cpu(), self._ref_slice(ref_last, 0), False)
            return self._emit_windows(trace, success, truncation_is_failure)

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        """G_t = sum_{k >= t} gamma^(k-t) r_k, over the rest of the episode."""
        out = [0.0] * len(rewards)
        acc = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            acc = rewards[t] + self.discount * acc
            out[t] = acc
        return out

    def _emit_windows(
        self, trace: _EpisodeTrace, success: bool, truncation_is_failure: bool = False
    ) -> int:
        c = self.chunk_len
        n = len(trace)
        # A simulator time limit declared to be a task failure is an MDP
        # terminal, not merely a trustworthy zero Monte-Carlo target.  Leaving
        # `done=0` here would still bootstrap Q from the post-timeout state and
        # contradict the MC label carried by the very same transition.
        if truncation_is_failure and trace.dones and not trace.dones[-1]:
            trace.dones[-1] = True
        # A window whose bootstrap state is missing cannot be stored at all, so
        # a full window needs its own anchor plus the one C steps later, while a
        # partial one falls back to the post-final anchor at `n`. A partial
        # window is then bootstrapped with gamma^C despite spanning fewer than C
        # steps (as in openpi-RLT). That is exact wherever it matters — a
        # terminated episode masks the bootstrap entirely, and terminated
        # episodes are the only ones that ever carry a reward — and only
        # over-discounts a truncated tail.
        starts = [
            start
            for start in range(0, n, self.stride)
            if start in trace.anchors
            and ((start + c) in trace.anchors if start + c <= n else n in trace.anchors)
        ]
        returns = self._discounted_returns(trace.rewards)
        # Monte-Carlo returns are only unbiased when the episode really ended:
        # a truncated one stops accumulating at the time limit, which reads as
        # "no reward was ever coming" rather than "we stopped looking". In a
        # simulator with a task-completion checker that distinction is empty —
        # running out the clock *is* the failure, and its return really is 0.
        # Dropping those rows leaves the MC target with no negatives at all
        # (measured: every mc_valid row on the 2026-08-20 buffer came from a
        # success or an operator failure key), which is worse than the bias.
        mc_valid = bool(trace.dones and trace.dones[-1]) or truncation_is_failure

        for start in starts:
            x, ref, aligned = trace.anchors[start]
            x_next, ref_next, _ = trace.anchors[min(start + c, n)]
            sources = trace.sources[start : start + c]
            action = self._pad_chunk(torch.stack(trace.actions[start : start + c]))
            self._store(
                x=x,
                action=action,
                # Normally preserve the VLA proposal as the actor condition.
                # `_human_ref` only changes it for the explicit legacy ablation.
                ref=self._human_ref(ref, action, sources),
                rewards=self._pad_rewards(trace.rewards[start : start + c]),
                x_next=x_next,
                ref_next=self._human_ref(
                    ref_next,
                    self._pad_chunk(torch.stack(trace.actions[start + c : start + 2 * c]))
                    if len(trace.actions[start + c : start + 2 * c])
                    else None,
                    trace.sources[start + c : start + 2 * c],
                ),
                done=float(any(trace.dones[start : start + c])),
                mc_return=returns[start],
                mc_valid=mc_valid,
                sources=sources,
                phase=trace.phase,
                success=success,
                aligned=aligned,
            )
        return len(starts)

    def _human_ref(self, ref: Tensor, action: Tensor | None, sources: list[int]) -> Tensor:
        """Paper-prose ablation: replace operator-step references by commands."""
        if action is None or not self.human_ref_override:
            return ref
        human = torch.tensor(
            [s in (int(TransitionSource.HUMAN), int(TransitionSource.MIXED)) for s in sources]
            + [False] * (self.chunk_len - len(sources)),
            dtype=torch.bool,
        )
        if not human.any():
            return ref
        return torch.where(human.unsqueeze(-1), action[: self.chunk_len], ref[: self.chunk_len])

    def _pad_chunk(self, actions: Tensor) -> Tensor:
        pad = self.chunk_len - actions.shape[0]
        if pad <= 0:
            return actions
        return torch.cat([actions, actions.new_zeros(pad, actions.shape[1])], dim=0)

    def _pad_rewards(self, rewards: list[float]) -> Tensor:
        return torch.tensor(rewards + [0.0] * (self.chunk_len - len(rewards)))

    def _ref_slice(self, ref_full: Tensor, offset: int) -> Tensor:
        return ref_slice(ref_full, offset, self.chunk_len)

    def aligned_pool(self) -> Tensor:
        return (self.aligned[: self.size] > 0).nonzero().flatten()

    def _store(
        self, x, action, ref, rewards, x_next, ref_next, done, mc_return, mc_valid,
        sources, phase, success, aligned=True
    ) -> None:
        p = self._ptr
        self.episode_id[p] = self._episode_id
        self.x[p] = x
        self.action[p] = action
        self.ref[p] = ref[: self.chunk_len]
        self.rewards[p] = rewards
        self.x_next[p] = x_next
        self.ref_next[p] = ref_next[: self.chunk_len]
        self.done[p] = done
        self.mc_return[p] = mc_return
        self.mc_valid[p] = int(mc_valid)
        # The window label is resolved before padding, so the zero-padded tail
        # of a partial window cannot fabricate a BASE-driven step.
        window_source = resolve_window_source(sources)
        padded = sources + [int(TransitionSource.BASE)] * (self.chunk_len - len(sources))
        self.source_chunk[p] = torch.tensor(padded, dtype=torch.uint8)
        self.source[p] = window_source
        self.intervention[p] = float(
            window_source in (int(TransitionSource.HUMAN), int(TransitionSource.MIXED))
        )
        self.phase[p] = phase
        self.success[p] = int(success)
        self.aligned[p] = int(aligned)
        self._ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    # ------------------------------------------------------------- sampling
    def sample(self, batch_size: int) -> dict[str, Tensor] | None:
        """Draw a batch, or None if the buffer is empty right now.

        The caller's `len(buffer) >= batch_size` check cannot be trusted: the
        rollout thread writes a whole episode at once in between, so emptiness
        is re-tested here, under the same lock that does the gather.
        """
        # The host-side gather happens under the lock; only the (asynchronous)
        # host-to-device copies are left outside it, so a concurrent write can
        # never land in a row this batch is still reading.
        with self._lock:
            if self.size == 0:
                return None
            idx = (
                self._stratified_indices(batch_size)
                if self.sample_strategy == "stratified"
                else torch.randint(0, self.size, (batch_size,), generator=self._gen)
            )
            batch = {
                "x": self.x[idx],
                "action": self.action[idx],
                "ref": self.ref[idx],
                "rewards": self.rewards[idx],
                "x_next": self.x_next[idx],
                "ref_next": self.ref_next[idx],
                "done": self.done[idx],
                "mc_return": self.mc_return[idx],
                "mc_valid": self.mc_valid[idx].float(),
                "aligned": self.aligned[idx].float(),
                "source_chunk": self.source_chunk[idx],
            }
            self._last_idx = idx
        if self.device.type != "cuda":
            return batch
        return {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in batch.items()}

    def composition(self) -> dict[str, float]:
        """What the buffer holds, and what the last batch drew from it.

        The two differ under `stratified`, and that difference is the only way to
        see whether the strata are actually reaching the learner: a `human` pool
        that is 5% of the buffer but 20% of every batch is the sampler working.
        """
        with self._lock:
            n = self.size
            if n == 0:
                return {"buffer_size": 0.0}
            latest_episode = int(self.episode_id[:n].max())
            recent_rows = self.episode_id[:n] >= latest_episode - self.recent_episode_window + 1
            recent_sources = self.source_chunk[:n][recent_rows]
            out = {
                "buffer_size": float(n),
                "buffer_total_added": float(self.total_added),
                "buffer_episodes": float(latest_episode + 1),
                "buffer_human_ratio": (self.intervention[:n] > 0).float().mean().item(),
                "buffer_human_step_ratio": (
                    (self.source_chunk[:n] == int(TransitionSource.HUMAN))
                    | (self.source_chunk[:n] == int(TransitionSource.MIXED))
                ).float().mean().item(),
                # Unlike the cumulative buffer composition, this responds on
                # the same horizon as the paper's falling intervention curve.
                "buffer_recent_human_step_ratio": (
                    (recent_sources == int(TransitionSource.HUMAN))
                    | (recent_sources == int(TransitionSource.MIXED))
                ).float().mean().item(),
                "buffer_warmup_ratio": (self.phase[:n] == COLLECTION_PHASE_WARMUP).float().mean().item(),
                "buffer_rewarded_ratio": (self.rewards[:n].sum(-1) > 0).float().mean().item(),
                # What the critic *should* be reporting on this buffer. A
                # `train/q_mean` an order of magnitude under this is a critic
                # that has not propagated the reward, not a converged one.
                "buffer_mc_mean": self.mc_return[:n].mean().item(),
                "buffer_mc_valid_ratio": (self.mc_valid[:n] > 0).float().mean().item(),
                "buffer_success_ratio": (self.success[:n] > 0).float().mean().item(),
                # Should sit at stride / chunk_len. Anything else means the
                # rollout stopped tagging the chunk-boundary anchors.
                "buffer_aligned_ratio": (self.aligned[:n] > 0).float().mean().item(),
                "buffer_done_ratio": (self.done[:n] > 0).float().mean().item(),
            }
            idx = self._last_idx
            if idx is not None and idx.numel():
                out |= {
                    "sample_human_ratio": (self.intervention[idx] > 0).float().mean().item(),
                    "sample_human_step_ratio": (
                        (self.source_chunk[idx] == int(TransitionSource.HUMAN))
                        | (self.source_chunk[idx] == int(TransitionSource.MIXED))
                    ).float().mean().item(),
                    "sample_warmup_ratio": (
                        self.phase[idx] == COLLECTION_PHASE_WARMUP
                    ).float().mean().item(),
                    "sample_rewarded_ratio": (self.rewards[idx].sum(-1) > 0).float().mean().item(),
                    "sample_aligned_ratio": (self.aligned[idx] > 0).float().mean().item(),
                    "sample_episode_age": float(
                        latest_episode - self.episode_id[idx].float().mean().item()
                    ),
                }
            return out

    def _draw(self, pool: Tensor, count: int) -> Tensor:
        """`count` rows from `pool`, with replacement. Empty pool draws nothing."""
        if count <= 0 or pool.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        return pool[torch.randint(0, pool.numel(), (count,), generator=self._gen)]

    def _stratified_indices(self, batch_size: int) -> Tensor:
        """Guarantee the rare-but-informative transitions a share of every batch.

        Uniform sampling weighs a transition by how often it was collected, which
        is exactly backwards for the three groups that carry the most signal:

        * **recent online** — the critic is asked for Q at the actor's *current*
          actions, so it has to keep fitting the current policy's distribution.
        * **warmup / demo** — the frozen VLA's own behaviour, which is what the
          BC term anchors to; it stops being collected once warmup ends.
        * **human** — a takeover is the only place the buffer holds an action the
          policy would not have produced at that state. That contrast is what
          teaches the critic to depend on the action at all.
        * **aligned** — the only rows the actor can imitate at all; they are just
          `stride / chunk_len` of the buffer.

        Pools deliberately overlap (a recent takeover counts in two), and each is
        drawn with replacement, as in openpi-RLT `_sample_stratified_indices`.
        Any shortfall — early on, every pool is small or empty — is topped up
        uniformly, so this degrades to uniform sampling rather than to a batch of
        five rows repeated fifty times.
        """
        n = self.size
        rows = torch.arange(n)
        episode = self.episode_id[:n]
        phase = self.phase[:n]
        newest = int(episode.max()) if n else -1

        recent_online = rows[
            (phase == COLLECTION_PHASE_ONLINE)
            & (episode >= newest - self.recent_episode_window + 1)
        ]
        warmup_demo = rows[phase == COLLECTION_PHASE_WARMUP]
        human = rows[
            (self.intervention[:n] > 0)
            | (
                (self.source_chunk[:n] == int(TransitionSource.HUMAN))
                | (self.source_chunk[:n] == int(TransitionSource.MIXED))
            ).any(dim=1)
        ]

        # Off by default: aligned rows already reach the batch at their natural
        # `stride / chunk_len` rate through the other three pools, and the four
        # ratios have to leave room for the uniform top-up — the existing three
        # already claim 0.9 of it. Turn this up only if `train/aligned_ratio`
        # shows the BC term running on too few rows.
        aligned = rows[self.aligned[:n] > 0]

        parts = [
            self._draw(recent_online, round(batch_size * self.recent_online_ratio)),
            self._draw(warmup_demo, round(batch_size * self.warmup_demo_ratio)),
            self._draw(human, round(batch_size * self.human_intervention_ratio)),
            self._draw(aligned, round(batch_size * self.aligned_ratio)),
        ]
        drawn = sum(p.numel() for p in parts)
        parts.append(self._draw(rows, batch_size - drawn))
        idx = torch.cat(parts)[:batch_size]
        if idx.numel() < batch_size:  # every ratio rounded to zero
            idx = torch.cat([idx, self._draw(rows, batch_size - idx.numel())])
        return idx[torch.randperm(idx.numel(), generator=self._gen)]

    def __len__(self) -> int:
        return self.size

    _COLUMNS = (
        "x", "action", "ref", "rewards", "x_next", "ref_next", "done",
        "mc_return", "mc_valid", "aligned",
        "source_chunk", "source", "phase", "success", "intervention", "episode_id",
    )

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        """Persist the buffer so a real-robot run can resume after a stop."""
        with self._lock:
            self._save_locked(path)

    def _save_locked(self, path: str | Path) -> None:
        n = self.size
        # A tensor slice keeps the storage of the full preallocated buffer.
        # torch.save preserves that storage, so saving (say) 8k rows from a
        # million-row buffer used to produce an almost 1 GB checkpoint.  Clone
        # each live prefix so the serialized storage is exactly the live data.
        blob = {key: getattr(self, key)[:n].clone() for key in self._COLUMNS}
        blob.update({"ptr": self._ptr, "size": n, "total_added": self.total_added})
        torch.save(blob, path)

    def load(self, path: str | Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        missing = [key for key in self._COLUMNS if key not in sd]
        if missing:
            # Buffers written before the openpi-aligned schema store a scalar
            # discounted return and a bc_weight column; reading them here would
            # mean guessing what the missing per-step rewards were.
            raise ValueError(f"replay buffer at {path} predates the current schema (missing {missing})")
        n = min(int(sd["size"]), self.capacity)
        for key in self._COLUMNS:
            getattr(self, key)[:n] = sd[key][:n]
        self.size = n
        self._ptr = n % self.capacity
        self.total_added = int(sd.get("total_added", n))
        self._episode_id = int(self.episode_id[:n].max()) if n else -1
