"""Replay buffer for chunk-level transitions with stride subsampling.

The paper (Sec. V "Subsampling Action Chunks") stores, for a chunk executed
from step t, the transitions <x_{t+o}, a_{t+o:t+o+C}> for offsets
o = 0, 2, 4, ... Because the VLA reference chunk has horizon H (50) while the
policy only executes C (10) steps, shifted references for intermediate
offsets come directly from the same reference chunk. Action/reward windows
for o > 0 span two consecutive executed chunks, so transitions for chunk k
are emitted once chunk k+1 has been executed (or the episode ended).

Stored per transition:
  x            RL state (z_rl ++ proprio)               (D_x,)
  action       executed action chunk                    (C, d)
  ref          reference chunk at x (VLA or human)      (C, d)
  reward_disc  sum_{j=1}^{k} gamma^{j-1} r_j            scalar
  x_next       RL state k control steps later           (D_x,)
  ref_next     reference chunk at x_next                (C, d)
  done         episode terminated within the window     scalar 0/1
  actual_steps k = steps actually executed in the window (<= C)
  bc_weight    multiplier on the actor's BC term         scalar 0/1

A window that could not be filled to C steps keeps its real length in
`actual_steps` so the critic bootstraps with gamma^k; the unexecuted tail is
padded by repeating the last action only so the tensor shape stays fixed. A
transition whose true `x_next` is unknown (e.g. an episode aborted mid-chunk)
is dropped rather than stored with a fabricated next state.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass
class ChunkRecord:
    """Data gathered while executing one C-step chunk."""

    xs: Tensor  # (n_off, D_x) RL states at offsets 0, stride, 2*stride, ...
    actions: Tensor  # (C, d) executed actions
    rewards: Tensor  # (C,) reward after each executed step
    ref_full: Tensor  # (>=C+max_off, d) reference (VLA chunk or tiled human action)
    done: bool = False
    done_step: int | None = None  # steps actually executed if done early
    from_human: bool = False  # operator drove this chunk, so `ref_full` is their command

    @property
    def n_executed(self) -> int:
        """Steps actually executed in this chunk."""
        return self.done_step if self.done_step is not None else self.actions.shape[0]

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
    ):
        self.capacity = capacity
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.discount = discount
        self.stride = stride
        self.device = torch.device(device)

        self.x = torch.zeros(capacity, x_dim)
        self.action = torch.zeros(capacity, chunk_len, action_dim)
        self.ref = torch.zeros(capacity, chunk_len, action_dim)
        self.reward_disc = torch.zeros(capacity)
        self.x_next = torch.zeros(capacity, x_dim)
        self.ref_next = torch.zeros(capacity, chunk_len, action_dim)
        self.done = torch.zeros(capacity)
        self.actual_steps = torch.zeros(capacity)
        # Multiplier on the actor's BC term. A takeover replaces `ref` with the
        # operator's own command, which is only a better imitation target than
        # the VLA if the correction actually worked; see `_finalize_episode`.
        self.bc_weight = torch.ones(capacity)

        self.size = 0
        self.total_added = 0  # monotone counter, used to pace the learner
        self._ptr = 0
        self._episode_ptr = 0  # write pointer when the current episode started
        self._episode_added = 0  # transitions stored since then
        self._episode_reward = 0.0  # reward seen this episode, for `_finalize_episode`
        self._episode_human_rows: list[int] = []  # rows whose `ref` is an operator command
        self._pending: ChunkRecord | None = None
        self._gammas = discount ** torch.arange(chunk_len).float()
        self._gen = torch.Generator().manual_seed(seed)
        # The rollout thread writes and the learner thread samples. Appending
        # alone would be safe enough (rows are filled before `size` moves), but
        # `discard_episode` *rewinds* the write pointer and shrinks `size`, so a
        # concurrent `sample` can draw rows that are about to be overwritten.
        # Write bursts are microseconds; sampling is the only hot path.
        self._lock = threading.Lock()

    # ------------------------------------------------------------- assembly
    # Everything below the public methods runs with `self._lock` already held;
    # `_store` and the `_emit_*` helpers must never take it themselves.
    def start_episode(self) -> None:
        with self._lock:
            if self._pending is not None:
                # Previous episode ended without an explicit terminal/truncation
                # signal; we have no trustworthy next state, so drop it.
                self._pending = None
            # Idempotent: the terminal and truncation paths already finalize, and
            # this clears the trackers. It only bites when an episode ended
            # through neither, which would otherwise leave a failed takeover
            # holding a full BC weight.
            self._finalize_episode()
            self._episode_ptr = self._ptr
            self._episode_added = 0
            self._episode_reward = 0.0
            self._episode_human_rows = []

    def _finalize_episode(self) -> None:
        """Drop the BC target of takeovers that led nowhere.

        A takeover overwrites `ref` with the operator's command so the actor
        imitates the correction instead of the VLA attempt it replaced (paper
        Sec. V). That reasoning only holds when the correction worked. An
        episode the operator ended with the failure key, or that ran out of
        steps, contains a human command that demonstrably did *not* finish the
        task, and BC-ing toward it teaches the actor to reproduce the failure.

        The executed action is left alone — it is what really happened and the
        critic needs it, with the reward it actually earned (zero). Only the
        imitation target is withdrawn.

        Credit is episode-level, which is coarse: a takeover early in an episode
        the policy later rescues keeps its weight. Per-intervention credit would
        need the reward to be attributable to one takeover, which sparse
        terminal rewards do not give us.
        """
        if self._episode_reward <= 0:
            for row in self._episode_human_rows:
                self.bc_weight[row] = 0.0
        self._episode_reward = 0.0
        self._episode_human_rows = []

    def discard_episode(self) -> int:
        """Drop everything the current episode has stored. Returns the count.

        The operator's discard key means "that episode was garbage" (a bad
        reset, a mislabelled outcome, an object knocked over), so dropping only
        the un-paired pending chunk would leave exactly the transitions they
        wanted gone. Rewinding the write pointer is only sound while the buffer
        has never wrapped: once it is full, the slots ahead of the rewound
        pointer hold valid older data, so shrinking `size` would delete the
        wrong transitions. In that case we keep them and say so — a real run
        (200k capacity, ~10^2 transitions per episode) never gets there.
        """
        with self._lock:
            self._pending = None
            self._episode_reward = 0.0
            self._episode_human_rows = []
            n = self._episode_added
            if n == 0:
                return 0
            if self.size >= self.capacity or n > self.size:
                self._episode_ptr = self._ptr
                self._episode_added = 0
                return 0
            self._ptr = self._episode_ptr
            self.size -= n
            self.total_added -= n  # keep the learner's UTD pacing honest
            self._episode_added = 0
            return n

    def add_chunk(self, rec: ChunkRecord) -> None:
        """Feed one executed chunk; emits transitions for the previous one."""
        with self._lock:
            self._episode_reward += float(rec.rewards.sum())
            if self._pending is not None:
                self._emit_pair(self._pending, rec)
            if rec.done:
                self._flush_terminal(rec)
                self._pending = None
                self._finalize_episode()
            else:
                self._pending = rec

    def end_episode(self, x_last: Tensor | None = None) -> None:
        """Call when an episode ends without `done` (e.g. time-limit truncation).

        `x_last` is the RL state observed after the final executed step. It is
        required to bootstrap a truncated window; without it the pending chunk
        is dropped instead of being stored against a fabricated next state.
        """
        with self._lock:
            if self._pending is not None:
                self._flush_terminal(self._pending, truncated=True, x_last=x_last)
            self._pending = None
            self._finalize_episode()

    def _emit_pair(self, k: ChunkRecord, k1: ChunkRecord) -> None:
        c = self.chunk_len
        n0, n1 = k.n_executed, k1.n_executed
        for i, o in enumerate(range(0, c, self.stride)):
            if i >= k.xs.shape[0] or o >= n0:
                break  # chunk k ended before this offset was observed
            # Only steps actually executed may enter the window (either chunk
            # can end early on success or truncation).
            take = min(o, n1)
            action = torch.cat([k.actions[o:n0], k1.actions[:take]], dim=0)
            rewards = torch.cat([k.rewards[o:n0], k1.rewards[:take]], dim=0)
            steps = action.shape[0]
            action, rewards = self._pad_window(action, rewards)
            x_next_idx = i

            # The window ends `steps` control steps after x, i.e. at offset o
            # of chunk k+1. If k+1 stopped before reaching that offset we do
            # not have the matching state — skip rather than misalign by up to
            # C steps.
            terminated_in_window = bool(k1.done and o > 0 and n1 <= o)
            if not terminated_in_window and x_next_idx >= k1.xs.shape[0]:
                continue

            x_next = k1.xs[min(x_next_idx, k1.xs.shape[0] - 1)]
            self._store(
                x=k.xs[i],
                action=action,
                ref=k.ref_full[o : o + c],
                rewards=rewards,
                x_next=x_next,
                ref_next=k1.ref_full[o : o + c],
                done=float(terminated_in_window),
                actual_steps=steps,
                from_human=k.from_human,  # `ref` is sliced out of chunk k
            )

    def _flush_terminal(
        self,
        rec: ChunkRecord,
        truncated: bool = False,
        x_last: Tensor | None = None,
    ) -> None:
        """Emit the windows of the final chunk of an episode.

        On termination the bootstrap is masked, so `x_next` is irrelevant. On
        truncation it is *not* masked, so a real post-episode state is needed;
        pointing `x_next` back at `x` would make the backup a self-loop and
        silently pin those states to zero value.

        The same asymmetry decides `ref_next`. Every window here ends at the
        same place — `x_last`, `d_step` control steps into the chunk — so its
        reference is this chunk's own reference shifted to `d_step`. Zeroing it
        instead would make the truncated backup evaluate `mu(x_last, 0)`, and
        with the residual parameterisation `mu = 0 + residual` that is a pure
        noise action inside the +-max_residual box, on an input the deployed
        policy never sees.
        """
        if truncated and x_last is None:
            return
        c = self.chunk_len
        d_step = rec.n_executed
        ref_next = self._ref_slice(rec.ref_full, d_step)
        for i, o in enumerate(range(0, c, self.stride)):
            if o >= d_step or i >= rec.xs.shape[0]:
                break
            action = rec.actions[o:d_step]
            rewards = rec.rewards[o:d_step]
            steps = action.shape[0]
            action, rewards = self._pad_window(action, rewards)
            self._store(
                x=rec.xs[i],
                action=action,
                ref=rec.ref_full[o : o + c],
                rewards=rewards,
                x_next=rec.xs[i] if x_last is None else x_last,
                ref_next=ref_next,
                done=0.0 if truncated else 1.0,
                actual_steps=steps,
                from_human=rec.from_human,
            )

    def _ref_slice(self, ref_full: Tensor, offset: int) -> Tensor:
        """`ref_full[offset : offset + C]`, repeating the last row if it runs out.

        Real references are long enough (the VLA chunk has horizon H = 50 and a
        human-authored one is tiled to 2C), but hand-built records in tests are
        not, and a short slice would be a shape error inside `_store`.
        """
        c = self.chunk_len
        out = ref_full[offset : offset + c]
        pad = c - out.shape[0]
        if pad <= 0:
            return out
        tail = out[-1:] if out.shape[0] else ref_full[-1:]
        return torch.cat([out, tail.expand(pad, -1)], dim=0)

    def _pad_window(self, action: Tensor, rewards: Tensor) -> tuple[Tensor, Tensor]:
        """Pad a short window to C steps; `actual_steps` records the real length."""
        pad = self.chunk_len - action.shape[0]
        if pad <= 0:
            return action, rewards
        action = torch.cat([action, action[-1:].expand(pad, -1)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(pad)], dim=0)
        return action, rewards

    def _store(
        self, x, action, ref, rewards, x_next, ref_next, done, actual_steps, from_human=False
    ) -> None:
        p = self._ptr
        self.bc_weight[p] = 1.0
        if from_human:
            self._episode_human_rows.append(p)
        self.x[p] = x.cpu()
        self.action[p] = action.cpu()
        self.ref[p] = ref[: self.chunk_len].cpu()
        self.reward_disc[p] = (rewards[: self.chunk_len] * self._gammas).sum()
        self.x_next[p] = x_next.cpu()
        self.ref_next[p] = ref_next[: self.chunk_len].cpu()
        self.done[p] = done
        self.actual_steps[p] = float(actual_steps)
        self._ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1
        self._episode_added += 1

    # ------------------------------------------------------------- sampling
    def sample(self, batch_size: int) -> dict[str, Tensor] | None:
        """Draw a batch, or None if the buffer is empty right now.

        The caller's `len(buffer) >= batch_size` check cannot be trusted: the
        rollout thread may call `discard_episode` in between, which shrinks the
        buffer and can empty it outright. So emptiness is re-tested here, under
        the same lock that does the gather.
        """
        # The host-side gather happens under the lock; only the (asynchronous)
        # host-to-device copies are left outside it, so a concurrent write can
        # never land in a row this batch is still reading.
        with self._lock:
            if self.size == 0:
                return None
            idx = torch.randint(0, self.size, (batch_size,), generator=self._gen)
            batch = {
                "x": self.x[idx],
                "action": self.action[idx],
                "ref": self.ref[idx],
                "reward_disc": self.reward_disc[idx],
                "x_next": self.x_next[idx],
                "ref_next": self.ref_next[idx],
                "done": self.done[idx],
                "actual_steps": self.actual_steps[idx],
                "bc_weight": self.bc_weight[idx],
            }
        if self.device.type != "cuda":
            return batch
        return {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in batch.items()}

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        """Persist the buffer so a real-robot run can resume after a stop."""
        with self._lock:
            self._save_locked(path)

    def _save_locked(self, path: str | Path) -> None:
        n = self.size
        torch.save(
            {
                "x": self.x[:n],
                "action": self.action[:n],
                "ref": self.ref[:n],
                "reward_disc": self.reward_disc[:n],
                "x_next": self.x_next[:n],
                "ref_next": self.ref_next[:n],
                "bc_weight": self.bc_weight[:n],
                "done": self.done[:n],
                "actual_steps": self.actual_steps[:n],
                "ptr": self._ptr,
                "size": n,
                "total_added": self.total_added,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        n = min(int(sd["size"]), self.capacity)
        for key in (
            "x", "action", "ref", "reward_disc", "x_next", "ref_next", "done", "actual_steps",
        ):
            getattr(self, key)[:n] = sd[key][:n]
        # Buffers written before BC weighting existed carry no column; those
        # transitions predate takeover gating, so full weight is the honest default.
        self.bc_weight[:n] = sd["bc_weight"][:n] if "bc_weight" in sd else 1.0
        self.size = n
        self._ptr = n % self.capacity
        self._episode_ptr = self._ptr
        self._episode_added = 0
        self.total_added = int(sd.get("total_added", n))
