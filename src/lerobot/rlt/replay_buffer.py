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
  reward_disc  sum_{j=1}^{C} gamma^{j-1} r_j            scalar
  x_next       RL state C control steps later           (D_x,)
  ref_next     reference chunk at x_next                (C, d)
  done         episode terminated within the window     scalar 0/1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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

        self.size = 0
        self._ptr = 0
        self._pending: ChunkRecord | None = None
        self._gammas = discount ** torch.arange(chunk_len).float()

    # ------------------------------------------------------------- assembly
    def start_episode(self) -> None:
        if self._pending is not None:
            # Previous episode ended without an explicit terminal/truncation
            # signal; flush it as truncated (no bootstrap zeroing).
            self._flush_terminal(self._pending, truncated=True)
        self._pending = None

    def add_chunk(self, rec: ChunkRecord) -> None:
        """Feed one executed chunk; emits transitions for the previous one."""
        if self._pending is not None:
            self._emit_pair(self._pending, rec)
        if rec.done:
            self._flush_terminal(rec)
            self._pending = None
        else:
            self._pending = rec

    def end_episode(self) -> None:
        """Call when an episode ends without `done` (e.g. time-limit truncation)."""
        if self._pending is not None:
            self._flush_terminal(self._pending, truncated=True)
        self._pending = None

    def _emit_pair(self, k: ChunkRecord, k1: ChunkRecord) -> None:
        c = self.chunk_len
        for i, o in enumerate(range(0, c, self.stride)):
            if o == 0:
                action = k.actions
                rewards = k.rewards
            else:
                # Only steps actually executed in k+1 may enter the window
                # (k+1 can end early on success or truncation).
                n1 = min(o, k1.done_step) if k1.done_step is not None else o
                action = torch.cat([k.actions[o:], k1.actions[:n1]], dim=0)
                rewards = torch.cat([k.rewards[o:], k1.rewards[:n1]], dim=0)
                if action.shape[0] < c:  # k1 ended before offset window filled
                    pad = c - action.shape[0]
                    action = torch.cat([action, action[-1:].expand(pad, -1)], dim=0)
                    rewards = torch.cat([rewards, torch.zeros(pad)], dim=0)
            done_in_window = bool(k1.done and o > 0 and (k1.done_step or 0) <= o)
            # x_next at offset o of chunk k+1 (valid while k1 ran that far).
            i_next = min(i, k1.xs.shape[0] - 1)
            self._store(
                x=k.xs[i],
                action=action,
                ref=k.ref_full[o : o + c],
                rewards=rewards,
                x_next=k1.xs[i_next],
                ref_next=k1.ref_full[o : o + c],
                done=float(done_in_window),
            )

    def _flush_terminal(self, rec: ChunkRecord, truncated: bool = False) -> None:
        c = self.chunk_len
        d_step = rec.done_step if rec.done_step is not None else c
        for i, o in enumerate(range(0, c, self.stride)):
            if o >= d_step:
                break
            action = rec.actions[o:d_step]
            rewards = rec.rewards[o:d_step]
            if action.shape[0] < c:
                pad = c - action.shape[0]
                action = torch.cat([action, action[-1:].expand(pad, -1)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(pad)], dim=0)
            self._store(
                x=rec.xs[i],
                action=action,
                ref=rec.ref_full[o : o + c],
                rewards=rewards,
                x_next=rec.xs[i],  # unused: done masks the bootstrap
                ref_next=torch.zeros_like(rec.ref_full[:c]),
                done=0.0 if truncated else 1.0,
            )

    def _store(self, x, action, ref, rewards, x_next, ref_next, done) -> None:
        p = self._ptr
        self.x[p] = x.cpu()
        self.action[p] = action.cpu()
        self.ref[p] = ref[: self.chunk_len].cpu()
        self.reward_disc[p] = (rewards[: self.chunk_len] * self._gammas).sum()
        self.x_next[p] = x_next.cpu()
        self.ref_next[p] = ref_next[: self.chunk_len].cpu()
        self.done[p] = done
        self._ptr = (p + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------- sampling
    def sample(self, batch_size: int) -> dict[str, Tensor]:
        idx = torch.from_numpy(np.random.randint(0, self.size, size=batch_size))
        dev = self.device
        return {
            "x": self.x[idx].to(dev),
            "action": self.action[idx].to(dev),
            "ref": self.ref[idx].to(dev),
            "reward_disc": self.reward_disc[idx].to(dev),
            "x_next": self.x_next[idx].to(dev),
            "ref_next": self.ref_next[idx].to(dev),
            "done": self.done[idx].to(dev),
        }

    def __len__(self) -> int:
        return self.size