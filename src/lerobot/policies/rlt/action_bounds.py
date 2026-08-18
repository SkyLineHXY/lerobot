"""Per-dimension action bounds derived from the dataset's action quantiles.

openpi-RLT does not reuse the VLA's own action normalisation for RL. It
re-normalises with dataset quantiles (`action_representation.py`:
``(x - q01) / (q99 - q01) * 2 - 1``) and *that* is why its ``+-1`` bound is
safe: ``+-1`` is the 1st/99th percentile, so it touches ~1% of the data.

SmolVLA normalises actions with mean/std, where ``+-1`` is one standard
deviation. Measured on libero_goal, 27.2% of action elements fall outside it —
a scalar ``action_clip=1.0`` truncates more than a quarter of every chunk, and
truncates precisely the largest, most task-relevant commands. Un-normalised,
that is ~4 mm per step of translation thrown away and a gripper-close command
delivered at 71% strength.

This module takes openpi-RLT's idea and keeps our learning space. The bound
becomes per-dimension, placed at the dataset quantiles but *expressed in the
VLA's normalised units*::

    bound_i = (q_i - mean_i) / std_i

Learning stays in mean/std space deliberately: there every channel already has
unit variance, so ``action_std`` is the same fraction of natural motion on all
of them and the summed BC term weighs them equally. Quantile space would
unbalance both — the q01..q99 span is 2.3 sigma on `delta_x` but 3.2 sigma on
`delta_roll`, so an identical BC error would count ~2x more on one than the
other. The bound is the one quantity that wants data range rather than data
variance, and it is the only one moved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

BOUNDS_FILENAME = "rl_action_bounds.json"

# Headroom beyond q01..q99. See `ActionBounds.from_quantiles` for why it cannot
# be 1.0. The env clips to its own actuator range afterwards (robosuite's
# `scale_action` clips to +-1 raw), so overshooting here costs nothing.
DEFAULT_MARGIN = 1.1


@dataclass
class ActionBounds:
    """Elementwise ``[low, high]`` for one action, in VLA-normalised units."""

    low: Tensor  # (d,)
    high: Tensor  # (d,)

    def __post_init__(self):
        if self.low.shape != self.high.shape:
            raise ValueError(f"low {tuple(self.low.shape)} != high {tuple(self.high.shape)}")
        if not bool((self.high > self.low).all()):
            raise ValueError("every high bound must exceed its low bound")

    @property
    def action_dim(self) -> int:
        return int(self.low.shape[0])

    @classmethod
    def symmetric(cls, action_dim: int, value: float = 1.0) -> ActionBounds:
        """The old scalar behaviour, kept for the mock env and for tests."""
        return cls(low=torch.full((action_dim,), -value), high=torch.full((action_dim,), value))

    @classmethod
    def from_config(cls, cfg) -> ActionBounds:
        """`cfg.action_bounds` when fitted, else the scalar `cfg.action_clip`."""
        if getattr(cfg, "action_bounds", None) is None:
            return cls.symmetric(cfg.action_dim, cfg.action_clip)
        low, high = cfg.action_bounds
        return cls(
            low=torch.tensor(low, dtype=torch.float32),
            high=torch.tensor(high, dtype=torch.float32),
        )

    def as_config_value(self) -> list[list[float]]:
        return [self.low.tolist(), self.high.tolist()]

    @classmethod
    def from_quantiles(
        cls,
        q_low: np.ndarray | Tensor,
        q_high: np.ndarray | Tensor,
        mean: np.ndarray | Tensor,
        std: np.ndarray | Tensor,
        margin: float = DEFAULT_MARGIN,
        min_half_sigma: float = 1.0,
    ) -> ActionBounds:
        """Map raw-space quantiles into the VLA's normalised space.

        `margin` widens the bound around the quantile midpoint. It must be > 1:
        a saturated channel puts its quantiles exactly on the data — LIBERO's
        gripper is +-1 and nothing else, so q01/q99 *are* the two spikes — and a
        bound sitting exactly there clips the exploration noise back every
        single step, one-sided. The bound is a guard against a diverged actor,
        not an operational constraint; `max_residual` is the operational one.
        """
        q_low, q_high, mean, std = (torch.as_tensor(v, dtype=torch.float32).flatten()
                                    for v in (q_low, q_high, mean, std))
        if margin <= 1.0:
            raise ValueError(f"margin must exceed 1.0 to leave headroom, got {margin}")
        centre = (q_high + q_low) / 2
        half = (q_high - q_low) / 2 * margin
        std = std.clamp_min(1e-6)
        low, high = (centre - half - mean) / std, (centre + half - mean) / std
        # A channel that never moves (a locked joint, a padded dim) has
        # q_low == q_high, and the bound would collapse onto the single value it
        # is allowed to take. One sigma either side is far inside the quantiles
        # of any channel that does move, so this only fires on degenerate ones.
        mid, halfn = (low + high) / 2, ((high - low) / 2).clamp_min(min_half_sigma)
        return cls(low=mid - halfn, high=mid + halfn)

    def to(self, device) -> ActionBounds:
        return ActionBounds(low=self.low.to(device), high=self.high.to(device))

    def clamp(self, action: Tensor) -> Tensor:
        """Clamp a (..., d) action; the bounds broadcast over leading dims."""
        return torch.clamp(action, self.low.to(action), self.high.to(action))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / BOUNDS_FILENAME
        path.write_text(
            json.dumps({"low": self.low.tolist(), "high": self.high.tolist()}, indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ActionBounds | None:
        """Load bounds from a file or the directory holding one; None if absent."""
        path = Path(path)
        if path.is_dir():
            path = path / BOUNDS_FILENAME
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            low=torch.tensor(payload["low"], dtype=torch.float32),
            high=torch.tensor(payload["high"], dtype=torch.float32),
        )


def fit_action_bounds(
    actions: np.ndarray,
    mean: np.ndarray | Tensor,
    std: np.ndarray | Tensor,
    quantile: float = 1.0,
    margin: float = DEFAULT_MARGIN,
    min_half_sigma: float = 1.0,
) -> ActionBounds:
    """Fit bounds from raw dataset actions and the VLA's own mean/std.

    `quantile` is the percentage trimmed from each tail (openpi-RLT uses q01/q99,
    i.e. 1.0). The tails matter: a max/min bound would be set by a single
    outlier frame, which is why quantiles are the right statistic here.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"expected (n_frames, action_dim) actions, got {actions.shape}")
    q_low = np.percentile(actions, quantile, axis=0)
    q_high = np.percentile(actions, 100.0 - quantile, axis=0)
    return ActionBounds.from_quantiles(
        q_low, q_high, mean, std, margin=margin, min_half_sigma=min_half_sigma
    )
