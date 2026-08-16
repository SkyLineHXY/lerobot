"""The contract a human-in-the-loop device implements for the rollout worker.

Interventions are decided *while* a chunk is running and the teleoperator has to
step the environment itself to produce commands, so :meth:`InterventionManager.run_chunk`
executes the whole chunk and hands back everything the rollout worker would
otherwise have gathered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class InterventionResult:
    """Outcome of a chunk driven by the human, mirroring the worker's own loop."""

    action_chunk: Tensor  # (C, action_dim), normalized action space
    obs_list: list[dict]  # observation after every executed step
    rewards: Tensor  # (C,)
    n_steps: int  # steps actually executed
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class InterventionManager:
    """No-op manager: never intervenes. Base class for real teleop devices."""

    def check(self) -> bool:
        return False

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        return None

    def on_reset(self) -> None:
        """Called at every episode reset, before the first chunk."""
        return None

    def close(self) -> None:
        return None
