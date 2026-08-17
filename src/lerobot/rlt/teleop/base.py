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

    # Set by the rollout worker when an operator view is running. The manager
    # steps the env itself, so without this hook the picture would only refresh
    # once the whole takeover chunk is over — a chunk late, which is exactly the
    # feedback the human is steering by.
    on_step = None

    def check(self) -> bool:
        return False

    def notify_step(self, obs) -> None:
        if self.on_step is not None:
            self.on_step(obs)

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        return None

    def on_reset(self) -> None:
        """Called at every episode reset, before the first chunk."""
        return None

    def close(self) -> None:
        return None
