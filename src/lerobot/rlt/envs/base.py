"""The environment contract the RLT rollout worker drives, and the normalisation
boundary every real environment has to cross.

The controller works entirely in the policy's *normalized* action space. An env
therefore has two conversions to own: un-normalize on the way out (policy action
-> physical command) and normalize on the way in (human teleop command -> the
space the replay buffer stores). Both must use the stage-1 statistics; deriving
either one independently is how mean/std and min/max modes silently drift apart.
"""

from __future__ import annotations

import copy
from typing import Protocol

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION

from ..teleop.base import InterventionResult


class ChunkEnv(Protocol):
    """Minimal single-env protocol used by the online trainer."""

    action_dim: int
    max_episode_steps: int

    def reset(self) -> dict: ...

    def step(self, action: Tensor) -> tuple[dict, float, bool]:
        """Apply one control step; returns (obs, reward, done)."""
        ...

    def obs_to_batch(self, obs_list: list[dict], device) -> dict[str, Tensor]:
        """Convert raw observations to a SmolVLA-preprocessed model batch."""
        ...

    def run_intervention(self, chunk_len: int) -> InterventionResult | None:
        """Let a human drive the next chunk, if one is taking over right now.

        Interventions are decided *during* execution on a real robot, and the
        teleoperator has to step the arm itself to produce the commands. So the
        manager runs the whole chunk and hands back everything the rollout
        worker would otherwise have collected; returning None means "no human,
        execute the policy chunk normally".
        """
        ...

    def intervention_pending(self) -> bool:
        """Is a human about to take over the next chunk?

        Cheap, side-effect free probe. The rollout worker uses it to move the
        VLA sampling for that chunk off the control thread — the reference is
        still stored, but the operator no longer waits for it.
        """
        ...

    def close(self) -> None: ...


def find_action_normalizer(preprocessor):
    """Return the pipeline step that knows how to normalize an ACTION tensor."""
    for step in getattr(preprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if stats and ACTION in stats and hasattr(step, "_normalize_action"):
            return step
    return None


class ActionNormalizer:
    """Env-space action -> the normalized space the policy and buffer live in.

    Only needed on the human-in-the-loop path: a teleop device produces a
    command in physical units (joint radians, or LIBERO's delta-EE box), but
    `ChunkRecord.actions` and the actor's BC target are both normalized. Reuses
    the stage-1 normalizer step rather than re-deriving the transform.

    It reuses a private *copy* of that step, pinned to the CPU. The pipeline's
    own step keeps its statistics on whichever device it was last called with
    and migrates them lazily (`_apply_transform` reads `first_stat.device`, then
    `to()` rebinds the whole `_tensor_stats` dict). Teleop commands arrive on the
    CPU while the rollout worker's assembler thread is inside that same step with
    CUDA observations, so normalizing through the shared step drags the stats to
    the CPU between the assembler's device check and its read, and observation
    normalization dies with "found at least two devices, cuda:0 and cpu". A copy
    keeps the exact stage-1 transform without the shared mutable device state.
    """

    def __init__(self, preprocessor, action_dim: int):
        self.action_dim = action_dim
        self._step = None
        self._preprocessor = preprocessor

    def __call__(self, raw: Tensor) -> Tensor:
        if self._step is None:
            self._step = self._private_step()
        raw = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
        return self._step._normalize_action(raw, inverse=False)[: self.action_dim]

    def _private_step(self):
        step = find_action_normalizer(self._preprocessor)
        if step is None:
            raise RuntimeError(
                "No action normalizer found in the stage-1 preprocessor; human "
                "interventions cannot be expressed in the policy's action space."
            )
        return copy.deepcopy(step).to(device="cpu", dtype=torch.float32)
