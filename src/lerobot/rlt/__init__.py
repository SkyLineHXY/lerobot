"""RLT online RL: training loops, replay buffer, environments, human-in-loop.

The model itself (RL token, chunk actor-critic, controller, configs) lives in
`lerobot.policies.rlt`; this package is everything around it — the two training
entry points, the chunk-level replay buffer, the environments the rollout
worker can drive (`envs/`), the async learner thread, and the operator
interfaces for sparse reward labels and teleoperated interventions (`teleop/`).

See `README.md` for the pipeline, the buffer's write ordering, and the
concurrency model.
"""

from lerobot.utils.multiprocess_compat import patch_resource_tracker

from .learner import ActorMirror, LearnerThread
from .replay_buffer import ChunkRecord, ChunkReplayBuffer
from .rollout import RolloutWorker
from .teleop import (
    DeviceIntervention,
    InterventionManager,
    InterventionResult,
    KeyboardEventListener,
)

# LIBERO drags in `multiprocess`, whose resource tracker raises at interpreter
# exit on CPython 3.12.0 and buries whatever really ended the run. Patched here
# rather than at the LIBERO import because a spawned learner re-imports from
# scratch and only ever reaches this package.
patch_resource_tracker()

__all__ = [
    "ActorMirror",
    "ChunkRecord",
    "ChunkReplayBuffer",
    "DeviceIntervention",
    "InterventionManager",
    "InterventionResult",
    "KeyboardEventListener",
    "LearnerThread",
    "RolloutWorker",
]
