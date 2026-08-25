"""RLT: RL Token — model code (Xu et al., 2026, Physical Intelligence).

Contains everything that defines the *model*: the configuration tree, the RL
token encoder/decoder that compresses the frozen VLA's embeddings, the
chunk-level actor-critic that refines the VLA's action chunks, and the
controller that ties them together at a chunk boundary.

The training loops, replay buffer and environments live in `lerobot.rlt`.
"""

from .configuration_rlt import (
    ActorCriticConfig,
    CameraSpec,
    ChunkEnvConfig,
    ConcurrencyConfig,
    InsertionEnvConfig,
    LiberoEnvConfig,
    MockEnvConfig,
    OnlineRLConfig,
    PiperEnvConfig,
    PiperLeaderTeleopConfig,
    RLTokenConfig,
    RLTokenTrainConfig,
    RLTOnlineTrainConfig,
    RolloutViewConfig,
    SimTeleopConfig,
)
from .modeling_rlt import RLTController
from .modeling_rlt_ac import ChunkActor, ChunkCritic, RLTAgent, TransitionSource
from .modeling_rlt_token import RLTokenModule, SmolVLAPrefixExtractor
from .processor_rlt import (
    check_stage1_matches_policy,
    load_smolvla_policy,
    load_stage1_processors,
)

__all__ = [
    "ActorCriticConfig",
    "CameraSpec",
    "ChunkActor",
    "ChunkCritic",
    "ChunkEnvConfig",
    "ConcurrencyConfig",
    "InsertionEnvConfig",
    "LiberoEnvConfig",
    "MockEnvConfig",
    "OnlineRLConfig",
    "PiperEnvConfig",
    "RolloutViewConfig",
    "check_stage1_matches_policy",
    "SimTeleopConfig",
    "PiperLeaderTeleopConfig",
    "RLTAgent",
    "RLTController",
    "RLTOnlineTrainConfig",
    "RLTokenConfig",
    "RLTokenModule",
    "RLTokenTrainConfig",
    "SmolVLAPrefixExtractor",
    "TransitionSource",
    "load_smolvla_policy",
    "load_stage1_processors",
]
