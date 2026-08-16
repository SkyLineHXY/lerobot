"""Multi-process rollout/learner split over gRPC.

The threaded backend (`rlt/learner.py`) remains the default and the reference
implementation: with RL you cannot tell "my process split has a weight-staleness
bug" from "this task is just hard" without a known-good baseline to compare
against. Select this one with `concurrency.mode=processes`.

Nothing here is imported by `lerobot.rlt`: grpcio is an optional extra
(`pip install -e ".[async]"`), so the import stays lazy inside `train_online`.
"""

from .client import RemoteActorMirror, RemoteBufferSink, start_client_threads
from .learner_proc import RemoteLearner, apply_op, serve

__all__ = [
    "RemoteActorMirror",
    "RemoteBufferSink",
    "RemoteLearner",
    "apply_op",
    "serve",
    "start_client_threads",
]
