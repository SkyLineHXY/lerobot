"""Wire format for the rollout <-> learner split.

The rollout process streams *buffer operations*, not finished transitions. The
buffer's own logic — lagging one chunk behind so stride-subsampled windows can
span two chunks, terminal vs truncated flushing, episode rewind — is subtle
enough that having two implementations of it would be a standing bug source. So
`ChunkReplayBuffer` stays exactly as it is, lives in the learner process, and
this module just carries the method calls across.

Ordering is what makes the discard semantics survive the split: a gRPC stream
preserves order, and the rollout emits `discard` before starting the next
episode, so the learner's rewind sees the same buffer state the in-process
version would have.

Everything on the wire is a plain dict of tensors and primitives, so the
receiving side can `torch.load(weights_only=True)`. That rules out arbitrary
code execution from the socket, which pickle would not.
"""

from __future__ import annotations

import io
from typing import Any

import torch
from torch import Tensor

from ..replay_buffer import ChunkRecord

START = "start"
CHUNK = "chunk"
END = "end"
DISCARD = "discard"


def start_op(episode_id: int) -> dict[str, Any]:
    return {"kind": START, "episode_id": episode_id}


def chunk_op(episode_id: int, rec: ChunkRecord, warmup: bool = False) -> dict[str, Any]:
    return {
        "kind": CHUNK,
        "episode_id": episode_id,
        # Whether this chunk was collected under the base VLA. The learner flips
        # actor updates on when these stop arriving; deriving it from a step
        # count on the learner side would need the rollout's step counter too.
        "warmup": bool(warmup),
        "xs": rec.xs,
        "actions": rec.actions,
        "rewards": rec.rewards,
        "ref_full": rec.ref_full,
        "done": bool(rec.done),
        # -1 rather than None: `weights_only=True` accepts None, but keeping the
        # payload free of optional types makes the decode side total.
        "done_step": -1 if rec.done_step is None else int(rec.done_step),
        "from_human": bool(rec.from_human),
    }


def end_op(episode_id: int, x_last: Tensor | None) -> dict[str, Any]:
    op: dict[str, Any] = {"kind": END, "episode_id": episode_id}
    if x_last is not None:
        op["x_last"] = x_last
    return op


def discard_op(episode_id: int) -> dict[str, Any]:
    return {"kind": DISCARD, "episode_id": episode_id}


def record_from_op(op: dict[str, Any]) -> ChunkRecord:
    done_step = int(op["done_step"])
    return ChunkRecord(
        xs=op["xs"],
        actions=op["actions"],
        rewards=op["rewards"],
        ref_full=op["ref_full"],
        done=bool(op["done"]),
        done_step=None if done_step < 0 else done_step,
        from_human=bool(op.get("from_human", False)),
    )


def ops_to_bytes(ops: list[dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    torch.save(ops, buffer)
    return buffer.getvalue()


def bytes_to_ops(payload: bytes) -> list[dict[str, Any]]:
    return torch.load(io.BytesIO(payload), weights_only=True)


def params_to_bytes(state_dict: dict[str, Tensor], stats: dict[str, float]) -> bytes:
    """Actor weights plus the learner's own counters.

    The rollout process has no buffer and no optimizer, so without piggybacking
    these it could not log buffer size, update count or losses at all.
    """
    buffer = io.BytesIO()
    torch.save({"actor": state_dict, "stats": stats}, buffer)
    return buffer.getvalue()


def bytes_to_params(payload: bytes) -> tuple[dict[str, Tensor], dict[str, float]]:
    blob = torch.load(io.BytesIO(payload), weights_only=True)
    return blob["actor"], blob.get("stats", {})
