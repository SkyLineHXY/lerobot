"""Wire format for the rollout <-> learner split.

The rollout process streams *buffer operations*, not finished transitions. The
buffer's own logic — assembling the episode trace, choosing window anchors,
terminal vs truncated handling — is subtle enough that having two
implementations of it would be a standing bug source. So `ChunkReplayBuffer`
stays exactly as it is, lives in the learner process, and this module just
carries the method calls across.

Ordering is what makes the episode semantics survive the split: a gRPC stream
preserves order, and the rollout emits `discard` before starting the next
episode, so the learner sees the same buffer state the in-process version would
have.

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


def start_op(episode_id: int, warmup: bool = False) -> dict[str, Any]:
    return {"kind": START, "episode_id": episode_id, "warmup": bool(warmup)}


def chunk_op(episode_id: int, rec: ChunkRecord, warmup: bool = False) -> dict[str, Any]:
    return {
        "kind": CHUNK,
        "episode_id": episode_id,
        # Whether this chunk was collected under the base VLA. It selects the
        # actor's BC/Q weight pair; deriving it from a step count on the learner
        # side would need the rollout's step counter too.
        "warmup": bool(warmup),
        "xs": rec.xs,
        "x_offsets": rec.x_offsets,
        "actions": rec.actions,
        "rewards": rec.rewards,
        "refs": rec.refs,
        "aligned": rec.aligned,
        "source": int(rec.source),
        "done": bool(rec.done),
    }


def end_op(
    episode_id: int,
    x_last: Tensor | None,
    ref_last: Tensor | None,
    success: bool = False,
    truncation_is_failure: bool = False,
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "kind": END,
        "episode_id": episode_id,
        "success": bool(success),
        # Travels with the op rather than being read from the learner's config:
        # whether running out the clock counts as a failure is a property of the
        # environment, which only the rollout process has.
        "truncation_is_failure": bool(truncation_is_failure),
    }
    if x_last is not None:
        op["x_last"] = x_last
    if ref_last is not None:
        op["ref_last"] = ref_last
    return op


def discard_op(episode_id: int) -> dict[str, Any]:
    return {"kind": DISCARD, "episode_id": episode_id}


def record_from_op(op: dict[str, Any]) -> ChunkRecord:
    return ChunkRecord(
        xs=op["xs"],
        x_offsets=op["x_offsets"],
        actions=op["actions"],
        rewards=op["rewards"],
        refs=op["refs"],
        aligned=op["aligned"],
        source=int(op["source"]),
        done=bool(op["done"]),
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
