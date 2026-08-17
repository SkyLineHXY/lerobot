"""Leader-arm teleoperation for a Piper follower."""

from __future__ import annotations

import logging

import torch

from lerobot.rlt.envs.piper import (
    JOINT_KEYS_6,
    follower_action_to_leader,
    leader_action_to_follower,
)

from .base import InterventionManager, InterventionResult
from .keys import KeyboardEventListener

logger = logging.getLogger(__name__)


class PiperLeaderIntervention(InterventionManager):
    """Leader-arm teleoperation for a Piper follower.

    The leader is a second Piper arm held under gravity compensation, so its
    joints map 1:1 onto the follower's 7-dim action and no IK is involved. On
    exit the leader is put back into command mode and driven to the follower's
    current pose, so the next takeover does not begin with a jump.

    The same alignment runs at every episode reset (:meth:`on_reset`), because
    the leader enters gravity compensation the moment it connects: without it
    the *first* takeover of a run would start from wherever the limp leader
    happens to be resting and drag the follower there.
    """

    def __init__(
        self,
        leader,
        env,
        keys: KeyboardEventListener,
        use_calibrated_offsets: bool = False,
        max_takeover_delta_rad: float = 0.15,
    ):
        self.leader = leader
        self.env = env  # PiperChunkEnv
        self.keys = keys
        # `get_action()` returns offsets from the calibrated neutral pose; the
        # stage-1 dataset actions are absolute joint angles. Default to the
        # absolute reading so human corrections land in the policy's own action
        # space instead of being shifted by the leader's neutral pose.
        self.use_calibrated_offsets = use_calibrated_offsets
        self.max_takeover_delta_rad = max_takeover_delta_rad
        self._active = False

    def _read_leader(self) -> dict[str, float]:
        """One leader sample, already renamed to the follower's joint keys."""
        raw = (
            self.leader.get_action()
            if self.use_calibrated_offsets
            else self.leader.get_raw_action()
        )
        return leader_action_to_follower(raw)

    def _takeover_delta(self) -> float:
        """Largest |leader - follower| over the 6 arm joints, in radians."""
        leader = self._read_leader()
        follower = self.env.raw_joint_action()
        return max(abs(leader[k] - follower[k]) for k in JOINT_KEYS_6)

    def check(self) -> bool:
        return self.keys.intervening

    def align(self) -> None:
        """Drive the leader to the follower's pose and hold it in command mode."""
        self.leader.send_feedback(follower_action_to_leader(self.env.raw_joint_action()))
        self.leader.set_manual_control(False)
        self._active = False

    def on_reset(self) -> None:
        self.align()

    def _enter(self) -> bool:
        """Release the leader for the operator. False if it is unsafe to do so."""
        if self._active:
            return True
        # The leader has been holding the follower's pose since the last exit
        # (or since `on_reset`), so a large gap means the operator has already
        # dragged it somewhere else — releasing now would make the follower
        # chase that pose. Refuse and make them re-press after re-aligning.
        delta = self._takeover_delta()
        if delta > self.max_takeover_delta_rad > 0:
            logger.warning(
                "Intervention refused: leader is %.3f rad away from the follower "
                "(limit %.3f). Re-aligning; press space again once the leader is "
                "back on the robot's pose.",
                delta,
                self.max_takeover_delta_rad,
            )
            self.keys.clear_intervention()
            self.align()
            return False
        logger.info("Intervention: leader arm released (gravity compensation).")
        self.leader.set_manual_control(True)
        self._active = True
        return True

    def _exit(self) -> None:
        if not self._active:
            return
        # Hand the leader back to command mode and align it with the follower
        # so the operator's next grab starts from the current robot pose.
        self.align()
        logger.info("Intervention: leader arm re-engaged.")

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        if not self.check():
            self._exit()
            return None
        if not self._enter():
            return None

        actions, rewards, obs_list = [], [], []
        done = truncated = False
        for _ in range(chunk_len):
            raw = self._read_leader()  # {"joint_N.pos": float, ...}
            norm_action = self.env.raw_action_to_normalized(raw)
            obs, r, done, truncated = self.env.apply_action(norm_action)
            actions.append(norm_action)
            rewards.append(r)
            obs_list.append(obs)
            self.notify_step(obs)
            if done or truncated:
                break
            if not self.check():  # operator let go mid-chunk
                break

        n = len(actions)
        chunk = torch.stack(actions)
        if n < chunk_len:  # hold the last human command for the unused tail
            chunk = torch.cat([chunk, chunk[-1:].expand(chunk_len - n, -1)], dim=0)
        rew = torch.zeros(chunk_len)
        rew[:n] = torch.tensor(rewards, dtype=torch.float32)

        if not self.check():
            self._exit()

        return InterventionResult(
            action_chunk=chunk,
            obs_list=obs_list,
            rewards=rew,
            n_steps=n,
            done=done,
            truncated=truncated,
            info={"intervention": True},
        )

    def close(self) -> None:
        try:
            self._exit()
        finally:
            # The leader owns a CAN handle and a 200 Hz gravity-compensation
            # thread; leaving it connected keeps the arm enabled after the run.
            if getattr(self.leader, "is_connected", False):
                self.leader.disconnect()
