"""Leader-arm teleoperation for a Piper follower."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np
import torch

from lerobot.rlt.envs.piper import (
    JOINT_KEYS_6,
    JOINT_ORDER,
    follower_action_to_leader,
    leader_action_to_follower,
)

from .base import InterventionManager, InterventionResult
from .keys import KeyboardEventListener

logger = logging.getLogger(__name__)

IDLE, ENTERING, ACTIVE, EXITING = "idle", "entering", "active", "exiting"


class PiperLeaderIntervention(InterventionManager):
    """Leader-arm teleoperation for a Piper follower.

    The leader is a second Piper arm held under gravity compensation, so its
    joints map 1:1 onto the follower's 7-dim action and no IK is involved.

    Two things here exist to keep the operator's hand and the follower in sync.

    **The command path is the streamer's, not this loop's.** `run_chunk` only
    *samples* the leader at the env's control rate to build the recorded
    action; `PiperCommandStreamer` is what drives the arm, at the leader's own
    ~188 Hz feedback rate, and it keeps driving through the chunk boundary
    while the rollout thread is busy with the VLA.

    **Entering and leaving a takeover happen off the control thread.**
    `set_manual_control` waits for the arm to report enabled (up to
    `enable_timeout_s`, 3 s) and starts or joins a gravity-compensation thread;
    running that inline froze the arm for seconds at exactly the moment the
    operator reached for it. A transition in flight simply lets the policy keep
    the current chunk, and the takeover begins at the next one.

    Alignment (leader driven onto the follower's pose, then held in command
    mode) runs at every episode reset, because the leader enters gravity
    compensation the moment it connects: without it the *first* takeover of a
    run would start from wherever the limp leader happens to be resting and
    drag the follower there.
    """

    def __init__(
        self,
        leader,
        env,
        keys: KeyboardEventListener,
        use_calibrated_offsets: bool = False,
        max_takeover_delta_rad: float = 0.15,
        align_settle_s: float = 1.0,
        action_source: str = "leader",
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
        # `align()` issues an asynchronous JointCtrl and does not wait for the
        # leader to arrive. Reading the takeover gap before it settles measures
        # a pose the leader has already left, and the operator gets a takeover
        # refused for a gap that no longer exists.
        self.align_settle_s = align_settle_s
        if action_source not in ("leader", "issued"):
            raise ValueError(f"unknown action_source {action_source!r}; use 'leader' or 'issued'")
        # What lands in the replay buffer as the human's action: the leader's
        # own pose, or the slew-limited command the follower actually executed.
        # They differ only while the limiter saturates, i.e. when the operator
        # is moving faster than `max_joint_vel_rad_s`.
        self.action_source = action_source

        self._state = IDLE
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._error: BaseException | None = None

    # ----------------------------------------------------------- leader I/O
    def _read_leader(self) -> dict[str, float]:
        """One leader sample, already renamed to the follower's joint keys."""
        raw = (
            self.leader.get_action()
            if self.use_calibrated_offsets
            else self.leader.get_raw_action()
        )
        return leader_action_to_follower(raw)

    def _leader_vector(self) -> np.ndarray:
        raw = self._read_leader()
        return np.array([raw[k] for k in JOINT_ORDER], dtype=np.float32)

    def _leader_timestamp(self) -> float:
        """Timestamp of the leader's joint-feedback frame, or 0 if unavailable.

        The streamer uses it to command on new samples instead of on a clock;
        0 makes it fall back to a fixed rate.
        """
        try:
            return float(getattr(self.leader.arm.GetArmJointMsgs(), "time_stamp", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _takeover_delta(self) -> float:
        """Largest |leader - follower| over the 6 arm joints, in radians."""
        leader = self._read_leader()
        follower = self.env.raw_joint_action()
        return max(abs(leader[k] - follower[k]) for k in JOINT_KEYS_6)

    # ------------------------------------------------------ state machine
    def check(self) -> bool:
        return self.keys.intervening

    def _raise_deferred(self) -> None:
        exc, self._error = self._error, None
        if exc is not None:
            raise exc

    def _spawn(self, fn) -> None:
        def run():
            try:
                fn()
            except BaseException as exc:  # re-raised on the control thread
                self._error = exc
                with self._lock:
                    self._state = IDLE

        self._worker = threading.Thread(target=run, name="piper-takeover", daemon=True)
        self._worker.start()

    def align(self) -> None:
        """Drive the leader to the follower's pose and hold it in command mode."""
        self.leader.send_feedback(follower_action_to_leader(self.env.raw_joint_action()))
        self.leader.set_manual_control(False)
        if self.align_settle_s > 0:
            time.sleep(self.align_settle_s)

    def on_reset(self) -> None:
        self._join_transition()
        with self._lock:
            self._state = IDLE
        self.env.streamer.follow_target(seed=self.env.streamer.read_joints())
        self.align()

    def _join_transition(self, timeout: float = 10.0) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
            self._worker = None
        self._raise_deferred()

    def _enter(self) -> None:
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
            with self._lock:
                self._state = IDLE
            return

        self.leader.set_manual_control(True)
        self.env.streamer.follow_leader(self._leader_vector, self._leader_timestamp)
        with self._lock:
            self._state = ACTIVE
        logger.info("Intervention: leader arm released (gravity compensation).")

    def _exit(self) -> None:
        # Seed the streamer from where the human actually left the arm before
        # handing the command path back: its stored target is the policy's,
        # from before the takeover, and switching to it unseeded would snap the
        # follower back across everything the operator just corrected.
        self.env.streamer.follow_target(seed=self.env.streamer.read_joints())
        self.align()
        with self._lock:
            self._state = IDLE
        logger.info("Intervention: leader arm re-engaged.")

    def _begin(self, state: str, fn) -> None:
        with self._lock:
            self._state = state
        self._spawn(fn)

    # ----------------------------------------------------------- the chunk
    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        self._raise_deferred()
        self.env.streamer.check()
        with self._lock:
            state = self._state

        if not self.check():
            if state == ACTIVE:
                self._begin(EXITING, self._exit)
            return None
        if state != ACTIVE:
            if state == IDLE:
                self._begin(ENTERING, self._enter)
            # ENTERING / EXITING: the policy keeps this chunk and the takeover
            # starts at the next boundary, rather than the operator waiting out
            # a multi-second enable handshake with the arm standing still.
            return None

        actions, rewards, obs_list = [], [], []
        done = truncated = False
        for _ in range(chunk_len):
            norm_action = self._sample_action()
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
            self._begin(EXITING, self._exit)

        return InterventionResult(
            action_chunk=chunk,
            obs_list=obs_list,
            rewards=rew,
            n_steps=n,
            done=done,
            truncated=truncated,
            info={"intervention": True},
        )

    def _sample_action(self) -> torch.Tensor:
        """One human action in the policy's normalized space.

        `leader` records what the operator's hand asked for, `issued` what the
        follower was actually told to do. They only diverge while the slew
        limiter saturates.
        """
        if self.action_source == "issued":
            command = self.env.streamer.last_command
            if command is not None:
                raw = dict(zip(JOINT_ORDER, command.tolist(), strict=True))
                return self.env.raw_action_to_normalized(raw)
        return self.env.raw_action_to_normalized(self._read_leader())

    def close(self) -> None:
        try:
            self._join_transition(timeout=5.0)
            with self._lock:
                active = self._state == ACTIVE
            if active:
                self._exit()
        finally:
            # The leader owns a CAN handle and a 200 Hz gravity-compensation
            # thread; leaving it connected keeps the arm enabled after the run.
            if getattr(self.leader, "is_connected", False):
                self.leader.disconnect()
