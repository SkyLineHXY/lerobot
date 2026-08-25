"""Real Piper arm as an RLT chunk environment (paper Sec. V).

This closes the two gaps that make `MockManipEnv` unusable on hardware:

1. **The normalisation boundary.** The controller works entirely in SmolVLA's
   normalized action space; observations must go through the same
   preprocessor, and actions must come back through the same postprocessor,
   that stage 1 was trained with. Loading both from the stage-1 output
   directory is what keeps z_rl and the reference chunks consistent.
2. **Motion limits and the command path.** Piper takes absolute joint targets,
   and in position mode every one of them is a position *step* whose size is
   set by the command period. `PiperCommandStreamer` therefore owns the bus on
   its own thread: the chunk loop only names a target, so the arm keeps being
   commanded across a chunk boundary instead of freezing for the length of a
   VLM forward. Limiting is a slew rate in rad/s anchored on the last command,
   and scales the whole vector rather than clipping per joint, so it never
   distorts the direction of a motion.

Proprioception is position *and* velocity (paper App. B); velocity is obtained
by finite differences of the measured joints, since the Piper bus exposes
positions only.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np
import torch
from torch import Tensor

from lerobot.utils.constants import OBS_STATE

from ..teleop.base import InterventionManager, InterventionResult
from ..teleop.keys import KeyboardEventListener
from .base import ActionNormalizer

logger = logging.getLogger(__name__)

JOINT_ORDER = [f"joint_{i}.pos" for i in range(1, 8)]

# The 6 arm joints. The gripper is excluded wherever a joint *angle* comparison
# is made — it is an opening in metres, so a mismatch there is harmless.
JOINT_KEYS_6 = JOINT_ORDER[:6]

# PiperLeader keeps PIPER_ACTION_KEYS and calls the gripper `gripper.pos`, while
# the Piper follower treats it as a 7th joint `joint_7.pos`. The physical
# quantities are identical (joint radians, gripper opening in metres), so this is
# a rename, never a conversion.
LEADER_KEY_FOR_FOLLOWER = {
    **{key: key for key in JOINT_ORDER[:6]},
    JOINT_ORDER[6]: "gripper.pos",
}


def rate_limit_joints(
    target: np.ndarray, anchor: np.ndarray, max_joint_step_rad: float
) -> tuple[np.ndarray, bool]:
    """Scale the whole step so no arm joint exceeds the per-step limit.

    Scaling uniformly (rather than clipping each joint) preserves the direction
    of the commanded motion; per-joint clipping would silently bend the
    trajectory. Returns the limited target and whether limiting kicked in.

    `anchor` is what the step is measured from. Prefer the previously issued
    command (see JointSlewLimiter) over the measured follower pose: anchoring on
    the measurement turns saturation into positive feedback, because a follower
    that lags pulls the next command back toward itself and can never catch up.
    """
    delta = target[:6] - anchor[:6]
    peak = float(np.abs(delta).max()) if delta.size else 0.0
    saturated = peak > max_joint_step_rad > 0
    if saturated:
        delta = delta * (max_joint_step_rad / peak)
    limited = target.copy()
    limited[:6] = anchor[:6] + delta
    return limited, saturated


class JointSlewLimiter:
    """Per-arm slew-rate limiter expressed in rad/s and anchored on the last command.

    Two things this fixes over calling `rate_limit_joints(target, measured, step)`:

    * **Units.** A per-tick radian budget silently changes meaning whenever the
      command rate changes — the same 0.05 was 1.5 rad/s at 30 Hz and 1.25 rad/s
      at 25 Hz, and would become 9.4 rad/s once commands run at the leader's
      ~188 Hz feedback rate. Holding the limit in rad/s keeps it physical.
    * **Anchor.** Anchoring on the previous command makes this a true slew-rate
      limiter, so follower encoder noise and torn CAN reads no longer leak into
      the command.

    `max_lead_rad` is the safety net that the measurement anchor used to provide
    for free: it stops the command from running away from a physically blocked
    arm (which would otherwise lunge once the obstruction clears). It only
    engages once the command is already that far ahead, so it is never part of
    the normal control path.
    """

    def __init__(self, max_vel_rad_s: float, max_lead_rad: float = 0.5):
        self.max_vel_rad_s = max_vel_rad_s
        self.max_lead_rad = max_lead_rad
        self._prev_cmd: np.ndarray | None = None

    @property
    def seeded(self) -> bool:
        return self._prev_cmd is not None

    def reset(self, measured: np.ndarray) -> None:
        """Re-seed from the measured pose — on init and on every takeover.

        Without this the first command after a handover would step from a stale
        target the arm has long since left.
        """
        self._prev_cmd = np.asarray(measured, dtype=np.float32).copy()

    def __call__(
        self, target: np.ndarray, measured: np.ndarray, dt: float
    ) -> tuple[np.ndarray, bool]:
        if self._prev_cmd is None:
            self.reset(measured)

        limited, saturated = rate_limit_joints(target, self._prev_cmd, self.max_vel_rad_s * dt)

        if self.max_lead_rad > 0:
            lead = limited[:6] - measured[:6]
            peak_lead = float(np.abs(lead).max()) if lead.size else 0.0
            if peak_lead > self.max_lead_rad:
                limited = limited.copy()
                limited[:6] = measured[:6] + lead * (self.max_lead_rad / peak_lead)

        self._prev_cmd = limited.astype(np.float32, copy=True)
        return limited, saturated


class PiperCommandStreamer:
    """Command the follower from its own thread, at its own rate.

    In position mode every command is a position step whose size is set by the
    command *period*, so bolting the command rate to the chunk loop makes the
    arm both coarse and — worse — silent for the whole of a chunk boundary,
    which on this stack is a VLM forward plus a flow-matching sample long. The
    operator feels that as a notch at every boundary of a takeover, and the
    correction they were in the middle of gets recorded as a stall followed by
    a rate-limited catch-up.

    Streaming separates the two clocks: the chunk loop only ever *names* a
    target, and the arm keeps being commanded while that loop is off doing
    inference. During a takeover the streamer is driven by the leader's own
    feedback timestamps instead, so the operator's hand reaches the follower at
    the leader's ~188 Hz without passing through the chunk loop at all.

    The frame-skipping rules and the timestamp pacing come from
    `scripts/lerobot_record_piper.py` (`ArmPair.send` / `_teleop_loop`), where
    they were measured on hardware. That module keeps its own copy on purpose:
    it is validated collection code, and sharing an abstraction with the online
    RL path would put a robot run at risk to save a few lines.
    """

    def __init__(
        self,
        bus,
        *,
        max_joint_vel_rad_s: float,
        max_lead_rad: float = 0.5,
        stream_hz: float = 250.0,
        idle_poll_s: float = 0.001,
        joint_deadband_rad: float = 0.0,
        gripper_deadband: int = 200,
        mode_refresh_interval_s: float = 1.0,
        move_speed_ratio: int = 60,
        dry_run: bool = False,
    ):
        self.bus = bus
        self.limiter = JointSlewLimiter(max_joint_vel_rad_s, max_lead_rad)
        self.stream_hz = stream_hz
        self.idle_poll_s = idle_poll_s
        self.joint_deadband_rad = joint_deadband_rad
        self.gripper_deadband = gripper_deadband
        self.mode_refresh_interval_s = mode_refresh_interval_s
        self.move_speed_ratio = move_speed_ratio
        # 0xAD matches `PiperMotorsBus.write`. Changing it changes the
        # follower's dynamics; do not touch it without a hardware run.
        self.can_mit_flag = 0xAD
        self.dry_run = dry_run

        self._lock = threading.Lock()
        self._target: np.ndarray | None = None
        self._leader_read = None
        self._leader_timestamp = None
        self._last_command: np.ndarray | None = None
        self._last_measured: np.ndarray | None = None

        self._last_mode_refresh = -1e9
        self._last_joints: np.ndarray | None = None
        self._last_gripper: int | None = None
        self._last_leader_ts = -1.0
        self._warned_no_timestamp = False

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.periods: deque[float] = deque(maxlen=512)
        self.saturated: deque[bool] = deque(maxlen=512)

    # ---------------------------------------------------------------- state
    def read_joints(self) -> np.ndarray:
        """The follower's measured pose, straight off the bus.

        Deliberately not `robot.get_observation()`: that also grabs every
        camera, and this runs a few hundred times a second.
        """
        return np.asarray(self.bus._read_joint_list(), dtype=np.float32)

    @property
    def last_command(self) -> np.ndarray | None:
        """The last target actually issued to the arm, after slew limiting."""
        with self._lock:
            return None if self._last_command is None else self._last_command.copy()

    def set_target(self, joints: np.ndarray) -> None:
        with self._lock:
            self._target = np.asarray(joints, dtype=np.float32).copy()

    def follow_leader(self, read_target, timestamp) -> None:
        """Drive from the leader arm until `follow_target` takes it back.

        `timestamp` returns the leader's joint-feedback frame time so the loop
        can run on new samples rather than on a clock; 0 means the SDK does not
        provide one and the loop falls back to `stream_hz`.
        """
        with self._lock:
            self._leader_read = read_target
            self._leader_timestamp = timestamp
            self._last_leader_ts = -1.0

    def follow_target(self, seed: np.ndarray | None = None) -> None:
        """Take the command path back from the leader.

        `seed` replaces the stored target in the same critical section as the
        source switch. Handing back without it would let one tick escape to the
        target the policy left behind before the takeover.
        """
        with self._lock:
            if seed is not None:
                self._target = np.asarray(seed, dtype=np.float32).copy()
            self._leader_read = None
            self._leader_timestamp = None

    def check(self) -> None:
        """Re-raise a streamer failure on the caller's thread."""
        exc, self._error = self._error, None
        if exc is not None:
            raise exc

    # --------------------------------------------------------------- thread
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="piper-command-streamer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            self._loop()
        except BaseException as exc:  # surfaced by `check()` on the control thread
            self._error = exc

    def _loop(self) -> None:
        measured = self.read_joints()
        self.limiter.reset(measured)
        with self._lock:
            if self._target is None:
                self._target = measured.copy()
        min_period = 1.0 / max(1e-6, self.stream_hz)
        last_t = time.perf_counter()

        while not self._stop.is_set():
            target = self._next_target()
            if target is None:  # leader has no new sample yet
                time.sleep(self.idle_poll_s)
                continue

            now = time.perf_counter()
            dt = now - last_t
            if dt < min_period:
                # `time.sleep`, never `precise_sleep`: the latter busy-waits its
                # last 10 ms holding the GIL, which would throttle the very loop
                # it is pacing — and the learner thread it shares the GIL with.
                time.sleep(min_period - dt)
                now = time.perf_counter()
                dt = now - last_t
            last_t = now

            measured = self.read_joints()
            limited, saturated = self.limiter(target, measured, dt)
            if not self.dry_run:
                self._send(limited)
            with self._lock:
                self._last_command = limited.copy()
                self._last_measured = measured.copy()
            self.periods.append(dt)
            self.saturated.append(bool(saturated))

    def _next_target(self) -> np.ndarray | None:
        with self._lock:
            read, timestamp, target = self._leader_read, self._leader_timestamp, self._target
        if read is None:
            return None if target is None else target.copy()

        ts = float(timestamp() or 0.0) if timestamp is not None else 0.0
        if ts:
            if ts == self._last_leader_ts:
                return None
            self._last_leader_ts = ts
        elif not self._warned_no_timestamp:
            # Degrading quietly here looks exactly like a working streamer, so
            # say it: without timestamps the loop cannot tell a new leader
            # sample from a repeat and falls back to a fixed rate.
            logger.warning(
                "Leader feedback carries no timestamp; streaming at a fixed %.0f Hz "
                "instead of on new samples.",
                self.stream_hz,
            )
            self._warned_no_timestamp = True
        return np.asarray(read(), dtype=np.float32)

    def _send(self, joints: np.ndarray) -> None:
        arm = self.bus.piper
        factor = self.bus.joint_factor

        now = time.perf_counter()
        if now - self._last_mode_refresh >= self.mode_refresh_interval_s:
            arm.MotionCtrl_2(0x01, 0x01, self.move_speed_ratio, self.can_mit_flag)
            self._last_mode_refresh = now

        # `joint_deadband_rad` defaults to 0 and should stay there. It compares
        # against the last *commanded* target, so at streaming rates it
        # quantizes slow continuous motion into steps: below
        # deadband * stream_hz the arm waits for the budget to accumulate, and
        # the operator feels that during exactly the slow, precise moves this
        # whole thread exists to make smooth.
        joints6 = np.asarray(joints[:6], dtype=np.float32)
        if (
            self._last_joints is None
            or float(np.abs(joints6 - self._last_joints).max()) >= self.joint_deadband_rad
        ):
            arm.JointCtrl(*(round(float(j) * factor) for j in joints6))
            self._last_joints = joints6.copy()

        gripper = round(abs(float(joints[6])) * 1e6)
        if self._last_gripper is None or abs(gripper - self._last_gripper) >= self.gripper_deadband:
            arm.GripperCtrl(gripper, 1000, 0x01, 0)
            self._last_gripper = gripper

    # ---------------------------------------------------------------- report
    def stats(self) -> dict[str, float]:
        periods = [p for p in self.periods if p > 0]
        return {
            "hz": (len(periods) / sum(periods)) if periods else 0.0,
            "saturation": (sum(self.saturated) / len(self.saturated)) if self.saturated else 0.0,
        }


def jitter_rms(traj: np.ndarray, dt: float) -> float:
    if traj.ndim != 2 or traj.shape[0] < 3:
        return float("nan")
    accel = np.diff(traj, n=2, axis=0) / (dt * dt)
    return float(np.sqrt((accel**2).mean()))


def build_piper_cameras(specs, control_hz: float) -> dict:
    """Only override PIPERConfig's own camera defaults when cameras are named.

    The camera set has to match the one the VLA was fine-tuned on, so an empty
    list means "use whatever the robot config already declares".
    """
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    cameras: dict = {}
    for spec in specs:
        if spec.serial:
            cameras[spec.name] = RealSenseCameraConfig(
                serial_number_or_name=spec.serial,
                width=spec.width,
                height=spec.height,
                fps=spec.fps or int(control_hz),
            )
        elif spec.index_or_path is not None:
            idx = spec.index_or_path
            cameras[spec.name] = OpenCVCameraConfig(
                index_or_path=int(idx) if str(idx).isdigit() else idx,
                width=spec.width,
                height=spec.height,
                fps=spec.fps or int(control_hz),
            )
        else:
            raise ValueError(f"Camera {spec.name!r} needs either `serial` or `index_or_path`.")
    return cameras


def leader_action_to_follower(raw: dict[str, float]) -> dict[str, float]:
    """Leader action dict -> follower key names (joint_1..joint_7.pos)."""
    missing = [k for k in LEADER_KEY_FOR_FOLLOWER.values() if k not in raw]
    if missing:
        raise KeyError(
            f"Leader action is missing {missing}; got {sorted(raw)}. "
            "PiperLeader should return joint_1..joint_6.pos + gripper.pos."
        )
    return {follower: raw[leader] for follower, leader in LEADER_KEY_FOR_FOLLOWER.items()}


def follower_action_to_leader(action: dict[str, float]) -> dict[str, float]:
    """Follower joint dict -> leader key names (joint_1..joint_6.pos + gripper.pos).

    `gripper.pos` must be included when handing control back, or
    `PiperLeader.send_feedback` silently skips the gripper and the leader's stays
    at whatever opening the operator let go at.
    """
    return {leader: action[follower] for follower, leader in LEADER_KEY_FOR_FOLLOWER.items()}


class PiperChunkEnv:
    """Single real Piper arm behind the :class:`ChunkEnv` protocol."""

    action_names = JOINT_ORDER

    def __init__(
        self,
        robot,
        preprocessor,
        postprocessor,
        task: str,
        action_dim: int = 7,
        max_episode_steps: int = 400,
        control_hz: float = 30.0,
        max_joint_vel_rad_s: float = 6.0,
        max_lead_rad: float = 0.5,
        stream_hz: float = 250.0,
        idle_poll_s: float = 0.001,
        joint_deadband_rad: float = 0.0,
        gripper_deadband: int = 200,
        mode_refresh_interval_s: float = 1.0,
        move_speed_ratio: int = 60,
        reset_pose: list[float] | None = None,
        reset_noise_rad: float = 0.0,
        keys: KeyboardEventListener | None = None,
        intervention: InterventionManager | None = None,
        dry_run: bool = False,
        seed: int = 0,
    ):
        if preprocessor is None or postprocessor is None:
            raise ValueError(
                "PiperChunkEnv needs the stage-1 preprocessor/postprocessor. Re-run "
                "stage 1 so they are saved next to rl_token.pt — running the robot "
                "with mismatched normalisation fails silently and dangerously."
            )
        self.robot = robot
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task = task
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz
        self.reset_pose = reset_pose
        self.reset_noise_rad = reset_noise_rad
        self.keys = keys or KeyboardEventListener()
        self.intervention = intervention or InterventionManager()
        self.dry_run = dry_run
        self.rng = np.random.default_rng(seed)

        self._steps = 0
        self._last_joints: np.ndarray | None = None
        self._last_velocity = np.zeros(len(JOINT_ORDER), dtype=np.float32)
        self._last_obs_t: float | None = None
        self._action_normalizer = ActionNormalizer(preprocessor, action_dim)

        if not self.robot.is_connected:
            self.robot.connect()

        self.streamer = PiperCommandStreamer(
            self.robot.bus,
            max_joint_vel_rad_s=max_joint_vel_rad_s,
            max_lead_rad=max_lead_rad,
            stream_hz=stream_hz,
            idle_poll_s=idle_poll_s,
            joint_deadband_rad=joint_deadband_rad,
            gripper_deadband=gripper_deadband,
            mode_refresh_interval_s=mode_refresh_interval_s,
            move_speed_ratio=move_speed_ratio,
            dry_run=dry_run,
        )

    # ------------------------------------------------------------ observation
    def _read_joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[k] for k in JOINT_ORDER], dtype=np.float32), obs

    def _observe(self) -> dict:
        joints, raw = self._read_joints()
        now = time.perf_counter()
        # Measured, not nominal: a camera grab pushes the real period past
        # `dt`, and dividing by the wrong number scales every velocity the
        # actor and critic see.
        dt = (now - self._last_obs_t) if self._last_obs_t is not None else self.dt
        self._last_obs_t = now
        if self._last_joints is None:
            velocity = np.zeros_like(joints)
        else:
            velocity = (joints - self._last_joints) / max(dt, 1e-6)
        self._last_joints = joints
        self._last_velocity = velocity
        return {"joints": joints, "velocity": velocity, "raw": raw}

    def render_frames(self, obs: dict) -> dict[str, np.ndarray]:
        """Camera frames out of one observation, for the operator view."""
        return {
            key: np.asarray(value)
            for key, value in (obs.get("raw") or {}).items()
            if key not in JOINT_ORDER and getattr(value, "ndim", 0) == 3
        }

    def raw_joint_action(self) -> dict[str, float]:
        """Current measured joints as a robot action dict (for leader sync)."""
        joints = self._last_joints
        if joints is None:
            joints, _ = self._read_joints()
        return dict(zip(JOINT_ORDER, joints.tolist(), strict=True))

    def obs_to_batch(self, obs_list: list[dict], device) -> dict[str, Tensor]:
        """Raw observations -> SmolVLA-preprocessed batch (normalized)."""
        batches = [self._single_obs_to_batch(o, device) for o in obs_list]
        if len(batches) == 1:
            return batches[0]
        keys = batches[0].keys()
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

    def _single_obs_to_batch(self, obs: dict, device) -> dict[str, Tensor]:
        frame: dict = {}
        for cam_key, image in obs["raw"].items():
            if cam_key in JOINT_ORDER:
                continue
            img = image
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 3 and img.shape[-1] == 3:  # HWC uint8 -> CHW float
                img = img.permute(2, 0, 1)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            frame[f"observation.images.{cam_key}"] = img
        frame[OBS_STATE] = torch.from_numpy(obs["joints"]).float()
        frame["task"] = self.task
        batch = self.preprocessor(frame)
        # Velocity is appended after normalisation: the stage-1 dataset stats
        # only cover the joint positions the VLA itself consumes.
        batch["rlt_velocity"] = torch.from_numpy(obs["velocity"]).float().unsqueeze(0).to(device)
        return batch

    # ---------------------------------------------------------------- control
    def _normalized_to_joints(self, action: Tensor) -> np.ndarray:
        """Un-normalize one action to absolute joint targets (radians / metres)."""
        out = self.postprocessor(action.detach().cpu().reshape(1, -1))
        return np.asarray(out[0].detach().cpu(), dtype=np.float32)[: len(JOINT_ORDER)]

    def apply_action(self, action: Tensor) -> tuple[dict, float, bool, bool]:
        """Name one normalized action as the streamer's target.

        This no longer writes to the bus. The streamer owns the command path so
        that the arm keeps moving through a chunk boundary, and it also owns
        rate limiting — in rad/s and anchored on the last command, which a
        per-tick budget anchored on the measurement could not be.
        """
        t0 = time.perf_counter()
        self.streamer.check()
        target = self._normalized_to_joints(action)

        if self.dry_run:
            current = self._last_joints if self._last_joints is not None else self._read_joints()[0]
            logger.info(
                "[dry-run] joint delta (rad): %s",
                np.array2string(target[:6] - current[:6], precision=4),
            )
        self.streamer.set_target(target)

        # Observe first, pad afterwards: the camera grab belongs inside the
        # control period, not on top of it, or the loop silently runs slower
        # than `control_hz` and every finite-difference velocity is wrong.
        obs = self._observe()
        time.sleep(max(self.dt - (time.perf_counter() - t0), 0.0))
        self._steps += 1

        success, failure = self.keys.poll_outcome()
        reward = 1.0 if success else 0.0
        done = success or failure
        truncated = (not done) and self._steps >= self.max_episode_steps
        return obs, reward, done, truncated

    def step(self, action: Tensor) -> tuple[dict, float, bool]:
        obs, reward, done, _ = self.apply_action(action)
        return obs, reward, done

    def run_intervention(self, chunk_len: int) -> InterventionResult | None:
        return self.intervention.run_chunk(chunk_len)

    def intervention_pending(self) -> bool:
        """Is the operator holding the leader right now? (cheap, non-blocking)

        The rollout worker uses this to defer the VLA's flow-matching sampling
        for a chunk whose actions would be thrown away anyway — on hardware
        that sampling is dead time the operator has to wait through at every
        chunk boundary of a takeover.
        """
        return self.intervention.check()

    # ------------------------------------------------------------------ reset
    def reset(self) -> dict:
        target = list(self.reset_pose) if self.reset_pose else list(self.robot.config.home_position)
        if self.reset_noise_rad > 0:
            # Paper: "a slightly randomized set of initial configurations".
            noise = self.rng.uniform(-self.reset_noise_rad, self.reset_noise_rad, size=6)
            target[:6] = [t + n for t, n in zip(target[:6], noise, strict=True)]

        # The streamer and `move_to_joint_smoothly` are both writers on the same
        # bus; stopping it here is what keeps the homing trajectory from
        # fighting the last commanded target. It restarts seeded from wherever
        # homing actually left the arm.
        self.streamer.stop()
        if not self.dry_run:
            self.robot.bus.move_to_joint_smoothly(target)
        self.streamer.start()

        self._steps = 0
        self._last_joints = None
        self._last_obs_t = None
        self.keys.reset_episode_flags()
        self.keys.clear_intervention()
        obs = self._observe()
        self._last_joints = obs["joints"]  # first velocity estimate is zero
        # Re-align the leader with the follower's fresh reset pose, so the first
        # takeover of the episode cannot drag the arm across the workspace.
        self.intervention.on_reset()
        return obs

    def close(self) -> None:
        try:
            self.streamer.stop()
        except Exception:
            logger.exception("stopping the command streamer failed")
        self.intervention.close()
        # `Piper.disconnect()` already runs `bus.safe_disconnect()`; calling it
        # here too would drive the arm to the safe pose twice.
        self.robot.disconnect()

    def raw_action_to_normalized(self, raw: dict[str, float]) -> Tensor:
        """Leader-arm joint dict -> normalized action, matching the VLA space."""
        return self._action_normalizer(
            torch.tensor([raw[k] for k in JOINT_ORDER], dtype=torch.float32)
        )
