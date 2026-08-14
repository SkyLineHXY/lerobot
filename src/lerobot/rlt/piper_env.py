"""Real Piper arm as an RLT chunk environment (paper Sec. V).

This closes the two gaps that make `MockManipEnv` unusable on hardware:

1. **The normalisation boundary.** The controller works entirely in SmolVLA's
   normalized action space; observations must go through the same
   preprocessor, and actions must come back through the same postprocessor,
   that stage 1 was trained with. Loading both from the stage-1 output
   directory is what keeps z_rl and the reference chunks consistent.
2. **Motion limits.** Piper takes absolute joint targets. Every commanded step
   is rate-limited against the arm's measured position, and the whole chunk is
   scaled down rather than clipped per joint, so limiting never distorts the
   direction of a motion.

Proprioception is position *and* velocity (paper App. B); velocity is obtained
by finite differences of the measured joints, since the Piper bus exposes
positions only.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch import Tensor

from lerobot.rlt.intervention import InterventionManager, InterventionResult, KeyboardEventListener
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

JOINT_ORDER = [f"joint_{i}.pos" for i in range(1, 8)]

# 主臂（PiperLeader）与跟随臂（Piper）的键名约定不同：主臂沿用 PIPER_ACTION_KEYS，
# 把夹爪叫 `gripper.pos`；跟随臂把夹爪当作第 7 个关节 `joint_7.pos`。两边的物理量
# 完全一致（关节弧度 / 夹爪开度米），只是名字不同，所以只需要改名，不需要换算。
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

        # Clamp the emitted command, not the anchor: clamping the anchor would still
        # let one full step be added on top, so the lead would settle at
        # max_lead_rad + max_vel_rad_s * dt instead of the bound we asked for.
        if self.max_lead_rad > 0:
            lead = limited[:6] - measured[:6]
            peak_lead = float(np.abs(lead).max()) if lead.size else 0.0
            if peak_lead > self.max_lead_rad:
                limited = limited.copy()
                limited[:6] = measured[:6] + lead * (self.max_lead_rad / peak_lead)

        self._prev_cmd = limited.astype(np.float32, copy=True)
        return limited, saturated


def jitter_rms(traj: np.ndarray, dt: float) -> float:
    """轨迹的抖动度量：二阶差分的 RMS，单位 rad/s²。

    一阶差分是速度，正常运动本来就有，拿它衡量抖动会把"动得快"误判成"抖"。
    二阶差分（加速度）才反映方向反复翻转的高频成分，也就是手上感觉到的抖。

    分别对"下发指令"和"跟随臂实测"各算一次，两者的关系能直接定位责任方：
    实测远大于指令 -> 跟随臂自己在振（该调 kp/kd 或换控制模式）；
    两者相当且都高 -> 输入本身就抖（主臂那端的问题）。
    """
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
    """主臂动作 dict -> 跟随臂键名（joint_1..joint_7.pos）。"""
    missing = [k for k in LEADER_KEY_FOR_FOLLOWER.values() if k not in raw]
    if missing:
        raise KeyError(
            f"主臂动作缺少键 {missing}；拿到的是 {sorted(raw)}。"
            "PiperLeader 应返回 joint_1..joint_6.pos + gripper.pos。"
        )
    return {follower: raw[leader] for follower, leader in LEADER_KEY_FOR_FOLLOWER.items()}


def follower_action_to_leader(action: dict[str, float]) -> dict[str, float]:
    """跟随臂关节 dict -> 主臂键名（joint_1..joint_6.pos + gripper.pos）。

    交接回位时必须带上 `gripper.pos`，否则 `PiperLeader.send_feedback` 会静默跳过
    夹爪，主臂夹爪停在人松手时的开度上。
    """
    return {leader: action[follower] for follower, leader in LEADER_KEY_FOR_FOLLOWER.items()}


class PiperChunkEnv:
    """Single real Piper arm behind the :class:`ChunkEnv` protocol."""

    def __init__(
        self,
        robot,
        preprocessor,
        postprocessor,
        task: str,
        action_dim: int = 7,
        max_episode_steps: int = 400,
        control_hz: float = 30.0,
        max_joint_step_rad: float = 0.05,
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
        self.max_joint_step_rad = max_joint_step_rad
        self.reset_pose = reset_pose
        self.reset_noise_rad = reset_noise_rad
        self.keys = keys or KeyboardEventListener()
        self.intervention = intervention or InterventionManager()
        self.dry_run = dry_run
        self.rng = np.random.default_rng(seed)

        self._steps = 0
        self._last_joints: np.ndarray | None = None
        self._last_velocity = np.zeros(len(JOINT_ORDER), dtype=np.float32)
        self._action_normalizer = None

        if not self.robot.is_connected:
            self.robot.connect()

    # ------------------------------------------------------------ observation
    def _read_joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[k] for k in JOINT_ORDER], dtype=np.float32), obs

    def _observe(self) -> dict:
        joints, raw = self._read_joints()
        if self._last_joints is None:
            velocity = np.zeros_like(joints)
        else:
            velocity = (joints - self._last_joints) / self.dt
        self._last_joints = joints
        self._last_velocity = velocity
        return {"joints": joints, "velocity": velocity, "raw": raw}

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

    def _rate_limit(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        limited, _saturated = rate_limit_joints(target, current, self.max_joint_step_rad)
        return limited

    def apply_action(self, action: Tensor) -> tuple[dict, float, bool, bool]:
        """Execute one normalized action; returns (obs, reward, done, truncated)."""
        t0 = time.perf_counter()
        target = self._normalized_to_joints(action)
        current = self._last_joints if self._last_joints is not None else self._read_joints()[0]
        target = self._rate_limit(target, current)

        if self.dry_run:
            logger.info(
                "[dry-run] joint delta (rad): %s",
                np.array2string(target[:6] - current[:6], precision=4),
            )
        else:
            self.robot.send_action(dict(zip(JOINT_ORDER, target.tolist(), strict=True)))

        precise_sleep(max(self.dt - (time.perf_counter() - t0), 0.0))
        obs = self._observe()
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

        The rollout worker uses this to skip the VLA's flow-matching sampling
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

        if not self.dry_run:
            self.robot.bus.move_to_joint_smoothly(target)

        self._steps = 0
        self._last_joints = None
        self.keys.reset_episode_flags()
        self.keys.clear_intervention()
        obs = self._observe()
        self._last_joints = obs["joints"]  # first velocity estimate is zero
        # Re-align the leader with the follower's fresh reset pose, so the first
        # takeover of the episode cannot drag the arm across the workspace.
        self.intervention.on_reset()
        return obs

    def close(self) -> None:
        self.intervention.close()
        # `Piper.disconnect()` already runs `bus.safe_disconnect()`; calling it
        # here too would drive the arm to the safe pose twice.
        self.robot.disconnect()

    # ------------------------------------------------------ teleop conversion
    def raw_action_to_normalized(self, raw: dict[str, float]) -> Tensor:
        """Leader-arm joint dict -> normalized action, matching the VLA space.

        Reuses the stage-1 normalizer step rather than re-deriving the
        transform, so mean/std vs min/max modes cannot drift apart between the
        policy's actions and the human's.
        """
        joints = torch.tensor([raw[k] for k in JOINT_ORDER], dtype=torch.float32)
        if self._action_normalizer is None:
            self._action_normalizer = _find_action_normalizer(self.preprocessor)
        if self._action_normalizer is None:
            raise RuntimeError(
                "No action normalizer found in the stage-1 preprocessor; human "
                "interventions cannot be expressed in the policy's action space."
            )
        norm = self._action_normalizer._normalize_action(joints, inverse=False)
        return norm[: self.action_dim]


def _find_action_normalizer(preprocessor):
    """Return the pipeline step that knows how to normalize an ACTION tensor."""
    for step in getattr(preprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if stats and ACTION in stats and hasattr(step, "_normalize_action"):
            return step
    return None
