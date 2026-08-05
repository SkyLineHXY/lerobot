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

from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

from .intervention import InterventionManager, InterventionResult, KeyboardEventListener

logger = logging.getLogger(__name__)

JOINT_ORDER = [f"joint_{i}.pos" for i in range(1, 8)]


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
        """Scale the whole step so no joint exceeds the per-step limit.

        Scaling uniformly (rather than clipping each joint) preserves the
        direction of the commanded motion; per-joint clipping would silently
        bend the trajectory.
        """
        delta = target[:6] - current[:6]
        peak = float(np.abs(delta).max()) if delta.size else 0.0
        if peak > self.max_joint_step_rad > 0:
            delta = delta * (self.max_joint_step_rad / peak)
        limited = target.copy()
        limited[:6] = current[:6] + delta
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
        return obs

    def close(self) -> None:
        self.intervention.close()
        try:
            if not self.dry_run:
                self.robot.bus.safe_disconnect()
        finally:
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
