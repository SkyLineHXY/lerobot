"""Environments the RLT rollout worker can drive, and the factory that builds them.

Four back ends behind one protocol (`base.ChunkEnv`):

* `mock` — plumbing only, no normalisation boundary. Never a validation signal.
* `libero` — simulation; the pre-hardware end-to-end validation path.
* `insertion` — simulation; short-horizon fine manipulation with a sparse reward.
* `piper` — the real arm.

The three real back ends are imported lazily: each drags in a heavy dependency
(robosuite, the CAN stack) that the others have no reason to load.
"""

from __future__ import annotations

import logging

from lerobot.policies.rlt import (
    InsertionEnvConfig,
    LiberoEnvConfig,
    MockEnvConfig,
    PiperEnvConfig,
)

from .base import ActionNormalizer, ChunkEnv, find_action_normalizer
from .mock import MockManipEnv

logger = logging.getLogger(__name__)

__all__ = [
    "ActionNormalizer",
    "ChunkEnv",
    "MockManipEnv",
    "find_action_normalizer",
    "make_chunk_env",
]


def piper_joint_speed_limit(env_cfg: PiperEnvConfig) -> float:
    """The follower's speed limit in rad/s, converting the deprecated key.

    `max_joint_step_rad` was a per-tick radian budget, so its physical meaning
    moved with the control rate — the same 0.05 meant 1.5 rad/s at 30 Hz and
    1.25 at 25 Hz, and would mean over 9 rad/s now that commands run at the
    leader's feedback rate. Convert once, warn, and keep running.
    """
    legacy = env_cfg.max_joint_step_rad
    if legacy is None:
        return env_cfg.max_joint_vel_rad_s
    converted = float(legacy) * env_cfg.control_hz
    logger.warning(
        "env.max_joint_step_rad is deprecated (radians per tick; its meaning "
        "drifts with the control rate). Converted %.3f rad x %.0f Hz to "
        "env.max_joint_vel_rad_s=%.2f rad/s for this run — please switch keys.",
        legacy,
        env_cfg.control_hz,
        converted,
    )
    return converted


def _with_sim_teleop(env, teleop_cfg, keys):
    if teleop_cfg is not None:
        from ..teleop.device import build_sim_intervention

        env.intervention = build_sim_intervention(teleop_cfg, env, keys)
    return env


def make_chunk_env(cfg, keys, policy=None):
    """Build the env named by `cfg.env`, wired to the operator keyboard."""
    env_cfg, ac = cfg.env, cfg.rl.ac

    if isinstance(env_cfg, MockEnvConfig):
        return _with_sim_teleop(
            MockManipEnv(
                action_dim=ac.action_dim,
                max_episode_steps=env_cfg.max_episode_steps,
                image_size=env_cfg.image_size,
                success_eps=env_cfg.success_eps,
                prompt=env_cfg.prompt,
                seed=cfg.seed,
                keys=keys,
            ),
            env_cfg.teleop,
            keys,
        )

    # Everything below runs the policy for real, so it must use exactly the
    # normalisation stage 1 was trained with.
    from lerobot.policies.rlt import load_stage1_processors

    preprocessor, postprocessor = load_stage1_processors(cfg.rl_token, device=cfg.device)

    if isinstance(env_cfg, LiberoEnvConfig):
        from .libero import LiberoChunkEnv

        return _with_sim_teleop(
            LiberoChunkEnv(
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task_suite_name=env_cfg.suite,
                task_id=env_cfg.task_id,
                action_dim=ac.action_dim,
                max_episode_steps=env_cfg.max_episode_steps,
                control_mode=env_cfg.control_mode,
                observation_size=env_cfg.observation_size,
                camera_name=env_cfg.camera_name,
                # Camera keys must follow the *policy's* image features, not the
                # dataset's; see LiberoChunkEnv.
                image_keys=list(policy.config.image_features) if policy is not None else None,
                seed=cfg.seed,
                init_state_offset=env_cfg.init_state_offset,
                n_init_states=env_cfg.n_init_states,
                keys=keys,
            ),
            env_cfg.teleop,
            keys,
        )

    if isinstance(env_cfg, InsertionEnvConfig):
        from .insertion import InsertionChunkEnv

        return _with_sim_teleop(
            InsertionChunkEnv(
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                prompt=env_cfg.prompt,
                task_stage=env_cfg.task_stage,
                action_dim=ac.action_dim,
                max_episode_steps=env_cfg.max_episode_steps,
                control_mode=env_cfg.control_mode,
                observation_size=env_cfg.observation_size,
                camera_name=env_cfg.camera_name,
                control_freq=env_cfg.control_freq,
                image_keys=list(policy.config.image_features) if policy is not None else None,
                seed=cfg.seed,
                init_state_offset=env_cfg.init_state_offset,
                n_init_states=env_cfg.n_init_states,
                pool_size=env_cfg.pool_size,
                pool_seed=env_cfg.pool_seed,
                task_kwargs=env_cfg.task_kwargs,
                keys=keys,
            ),
            env_cfg.teleop,
            keys,
        )

    if not isinstance(env_cfg, PiperEnvConfig):
        raise NotImplementedError(f"Unsupported env type {env_cfg.type!r}.")

    from lerobot.robots.piper.config_piper import PIPERConfig
    from lerobot.robots.piper.piper import Piper
    from lerobot.utils.piper_sdk import apply_piper_can_filters

    from ..teleop.piper_leader import PiperLeaderIntervention
    from .piper import PiperChunkEnv, build_piper_cameras

    robot_cfg = PIPERConfig(can_port=env_cfg.can_port)
    if env_cfg.cameras:
        robot_cfg.cameras = build_piper_cameras(env_cfg.cameras, env_cfg.control_hz)
    if env_cfg.max_joint_step_rad is not None:
        robot_cfg.max_joint_step_rad = env_cfg.max_joint_step_rad
    robot = Piper(robot_cfg)

    env = PiperChunkEnv(
        robot=robot,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task=env_cfg.task,
        action_dim=ac.action_dim,
        max_episode_steps=env_cfg.max_episode_steps,
        control_hz=env_cfg.control_hz,
        max_joint_vel_rad_s=piper_joint_speed_limit(env_cfg),
        max_lead_rad=env_cfg.max_lead_rad,
        stream_hz=env_cfg.stream_hz,
        idle_poll_s=env_cfg.idle_poll_s,
        joint_deadband_rad=env_cfg.joint_deadband_rad,
        gripper_deadband=env_cfg.gripper_deadband,
        mode_refresh_interval_s=env_cfg.mode_refresh_interval_s,
        move_speed_ratio=env_cfg.move_speed_ratio,
        reset_pose=env_cfg.reset_pose,
        reset_noise_rad=env_cfg.reset_noise_rad,
        keys=keys,
        dry_run=env_cfg.dry_run,
        seed=cfg.seed,
    )
    apply_piper_can_filters(robot.bus.piper, "follower")

    if env_cfg.teleop is not None:
        from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

        leader = PiperLeader(PiperLeaderConfig(port=env_cfg.teleop.port, id=env_cfg.teleop.id))
        leader.connect()
        # piper_sdk parses every frame in two Python threads per interface, all
        # of it holding the GIL. Dropping the IDs nobody reads roughly halves
        # that load, which the streamer and the learner both feel.
        apply_piper_can_filters(leader.arm, "leader")
        env.intervention = PiperLeaderIntervention(
            leader,
            env,
            keys,
            use_calibrated_offsets=env_cfg.teleop.use_calibrated_offsets,
            max_takeover_delta_rad=env_cfg.teleop.max_takeover_delta_rad,
            align_settle_s=env_cfg.teleop.align_settle_s,
            action_source=env_cfg.teleop.action_source,
        )
        # The leader entered gravity compensation on connect; park it on the
        # follower's pose so the first takeover has a defined starting point.
        env.intervention.align()
    return env
