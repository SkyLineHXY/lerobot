"""Environments the RLT rollout worker can drive, and the factory that builds them.

Three back ends behind one protocol (`base.ChunkEnv`):

* `mock` — plumbing only, no normalisation boundary. Never a validation signal.
* `libero` — simulation; the pre-hardware end-to-end validation path.
* `piper` — the real arm.

LIBERO and Piper are imported lazily: each drags in a heavy dependency (robosuite,
the CAN stack) that the other two runs have no reason to load.
"""

from __future__ import annotations

from lerobot.policies.rlt import LiberoEnvConfig, MockEnvConfig, PiperEnvConfig

from .base import ActionNormalizer, ChunkEnv, find_action_normalizer
from .mock import MockManipEnv

__all__ = [
    "ActionNormalizer",
    "ChunkEnv",
    "MockManipEnv",
    "find_action_normalizer",
    "make_chunk_env",
]


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
                keys=keys,
            ),
            env_cfg.teleop,
            keys,
        )

    if not isinstance(env_cfg, PiperEnvConfig):
        raise NotImplementedError(f"Unsupported env type {env_cfg.type!r}.")

    from lerobot.robots.piper.config_piper import PIPERConfig
    from lerobot.robots.piper.piper import Piper

    from ..teleop.piper_leader import PiperLeaderIntervention
    from .piper import PiperChunkEnv, build_piper_cameras

    robot_cfg = PIPERConfig(can_port=env_cfg.can_port)
    if env_cfg.cameras:
        robot_cfg.cameras = build_piper_cameras(env_cfg.cameras, env_cfg.control_hz)
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
        max_joint_step_rad=env_cfg.max_joint_step_rad,
        reset_pose=env_cfg.reset_pose,
        reset_noise_rad=env_cfg.reset_noise_rad,
        keys=keys,
        dry_run=env_cfg.dry_run,
        seed=cfg.seed,
    )

    if env_cfg.teleop is not None:
        from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

        leader = PiperLeader(PiperLeaderConfig(port=env_cfg.teleop.port, id=env_cfg.teleop.id))
        leader.connect()
        env.intervention = PiperLeaderIntervention(
            leader,
            env,
            keys,
            use_calibrated_offsets=env_cfg.teleop.use_calibrated_offsets,
            max_takeover_delta_rad=env_cfg.teleop.max_takeover_delta_rad,
        )
        # The leader entered gravity compensation on connect; park it on the
        # follower's pose so the first takeover has a defined starting point.
        env.intervention.align()
    return env
