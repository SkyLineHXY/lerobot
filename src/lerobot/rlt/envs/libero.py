"""LIBERO simulation as an RLT chunk environment.

This is the pre-hardware validation path: it exercises the *same* code the real
robot runs — normalisation boundary, VLA reference sampling, RL-token
extraction, stride subsampling, async learner — with a sparse binary reward and
a real 7-dim delta end-effector action space, but no arm to damage and no human
in the loop. Getting a success-rate improvement here before touching the Piper
is much cheaper than debugging a chunk-alignment bug on hardware.

Observation flow mirrors `lerobot_eval.py`:

    raw LIBERO obs -> preprocess_observation -> LiberoProcessorStep
                   -> stage-1 SmolVLA preprocessor -> model batch

`LiberoProcessorStep` is what turns the nested `robot_state` dict into the flat
8-dim state (eef pos, eef axis-angle, gripper qpos) that the LeRobot LIBERO
datasets use, and rotates the images 180 degrees to match their convention.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from lerobot.envs.utils import preprocess_observation
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX

from ..teleop.base import InterventionManager, InterventionResult
from ..teleop.keys import KeyboardEventListener
from .base import ActionNormalizer

logger = logging.getLogger(__name__)


def _batch_robot_state_(frame: dict) -> None:
    """Add the leading batch dim `preprocess_observation` gives images but not robot_state.

    `lerobot_eval` drives a *vectorised* env, so its raw `robot_state` arrays already
    carry a batch axis and only `pixels`/`agent_pos` need unsqueezing. RLT runs a
    single env, where the nested state stays unbatched and `LiberoProcessorStep`
    rejects it with "expected shape (B, 4)".
    """
    key = f"{OBS_PREFIX}robot_state"
    state = frame.get(key)
    if state is None:
        return
    for group in state.values():
        for name, value in group.items():
            if isinstance(value, Tensor) and value.ndim in (1, 2):
                group[name] = value.unsqueeze(0)


def _stage1_image_keys(preprocessor) -> list[str]:
    """Image keys the stage-1 normalizer was fitted on, in the policy's own order."""
    for step in preprocessor.steps:
        features = getattr(step, "features", None) or {}
        keys = [k for k in features if k.startswith(f"{OBS_IMAGES}.")]
        if keys:
            return keys
    return []


class LiberoChunkEnv:
    """Single LIBERO task behind the :class:`ChunkEnv` protocol."""

    # robosuite OSC_POSE, all channels in [-1, 1]. The names match what the
    # LeRobot teleoperators emit, so `DeviceIntervention` can assemble the
    # vector by name instead of by position.
    action_names = [
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
        "gripper",
    ]

    def __init__(
        self,
        preprocessor,
        postprocessor,
        task_suite_name: str = "libero_10",
        task_id: int = 0,
        action_dim: int = 7,
        max_episode_steps: int = 400,
        control_mode: str = "relative",
        observation_size: int = 256,
        camera_name: str = "agentview_image,robot0_eye_in_hand_image",
        image_keys: list[str] | None = None,
        seed: int = 0,
        keys: KeyboardEventListener | None = None,
        intervention: InterventionManager | None = None,
    ):
        if preprocessor is None or postprocessor is None:
            raise ValueError(
                "LiberoChunkEnv needs the stage-1 preprocessor/postprocessor so the "
                "simulation runs the exact normalisation the RL token was trained with."
            )
        from lerobot.envs.libero import LiberoEnv, _get_suite, _parse_camera_names

        # Three naming conventions have to be reconciled here, and getting it wrong
        # is silent: LiberoEnv calls the wrist camera `image2`, the stage-1
        # processors use the demo dataset's names (`wrist_image` for
        # lerobot/libero_10), and the VLA checkpoint declares its own
        # (`camera1..3` for lerobot/smolvla_libero, trained on lerobot/libero).
        # SmolVLA's vision tower is shared across cameras and keyed only by
        # position in `config.image_features`, so what matters is *order*, not the
        # names — but `prepare_images` skips keys missing from the batch without
        # complaint, so a mismatch quietly drops a whole camera.
        #
        # `image_keys` therefore comes from the policy when the caller has one;
        # otherwise fall back to what stage 1 was fitted on.
        cameras = _parse_camera_names(camera_name)
        keys = list(image_keys) if image_keys else _stage1_image_keys(preprocessor)
        if len(keys) < len(cameras):
            raise ValueError(
                f"The policy declares {len(keys)} image features {keys}, fewer than "
                f"the {len(cameras)} cameras configured: {cameras}."
            )
        if len(keys) > len(cameras):
            # smolvla_libero's config.json lists a camera3 its training data never
            # had; SmolVLA pads or skips the surplus, so this is a note, not an error.
            logger.warning(
                "Policy declares %d image features %s but the env supplies %d cameras; "
                "mapping the first %d in order.",
                len(keys), keys, len(cameras), len(cameras),
            )
            keys = keys[: len(cameras)]
        self.expected_image_keys = keys
        camera_name_mapping = {
            cam: key.removeprefix(f"{OBS_IMAGES}.")
            for cam, key in zip(cameras, keys, strict=True)
        }

        suite = _get_suite(task_suite_name)
        self._env = LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=task_suite_name,
            obs_type="pixels_agent_pos",
            observation_width=observation_size,
            observation_height=observation_size,
            camera_name=camera_name,
            camera_name_mapping=camera_name_mapping,
            control_mode=control_mode,
            episode_length=max_episode_steps,
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.keys = keys or KeyboardEventListener(backend="none")
        self.intervention = intervention or InterventionManager()
        self._action_normalizer = ActionNormalizer(preprocessor, action_dim)
        # Same step the eval script inserts for LIBERO; without it the policy
        # receives a nested robot_state dict instead of `observation.state`.
        self.env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])

        self._steps = 0
        self._episode = 0

    @property
    def task(self) -> str:
        return self._env.task

    # ------------------------------------------------------------ observation
    def obs_to_batch(self, obs_list: list[dict], device) -> dict[str, Tensor]:
        batches = [self._single_obs_to_batch(o, device) for o in obs_list]
        if len(batches) == 1:
            return batches[0]
        keys = batches[0].keys()
        return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}

    def _single_obs_to_batch(self, obs: dict, device) -> dict[str, Tensor]:
        frame = preprocess_observation(obs)
        _batch_robot_state_(frame)
        frame = self.env_preprocessor(frame)
        frame["task"] = self.task
        batch = self.preprocessor(frame)
        missing = [k for k in self.expected_image_keys if k not in batch]
        if missing:
            raise RuntimeError(
                f"Observation is missing image keys {missing} the stage-1 processors "
                f"expect; got {sorted(k for k in batch if k.startswith(OBS_IMAGES))}."
            )
        # Joint velocity is available in sim, but the flat LIBERO state already
        # ends at the gripper; keep proprio identical to what stage 1 saw.
        return batch

    # ---------------------------------------------------------------- control
    def _normalized_to_env_action(self, action: Tensor) -> np.ndarray:
        out = self.postprocessor(action.detach().cpu().reshape(1, -1))
        arr = np.asarray(out[0].detach().cpu(), dtype=np.float32)
        return arr[: self.action_dim]

    def env_action_to_normalized(self, raw) -> Tensor:
        """Teleop command in LIBERO's [-1, 1] delta-EE box -> normalized action."""
        return self._action_normalizer(raw)

    def apply_action(self, action: Tensor) -> tuple[dict, float, bool, bool]:
        """Execute one normalized action; returns (obs, reward, done, truncated)."""
        env_action = self._normalized_to_env_action(action)
        obs, _reward, terminated, _truncated, info = self._env.step(env_action)
        self._steps += 1

        success = bool(info.get("is_success", False))
        # The failure key is the operator's only way to end a doomed episode:
        # the simulator itself never reports failure, so without it a bad
        # rollout burns the full step limit before the buffer sees a terminal.
        _key_success, key_failure = self.keys.poll_outcome()
        reward = 1.0 if success else 0.0
        done = terminated or success or key_failure
        truncated = (not done) and self._steps >= self.max_episode_steps
        # LiberoEnv.step auto-resets on termination, so on a terminal step `obs`
        # already belongs to the *next* episode. Safe only because a terminal
        # transition masks its bootstrap — never use it as a non-terminal x_next.
        return obs, reward, done, truncated

    def step(self, action: Tensor) -> tuple[dict, float, bool]:
        obs, reward, done, _ = self.apply_action(action)
        return obs, reward, done

    def run_intervention(self, chunk_len: int) -> InterventionResult | None:
        return self.intervention.run_chunk(chunk_len)

    def intervention_pending(self) -> bool:
        return self.intervention.check()

    def reset(self) -> dict:
        obs, _info = self._env.reset(seed=self.seed + self._episode)
        self._steps = 0
        self._episode += 1
        self.keys.reset_episode_flags()
        self.keys.clear_intervention()
        self.intervention.on_reset()
        return obs

    def close(self) -> None:
        try:
            self.intervention.close()
        finally:
            self._env.close()
