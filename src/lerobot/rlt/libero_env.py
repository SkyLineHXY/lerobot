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
from lerobot.utils.constants import OBS_IMAGES

from .intervention import InterventionResult

logger = logging.getLogger(__name__)


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
        seed: int = 0,
    ):
        if preprocessor is None or postprocessor is None:
            raise ValueError(
                "LiberoChunkEnv needs the stage-1 preprocessor/postprocessor so the "
                "simulation runs the exact normalisation the RL token was trained with."
            )
        from lerobot.envs.libero import LiberoEnv, _get_suite, _parse_camera_names

        # LiberoEnv's default mapping names the wrist camera `image2`, but the
        # stage-1 processors were fitted on whatever the demo dataset called it
        # (`wrist_image` for lerobot/libero_10). SmolVLA.prepare_images silently
        # *skips* image keys missing from the batch, so a name mismatch costs a
        # whole camera with no error at all — pin the mapping to stage 1 instead.
        cameras = _parse_camera_names(camera_name)
        self.expected_image_keys = _stage1_image_keys(preprocessor)
        if len(self.expected_image_keys) != len(cameras):
            raise ValueError(
                f"Stage 1 was trained with {len(self.expected_image_keys)} cameras "
                f"{self.expected_image_keys}, but the env is configured with "
                f"{len(cameras)}: {cameras}. The camera set must match the VLA's."
            )
        camera_name_mapping = {
            cam: key.removeprefix(f"{OBS_IMAGES}.")
            for cam, key in zip(cameras, self.expected_image_keys, strict=True)
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

    def step(self, action: Tensor) -> tuple[dict, float, bool]:
        env_action = self._normalized_to_env_action(action)
        obs, _reward, terminated, _truncated, info = self._env.step(env_action)
        self._steps += 1

        success = bool(info.get("is_success", False))
        reward = 1.0 if success else 0.0
        # LiberoEnv.step auto-resets on termination, so on a terminal step `obs`
        # already belongs to the *next* episode. Safe only because a terminal
        # transition masks its bootstrap — never use it as a non-terminal x_next.
        return obs, reward, terminated or success

    def run_intervention(self, chunk_len: int) -> InterventionResult | None:
        return None  # no human in the loop in simulation

    def intervention_pending(self) -> bool:
        return False

    # ------------------------------------------------------------------ reset
    def reset(self) -> dict:
        obs, _info = self._env.reset(seed=self.seed + self._episode)
        self._steps = 0
        self._episode += 1
        return obs

    def close(self) -> None:
        self._env.close()
