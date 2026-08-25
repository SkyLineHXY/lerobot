"""LIBERO simulation as an RLT chunk environment.

This is the pre-hardware validation path: it exercises the *same* code the real
robot runs — normalisation boundary, VLA reference sampling, RL-token
extraction, stride subsampling, async learner — with a sparse binary reward and
a real 7-dim delta end-effector action space, but no arm to damage and no human
in the loop. Getting a success-rate improvement here before touching the Piper
is much cheaper than debugging a chunk-alignment bug on hardware.

Everything shared with the other robosuite task lives in `robosuite_base.py`;
what is left here is LIBERO's suite/task handling and its init-state pool.

Note that `LiberoEnv.step` auto-resets on termination, so on a terminal step the
observation already belongs to the *next* episode. That is safe only because a
terminal transition masks its bootstrap — never use it as a non-terminal x_next.
"""

from __future__ import annotations

from ..teleop.base import InterventionManager
from ..teleop.keys import KeyboardEventListener
from .robosuite_base import RobosuiteChunkEnv, resolve_camera_mapping


class LiberoChunkEnv(RobosuiteChunkEnv):
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
        image_keys: list[str] | None = None,
        seed: int = 0,
        init_state_offset: int = 0,
        n_init_states: int = 0,
        keys: KeyboardEventListener | None = None,
        intervention: InterventionManager | None = None,
    ):
        self.require_processors("LiberoChunkEnv", preprocessor, postprocessor)
        from lerobot.envs.libero import LiberoEnv, _get_suite, _parse_camera_names

        cameras = _parse_camera_names(camera_name)
        img_keys, camera_name_mapping = resolve_camera_mapping(cameras, image_keys, preprocessor)

        self._suite = _get_suite(task_suite_name)
        self.suite_name = task_suite_name
        env = LiberoEnv(
            task_suite=self._suite,
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
        super().__init__(
            env=env,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            expected_image_keys=img_keys,
            action_dim=action_dim,
            max_episode_steps=max_episode_steps,
            seed=seed,
            keys=keys,
            intervention=intervention,
        )

        # LIBERO picks the initial state by a counter, not by the seed
        # (`LiberoEnv.reset`: `_init_states[init_state_id % len]`), so two runs
        # with different seeds still start from the same 50 states in the same
        # order. Training and evaluating both across all 50 would therefore be
        # train-on-test. These two carve the pool into disjoint slices instead:
        # train on `offset=0, n=30`, report on `offset=30, n=20`.
        self.init_state_offset = init_state_offset
        self.n_init_states = n_init_states or len(self._env._init_states)

    @property
    def task(self) -> str:
        """The language prompt handed to the VLA — `task.language`, not `task.name`.

        `LiberoEnv.task` is the bddl identifier
        (`open_the_middle_drawer_of_the_cabinet`); `LiberoEnv.task_description` is
        the instruction the datasets and `add_envs_task` use ("open the middle
        drawer of the cabinet"). They differ only by underscores, so feeding the
        wrong one looks harmless and tokenizes to something else entirely — 17
        tokens instead of 8, none of them shared past the first. The VLA then runs
        on a prompt it never saw in training and quietly scores 0.
        """
        return self._env.task_description

    @property
    def task_description(self) -> str:
        return self._env.task_description

    @property
    def task_name(self) -> str:
        """The bddl identifier. For logging only — never as a policy prompt."""
        return self._env.task

    @property
    def task_id(self) -> int:
        return self._env.task_id

    @property
    def n_tasks(self) -> int:
        return self._suite.n_tasks

    def set_task_id(self, task_id: int) -> bool:
        """Switch to another task of the same suite. Returns whether it changed.

        Rebuilding the simulator takes a second or two, so this is a between-episode
        operation: the caller has to `reset()` afterwards. Note that the RL token,
        the critic and everything already in the replay buffer are conditioned on
        the task language, so mixing tasks within one run is only sound for
        collecting demonstrations — not for a single-task RL curve.
        """
        if not 0 <= task_id < self.n_tasks:
            raise ValueError(f"task_id {task_id} out of range for {self.suite_name} (0..{self.n_tasks - 1})")
        if task_id == self.task_id:
            return False
        self._env.set_task(self._suite, task_id)
        self._episode = 0
        return True

    def reset(self) -> dict:
        self._env.init_state_id = self.init_state_offset + self._episode % self.n_init_states
        obs, _info = self._env.reset(seed=self.seed + self._episode)
        self._on_reset()
        return obs
