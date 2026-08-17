"""数字键切换 LIBERO task_id，以及手柄按键接管/判定成功失败的转发。

两条都不碰仿真器和真实手柄：切任务用只实现 set_task_id 的假 env，手柄事件用只
实现 get_teleop_events 的假设备。真正会咬人的是 `poll_extra` 的单次排空语义
（第二个读取者永远读不到键）和 use_device_events 不能绕过 KeyboardEventListener。
"""

import numpy as np
import pytest

from lerobot.rlt.envs.mock import MockManipEnv
from lerobot.rlt.teleop.device import DeviceIntervention
from lerobot.rlt.teleop.keys import KeyboardEventListener
from lerobot.rlt.train_online import poll_task_switch
from lerobot.teleoperators.utils import TeleopEvents

LIBERO_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]


class FakeKeys(KeyboardEventListener):
    """真的 listener，但按键从测试塞进来而不是从终端读。"""

    def __init__(self, extra=()):
        super().__init__(backend="none")
        self._scripted = list(extra)

    def poll_extra(self):
        out = self._scripted
        self._scripted = []
        return out


class FakeSuiteEnv:
    n_tasks = 10

    def __init__(self, task_id=0):
        self._task_id = task_id
        self.switches = []

    @property
    def task_id(self):
        return self._task_id

    @property
    def task_description(self):
        return f"task {self._task_id}"

    def set_task_id(self, task_id):
        if not 0 <= task_id < self.n_tasks:
            raise ValueError(f"task_id {task_id} out of range for fake (0..{self.n_tasks - 1})")
        if task_id == self._task_id:
            return False
        self._task_id = task_id
        self.switches.append(task_id)
        return True


class FakeGamepad:
    """只实现 get_teleop_events / action_features / get_action 的假手柄。"""

    def __init__(self, events=None):
        # TeleopEvents members are enum instances, not plain str, so these cannot
        # be passed as **kwargs.
        self.events = {
            TeleopEvents.IS_INTERVENTION: False,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }
        self.events.update(events or {})
        self.n_event_polls = 0

    @property
    def action_features(self):
        return {"names": {name: i for i, name in enumerate(LIBERO_NAMES)}}

    def get_action(self):
        # GripperAction: 0 close, 1 stay, 2 open. A resting pad means "stay" —
        # 0.0 would be a close command on every single step.
        return {**dict.fromkeys(LIBERO_NAMES, 0.0), "gripper": 1.0}

    def get_teleop_events(self):
        self.n_event_polls += 1
        return self.events


class FakeEnv(MockManipEnv):
    def __init__(self):
        super().__init__(action_dim=len(LIBERO_NAMES))
        self.reset()

    @property
    def action_names(self):
        return list(LIBERO_NAMES)

    def apply_action(self, action):
        return self._obs(), 0.0, False, False


def make_intervention(gamepad, keys, use_device_events=True):
    return DeviceIntervention(gamepad, FakeEnv(), keys, use_device_events=use_device_events)


# ------------------------------------------------------------- 数字键切任务
def test_digit_key_switches_task():
    env, keys = FakeSuiteEnv(), FakeKeys(["3"])
    assert poll_task_switch(env, keys) == 3
    assert env.task_id == 3


def test_last_digit_wins_within_one_poll():
    """poll_extra 一次排空，同一批里连按两个数字只应切到最后一个。"""
    env, keys = FakeSuiteEnv(), FakeKeys(["2", "7"])
    assert poll_task_switch(env, keys) == 7
    assert env.switches == [7]


def test_same_task_is_not_a_switch():
    env, keys = FakeSuiteEnv(task_id=5), FakeKeys(["5"])
    assert poll_task_switch(env, keys) is None
    assert env.switches == []


def test_out_of_range_digit_is_reported_not_raised(capsys):
    env = FakeSuiteEnv()
    env.n_tasks = 3
    assert poll_task_switch(env, FakeKeys(["9"])) is None
    assert "out of range" in capsys.readouterr().out
    assert env.task_id == 0


def test_non_digit_keys_are_ignored():
    env = FakeSuiteEnv()
    assert poll_task_switch(env, FakeKeys(["q", "up", ""])) is None
    assert env.switches == []


def test_env_without_task_switching_is_left_alone():
    """真机 env 没有 set_task_id，数字键在那边只是一个没人认领的按键。"""

    class Hardware:
        pass

    keys = FakeKeys(["4"])
    assert poll_task_switch(Hardware(), keys) is None
    # 键没被消费掉，其它读取者还能看到
    assert keys.poll_extra() == ["4"]


# --------------------------------------------------- 手柄按键 -> 算子键状态
def test_device_button_requests_takeover():
    keys = KeyboardEventListener(backend="none")
    manager = make_intervention(FakeGamepad({TeleopEvents.IS_INTERVENTION: True}), keys)
    assert manager.check() is True
    # 手柄松开后仍然只看键盘
    manager.teleop.events[TeleopEvents.IS_INTERVENTION] = False
    assert manager.check() is False


def test_keyboard_takeover_still_works_alongside_the_pad():
    keys = KeyboardEventListener(backend="none")
    manager = make_intervention(FakeGamepad(), keys)
    keys.set_intervening(True)
    assert manager.check() is True


def test_device_success_lands_in_the_keyboard_latch():
    """env 每一步只读 keys.poll_outcome()，手柄按键必须走同一个闩。"""
    keys = KeyboardEventListener(backend="none")
    manager = make_intervention(FakeGamepad({TeleopEvents.SUCCESS: True}), keys)
    manager.check()
    assert keys.poll_outcome() == (True, False)


def test_device_terminate_is_a_failure_not_a_success():
    keys = KeyboardEventListener(backend="none")
    manager = make_intervention(FakeGamepad({TeleopEvents.TERMINATE_EPISODE: True}), keys)
    manager.check()
    assert keys.poll_outcome() == (False, True)


def test_device_events_off_never_polls_the_pad():
    keys = KeyboardEventListener(backend="none")
    gamepad = FakeGamepad({TeleopEvents.IS_INTERVENTION: True})
    manager = make_intervention(gamepad, keys, use_device_events=False)
    assert manager.check() is False
    assert gamepad.n_event_polls == 0


def test_device_without_events_degrades_to_keyboard_only():
    class Plain:
        @property
        def action_features(self):
            return {"names": {name: i for i, name in enumerate(LIBERO_NAMES)}}

        def get_action(self):
            return dict.fromkeys(LIBERO_NAMES, 0.0)

    keys = KeyboardEventListener(backend="none")
    manager = DeviceIntervention(Plain(), FakeEnv(), keys, use_device_events=True)
    assert manager.use_device_events is False
    assert manager.check() is False


# ------------------------------------------------- 接管不能把夹爪松开
class GripperEnv(FakeEnv):
    """记录执行过的动作，并像 LiberoChunkEnv 一样公开最后一条。"""

    def __init__(self):
        super().__init__()
        self.last_env_action = None
        self.executed = []

    def apply_action(self, action):
        self.executed.append(action)
        return self._obs(), 0.0, False, False


def test_takeover_holds_the_gripper_the_policy_left_closed():
    """夹爪指令是连续目标：接管时不继承当前状态，第一步就会把抓着的东西松掉。"""
    keys = KeyboardEventListener(backend="none")
    env = GripperEnv()
    env.last_env_action = np.array([0.0] * 6 + [1.0], dtype=np.float32)  # +1 = 闭合
    gamepad = FakeGamepad({TeleopEvents.IS_INTERVENTION: True})
    manager = DeviceIntervention(gamepad, env, keys, use_device_events=True)
    assert manager._gripper == manager.gripper_open_value  # 复位后的初值

    result = manager.run_chunk(3)
    assert result is not None
    grip_idx = LIBERO_NAMES.index("gripper")
    # 手柄一直是 STAY，那么整段都应该保持闭合
    assert all(float(a[grip_idx]) == 1.0 for a in env.executed)


def test_takeover_without_a_published_last_action_keeps_the_latch():
    keys = KeyboardEventListener(backend="none")
    env = GripperEnv()  # last_env_action 仍是 None，例如回合刚复位
    manager = make_intervention(FakeGamepad({TeleopEvents.IS_INTERVENTION: True}), keys)
    manager.env = env
    manager.run_chunk(1)
    assert manager._gripper == manager.gripper_open_value


def test_latch_outcome_does_not_clear_a_pending_flag():
    keys = KeyboardEventListener(backend="none")
    keys.latch_outcome(success=True)
    keys.latch_outcome(failure=True)
    assert keys.poll_outcome() == (True, True)
    assert keys.poll_outcome() == (False, False)


# --------------------------------------------------- 喂给 VLA 的是自然语言指令
def test_task_property_is_the_language_instruction_not_the_bddl_name():
    """两者只差下划线，喂错了不报错，只是策略成功率悄悄变成 0。"""
    from types import SimpleNamespace

    from lerobot.rlt.envs.libero import LiberoChunkEnv

    env = LiberoChunkEnv.__new__(LiberoChunkEnv)
    env._env = SimpleNamespace(
        task="open_the_middle_drawer_of_the_cabinet",
        task_description="open the middle drawer of the cabinet",
    )
    assert env.task == "open the middle drawer of the cabinet"
    assert "_" not in env.task
    # 标识符仍然拿得到，但只能用于日志
    assert env.task_name == "open_the_middle_drawer_of_the_cabinet"


def test_the_prompt_that_reaches_the_preprocessor_is_the_instruction(monkeypatch):
    from types import SimpleNamespace

    from lerobot.rlt.envs import libero as libero_mod

    monkeypatch.setattr(libero_mod, "preprocess_observation", dict)
    monkeypatch.setattr(libero_mod, "_batch_robot_state_", lambda frame: None)

    seen = {}

    env = libero_mod.LiberoChunkEnv.__new__(libero_mod.LiberoChunkEnv)
    env._env = SimpleNamespace(
        task="put_the_bowl_on_the_stove", task_description="put the bowl on the stove"
    )
    env.env_preprocessor = lambda frame: frame
    env.expected_image_keys = []
    env.preprocessor = lambda frame: seen.update(frame) or {}

    env._single_obs_to_batch({}, "cpu")
    assert seen["task"] == "put the bowl on the stove"


# ------------------------------------------------- 仿真里操作员也是奖励来源
class FakeLiberoEnv:
    def __init__(self, is_success=False):
        self.is_success = is_success

    def step(self, action):
        return {"pixels": {}}, 0.0, False, False, {"is_success": self.is_success}


def make_libero_env(*, sim_success=False, keys=None):
    import torch

    from lerobot.rlt.envs.libero import LiberoChunkEnv

    env = LiberoChunkEnv.__new__(LiberoChunkEnv)
    env._env = FakeLiberoEnv(sim_success)
    env._steps = 0
    env.action_dim = 7
    env.max_episode_steps = 100
    env.keys = keys or KeyboardEventListener(backend="none")
    env._normalized_to_env_action = lambda action: np.zeros(7, dtype=np.float32)
    env._action = torch.zeros(7)
    return env


def test_operator_success_key_produces_reward():
    """人接管把任务做成了，LIBERO 的 checker 不认，但那一段确实值 reward=1。"""
    keys = KeyboardEventListener(backend="none")
    env = make_libero_env(keys=keys)
    keys.latch_outcome(success=True)
    _obs, reward, done, _truncated = env.apply_action(env._action)
    assert reward == 1.0
    assert done is True


def test_sim_checker_still_produces_reward_on_its_own():
    env = make_libero_env(sim_success=True)
    _obs, reward, done, _truncated = env.apply_action(env._action)
    assert reward == 1.0
    assert done is True


def test_no_success_anywhere_is_still_zero_reward():
    env = make_libero_env()
    _obs, reward, done, truncated = env.apply_action(env._action)
    assert reward == 0.0
    assert done is False
    assert truncated is False


def test_failure_key_ends_the_episode_without_reward():
    keys = KeyboardEventListener(backend="none")
    env = make_libero_env(keys=keys)
    keys.latch_outcome(failure=True)
    _obs, reward, done, _truncated = env.apply_action(env._action)
    assert reward == 0.0
    assert done is True


def test_outcome_is_consumed_so_it_scores_exactly_one_step():
    keys = KeyboardEventListener(backend="none")
    env = make_libero_env(keys=keys)
    keys.latch_outcome(success=True)
    assert env.apply_action(env._action)[1] == 1.0
    assert env.apply_action(env._action)[1] == 0.0


# ------------------------------------------------------------------ 观测拼接
def test_obs_to_batch_merges_only_stackable_values():
    """处理器返回的是整条 transition，action=None / task=list 不能直接 cat。"""
    import torch

    from lerobot.rlt.envs.libero import LiberoChunkEnv

    frames = [
        {
            "observation.state": torch.zeros(1, 8),
            "action": None,
            "task": [f"t{i}"],
            "next.reward": float(i),
            "info": {},
        }
        for i in range(3)
    ]
    env = LiberoChunkEnv.__new__(LiberoChunkEnv)
    env._single_obs_to_batch = lambda obs, device: obs
    merged = LiberoChunkEnv.obs_to_batch(env, frames, "cpu")
    assert merged["observation.state"].shape == (3, 8)
    assert merged["task"] == ["t0", "t1", "t2"]
    assert merged["action"] is None
    assert merged["next.reward"] == 0.0


def test_obs_to_batch_single_observation_is_passthrough():
    from lerobot.rlt.envs.libero import LiberoChunkEnv

    env = LiberoChunkEnv.__new__(LiberoChunkEnv)
    env._single_obs_to_batch = lambda obs, device: obs
    only = {"observation.state": np.zeros((1, 8))}
    assert LiberoChunkEnv.obs_to_batch(env, [only], "cpu") is only


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
