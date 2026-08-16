"""DeviceIntervention：把手持设备的命名通道装配成 env 的动作向量。

这里的每条用例都对应一个真实会咬人的地方：按名字装配（而不是按下标）、
夹爪锁存、设备缺旋转通道时补 0 而不是错位、以及探测接管状态时不能有副作用。
不碰真实设备，用只实现 action_features/get_action 的假 teleop。
"""

import numpy as np
import torch

from lerobot.rlt.envs.mock import MockManipEnv
from lerobot.rlt.teleop.device import DeviceIntervention
from lerobot.rlt.teleop.keys import KeyboardEventListener

LIBERO_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]


class FakeTeleop:
    """只按 Teleoperator 契约暴露 action_features 和 get_action 的假设备。"""

    def __init__(self, names, script):
        self._names = {name: i for i, name in enumerate(names)}
        self._script = list(script)
        self.n_calls = 0

    @property
    def action_features(self):
        return {"dtype": "float32", "shape": (len(self._names),), "names": self._names}

    def get_action(self):
        self.n_calls += 1
        # 脚本用尽后保持最后一帧，模拟操作员松手不动
        return self._script[min(self.n_calls - 1, len(self._script) - 1)]


class FakeEnv(MockManipEnv):
    """记录每一步实际收到的动作向量的 7 维 env。"""

    def __init__(self, names=LIBERO_NAMES, **kw):
        super().__init__(action_dim=len(names), **kw)
        self._names = list(names)
        self.applied = []
        self.reset()  # 让 _obs() 有状态可返回

    @property
    def action_names(self):
        return self._names

    def apply_action(self, action):
        self.applied.append(np.asarray(action, dtype=np.float32).copy())
        return self._obs(), 0.0, False, False


class EngagedKeys(KeyboardEventListener):
    """始终处于接管状态的算子键监听器。"""

    def __init__(self):
        super().__init__(backend="none")
        self._intervene = True


def _intervention(teleop, env, **kw):
    return DeviceIntervention(teleop, env, EngagedKeys(), **kw)


def test_channels_are_assembled_by_name_not_by_position():
    # 设备只出平移三轴 + 夹爪（键盘/手柄的默认形态），env 要 7 维。
    teleop = FakeTeleop(
        ["delta_x", "delta_y", "delta_z", "gripper"],
        [{"delta_x": 1.0, "delta_y": -1.0, "delta_z": 0.5, "gripper": 1.0}],
    )
    env = FakeEnv()
    result = _intervention(teleop, env, position_scale=1.0).run_chunk(1)

    assert result is not None
    step = env.applied[0]
    assert step[:3].tolist() == [1.0, -1.0, 0.5]
    # 关键：缺失的三个旋转通道必须补 0，而不是把 gripper 挤到 delta_roll 上
    assert step[3:6].tolist() == [0.0, 0.0, 0.0]
    assert step[6] == -1.0  # gripper "stay" -> 保持初始的张开


def test_rotation_channels_are_scaled_independently():
    teleop = FakeTeleop(
        LIBERO_NAMES,
        [dict.fromkeys(LIBERO_NAMES, 1.0)],
    )
    env = FakeEnv()
    _intervention(teleop, env, position_scale=0.3, rotation_scale=0.2).run_chunk(1)

    step = env.applied[0]
    assert np.allclose(step[:3], 0.3)
    assert np.allclose(step[3:6], 0.2)


def test_gripper_latches_across_stay_commands():
    """stay 必须重复上一次命令：发 0 会让夹爪在抓取途中自己松开。"""
    teleop = FakeTeleop(
        LIBERO_NAMES,
        [
            {"gripper": 0.0},  # close
            {"gripper": 1.0},  # stay
            {"gripper": 1.0},  # stay
            {"gripper": 2.0},  # open
            {"gripper": 1.0},  # stay
        ],
    )
    env = FakeEnv()
    _intervention(teleop, env).run_chunk(5)

    grip = [step[6] for step in env.applied]
    assert grip == [1.0, 1.0, 1.0, -1.0, -1.0]


def test_chunk_is_padded_with_the_last_human_action():
    teleop = FakeTeleop(LIBERO_NAMES, [dict.fromkeys(LIBERO_NAMES, 0.0)])
    env = FakeEnv()
    intervention = _intervention(teleop, env)

    class ReleaseAfterTwo(EngagedKeys):
        def __init__(self):
            super().__init__()
            self.n = 0

        @property
        def intervening(self):
            # run_chunk 进入时查一次，之后每执行一步再查一次
            self.n += 1
            return self.n <= 3

    intervention.keys = ReleaseAfterTwo()
    result = intervention.run_chunk(6)

    assert result.n_steps == 3
    assert result.action_chunk.shape == (6, len(LIBERO_NAMES))
    # 未执行的尾部重复最后一条人工动作，形状固定但 n_steps 记录真实长度
    assert torch.equal(result.action_chunk[5], result.action_chunk[2])
    assert result.rewards.shape == (6,)


def test_returns_none_when_the_operator_is_not_engaged():
    teleop = FakeTeleop(LIBERO_NAMES, [dict.fromkeys(LIBERO_NAMES, 1.0)])
    env = FakeEnv()
    intervention = DeviceIntervention(
        teleop, env, KeyboardEventListener(backend="none")
    )

    assert intervention.run_chunk(4) is None
    assert env.applied == []
    assert teleop.n_calls == 0  # 没接管时不该去读设备


def test_reset_reopens_the_gripper():
    teleop = FakeTeleop(LIBERO_NAMES, [{"gripper": 0.0}])
    env = FakeEnv()
    intervention = _intervention(teleop, env)
    intervention.run_chunk(1)
    assert env.applied[0][6] == 1.0

    intervention.on_reset()
    intervention.teleop = FakeTeleop(LIBERO_NAMES, [{"gripper": 1.0}])
    intervention.run_chunk(1)
    assert env.applied[1][6] == -1.0
