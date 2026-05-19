# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SpaceMouse 干预包装器与 gym_hil 仿真环境注册。

在不修改 gym_hil 站点包的前提下，为 Franka 仿真 env 提供 SpaceMouse 支持：
注册 `gym_hil/PandaPickCubeSpaceMouse-v0` 和 `gym_hil/PandaArrangeBoxesSpaceMouse-v0`，
对外接口与 Gamepad / Keyboard 变体完全一致。
"""

import logging

import gymnasium as gym
import numpy as np

from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
from lerobot.teleoperators.spacemouse.teleop_spacemouse import SpaceMouseTeleop
from lerobot.teleoperators.utils import TeleopEvents

# SpaceMouse 运动干预的归一化幅度死区阈值（约为 ee_step_size 的 1%）。
# 低于此值视为噪声/静止，is_intervention=False，policy 自主执行。
_INTERVENTION_THRESHOLD = 0.01


class SpaceMouseInterventionWrapper(gym.Wrapper):
    """用 SpaceMouse 遥操作器替代 gym_hil 内置 Gamepad/Keyboard 的干预包装器。

    与 InputsControlWrapper 保持相同 info 字段契约：
        info["is_intervention"]  —— 当前帧是否处于人工干预状态
        info["teleop_action"]    —— 发给底层 env 的实际动作
        info["rerecord_episode"] —— 是否需要重录当前 episode
        info["next.success"]     —— episode 终止时的成功标志（仅写入一次）

    动作空间保持 4-D [dx_norm, dy_norm, dz_norm, gripper]（与 Gamepad 变体相同），
    其中 xyz 已按 ee_step_size 归一化至 [-1, 1]，gripper 在 [0, 2]（0=合/1=保持/2=开）。
    """

    def __init__(
        self,
        env: gym.Env,
        ee_step_size: dict,
        spacemouse_config: SpaceMouseTeleopConfig | None = None,
    ):
        super().__init__(env)
        if spacemouse_config is None:
            # 仿真默认按键布局：左键(0)=夹爪合，右键(1)=夹爪开。
            # terminate_button/intervention_button 设为越界索引(99)，_safe_button 越界返回 0，
            # 边沿检测永不成立，避免与夹爪按键冲突导致误终止 episode。
            spacemouse_config = SpaceMouseTeleopConfig(
                gripper_open_button=1,
                gripper_close_button=0,
                intervention_button=99,
                terminate_button=99,
            )

        self._teleop = SpaceMouseTeleop(spacemouse_config)
        self._teleop.connect()

        # 用于将 SpaceMouse 输出的米制增量归一化到 [-1, 1]
        self._ee_step_size = np.array(
            [ee_step_size["x"], ee_step_size["y"], ee_step_size["z"]], dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["is_intervention"] = False
        return obs, info

    def step(self, action):
        action = np.array(action, dtype=np.float32)

        # get_teleop_events 须先于 get_action 调用：
        # get_action 会更新 _prev_buttons，若先调用则同帧内边沿检测全部失效
        events = self._teleop.get_teleop_events()
        teleop_dict = self._teleop.get_action()

        dx_norm = np.clip(teleop_dict.get("delta_x", 0.0) / self._ee_step_size[0], -1.0, 1.0)
        dy_norm = np.clip(teleop_dict.get("delta_y", 0.0) / self._ee_step_size[1], -1.0, 1.0)
        dz_norm = np.clip(teleop_dict.get("delta_z", 0.0) / self._ee_step_size[2], -1.0, 1.0)
        # gripper: 0=合 / 1=保持 / 2=开，EEActionWrapper 内部转为 [-1,1]
        gripper = float(teleop_dict.get("gripper", 1.0))

        # 以 SpaceMouse 是否有实质性输入判断是否处于干预状态：
        # 归一化运动幅度超过死区阈值，或夹爪处于非保持状态（按键被按下）。
        # HIL-RL 中 SpaceMouse 静止时 is_intervention=False，policy 自主执行；
        # 主动操作或按下夹爪键时 is_intervention=True，SpaceMouse 接管动作。
        is_intervention = (
            abs(dx_norm) > _INTERVENTION_THRESHOLD
            or abs(dy_norm) > _INTERVENTION_THRESHOLD
            or abs(dz_norm) > _INTERVENTION_THRESHOLD
            or gripper != 1.0
        )

        if is_intervention:
            action = np.array([dx_norm, dy_norm, dz_norm, gripper], dtype=np.float32)

        obs, reward, terminated, truncated, info = self.env.step(action)

        rerecord = bool(events.get(TeleopEvents.RERECORD_EPISODE, False))
        terminate_manual = bool(events.get(TeleopEvents.TERMINATE_EPISODE, False))
        success = bool(events.get(TeleopEvents.SUCCESS, False))

        if terminate_manual:
            terminated = True
            if success:
                reward = 1.0
                logging.info("Episode ended successfully with reward 1.0")
            else:
                logging.info("Episode manually ended: FAILURE")

        info["is_intervention"] = is_intervention
        info["teleop_action"] = action
        info["rerecord_episode"] = rerecord

        if terminated or truncated:
            info["next.success"] = success

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self, "_teleop"):
            self._teleop.disconnect()
        return super().close()


def make_spacemouse_env(
    env_id: str,
    ee_step_size: dict | None = None,
    use_viewer: bool = True,
    use_gripper: bool = True,
    gripper_penalty: float = -0.02,
    reset_delay_seconds: float = 1.0,
    spacemouse_config: SpaceMouseTeleopConfig | None = None,
    **kwargs,
) -> gym.Env:
    """创建带 SpaceMouse 干预支持的 gym_hil Franka 仿真环境。

    包装顺序（由内到外）：
        PandaPickCubeGymEnv / PandaArrangeBoxesGymEnv  ← 基础 MuJoCo 环境
        GripperPenaltyWrapper                           ← 夹爪惩罚
        EEActionWrapper                                 ← 归一化 4-D 动作空间
        SpaceMouseInterventionWrapper                   ← SpaceMouse 干预注入
        PassiveViewerWrapper                            ← 可选 MuJoCo 被动视图
        ResetDelayWrapper                               ← reset 延时

    Args:
        env_id: 基础 env id，仅支持 "gym_hil/PandaPickCubeBase-v0" 与
            "gym_hil/PandaArrangeBoxesBase-v0"。
        ee_step_size: 末端执行器步长（米），归一化动作 1.0 对应的物理位移。
        use_viewer: 是否添加 MuJoCo 被动视图窗口。
        use_gripper: 是否启用夹爪控制通道及夹爪惩罚。
        gripper_penalty: 夹爪惩罚系数（仅 use_gripper=True 时生效）。
        reset_delay_seconds: reset 后的等待秒数。
        spacemouse_config: SpaceMouse 配置；None 时使用默认值。
        **kwargs: 透传给基础 env 构造函数（如 image_obs、render_mode 等）。
    """
    from gym_hil.envs.panda_arrange_boxes_gym_env import PandaArrangeBoxesGymEnv
    from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
    from gym_hil.wrappers.hil_wrappers import DEFAULT_EE_STEP_SIZE, EEActionWrapper, GripperPenaltyWrapper, ResetDelayWrapper
    from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

    if ee_step_size is None:
        ee_step_size = DEFAULT_EE_STEP_SIZE

    if env_id == "gym_hil/PandaPickCubeBase-v0":
        env = PandaPickCubeGymEnv(**kwargs)
    elif env_id == "gym_hil/PandaArrangeBoxesBase-v0":
        env = PandaArrangeBoxesGymEnv(**kwargs)
    else:
        raise ValueError(f"不支持的基础环境 id: {env_id}，仅支持 PandaPickCubeBase-v0 / PandaArrangeBoxesBase-v0")

    if use_gripper:
        env = GripperPenaltyWrapper(env, penalty=gripper_penalty)

    env = EEActionWrapper(env, ee_action_step_size=ee_step_size, use_gripper=use_gripper)
    env = SpaceMouseInterventionWrapper(env, ee_step_size=ee_step_size, spacemouse_config=spacemouse_config)

    if use_viewer:
        env = PassiveViewerWrapper(env)

    env = ResetDelayWrapper(env, delay_seconds=reset_delay_seconds)

    return env


# ---------------------------------------------------------------------------
# 模块级 env 注册（import 本模块时自动执行，防止重复注册）
# ---------------------------------------------------------------------------

_SPACEMOUSE_ENVS = [
    ("gym_hil/PandaPickCubeSpaceMouse-v0", "gym_hil/PandaPickCubeBase-v0"),
    ("gym_hil/PandaArrangeBoxesSpaceMouse-v0", "gym_hil/PandaArrangeBoxesBase-v0"),
]

for _sm_id, _base_id in _SPACEMOUSE_ENVS:
    # gym.spec() 在 gym_hil 命名空间未导入时会抛 NamespaceNotFound，
    # 命名空间已加载但 id 不存在时抛 NameNotFound，两者都应注册
    _registered = False
    try:
        gym.spec(_sm_id)
        _registered = True
    except (gym.error.NameNotFound, gym.error.NamespaceNotFound):
        pass

    if not _registered:
        gym.register(
            id=_sm_id,
            entry_point="lerobot.envs.gym_hil_spacemouse:make_spacemouse_env",
            max_episode_steps=100,
            kwargs={"env_id": _base_id, "use_viewer": True},
        )
