# !/usr/bin/env python

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

from enum import IntEnum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.processor import RobotAction

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_spacemouse import SpaceMouseTeleopConfig
from .spacemouse_expert import SpaceMouseExpert


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


class SpaceMouseTeleop(Teleoperator):
    """3Dconnexion SpaceMouse 遥操作器。

    从 SpaceMouseExpert 读取 6-DoF 增量（位移 + 旋转）与按钮状态，
    输出符合 LeRobot Teleoperator 规范的动作字典。
    """

    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._expert: SpaceMouseExpert | None = None
        self._last_gripper = GripperAction.OPEN.value
        self._prev_buttons: list[int] = []

    # ------------------------------------------------------------------
    # Feature descriptors
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        """动作特征描述符（shape/dtype/names）。"""
        names: dict[str, int] = {"delta_x": 0, "delta_y": 1, "delta_z": 2}
        idx = 3
        if self.config.use_rotation:
            names["delta_roll"] = idx
            names["delta_pitch"] = idx + 1
            names["delta_yaw"] = idx + 2
            idx += 3
        if self.config.use_gripper:
            names["gripper"] = idx
            idx += 1
        return {"dtype": "float32", "shape": (idx,), "names": names}

    @property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._expert is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self, calibrate: bool = True) -> None:
        if self._expert is not None:
            return
        self._expert = SpaceMouseExpert()
        self._prev_buttons = []

    def disconnect(self) -> None:
        if self._expert is not None:
            self._expert.close()
            self._expert = None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    # Action & events
    # ------------------------------------------------------------------

    def get_action(self) -> RobotAction:
        """读取当前 SpaceMouse 状态，返回平坦动作字典。

        返回值的 key 集合由 action_features 声明的 names 决定：
        - 始终包含 delta_x / delta_y / delta_z（经缩放与符号翻转后，单位：米）；
        - use_rotation=True 时加入 delta_roll / delta_pitch / delta_yaw（弧度）；
        - use_gripper=True 时加入 gripper（0=合/1=保持/2=开），与 GamepadTeleop 约定一致。
        """
        if self._expert is None:
            zeros = {k: 0.0 for k in ("delta_x", "delta_y", "delta_z")}
            if self.config.use_rotation:
                zeros.update({"delta_roll": 0.0, "delta_pitch": 0.0, "delta_yaw": 0.0})
            if self.config.use_gripper:
                zeros["gripper"] = float(GripperAction.STAY.value)
            return zeros

        raw, buttons = self._expert.get_action()
        self._prev_buttons = list(buttons)

        # 6-DoF 增量（spacemouse_expert 已完成轴重映射）
        action6 = np.array(raw[:6], dtype=np.float64)
        signs = np.array(self.config.channel_signs, dtype=np.float64)
        action6 *= signs

        pos_scale = self.config.position_scale
        rot_scale = self.config.orientation_scale

        result: RobotAction = {
            "delta_x": float(action6[0] * pos_scale),
            "delta_y": float(action6[1] * pos_scale),
            "delta_z": float(action6[2] * pos_scale),
        }

        if self.config.use_rotation:
            result["delta_roll"] = float(action6[3] * rot_scale)
            result["delta_pitch"] = float(action6[4] * rot_scale)
            result["delta_yaw"] = float(action6[5] * rot_scale)

        if self.config.use_gripper:
            btn_open = self._safe_button(buttons, self.config.gripper_open_button)
            btn_close = self._safe_button(buttons, self.config.gripper_close_button)
            if btn_open:
                gripper_action = GripperAction.OPEN.value
            elif btn_close:
                gripper_action = GripperAction.CLOSE.value
            else:
                gripper_action = GripperAction.STAY.value
            self._last_gripper = gripper_action
            result["gripper"] = float(gripper_action)

        return result

    def get_teleop_events(self) -> dict[str, Any]:
        """从按钮状态读取遥操作事件。

        IS_INTERVENTION 为电平触发（按下期间持续为 True）；
        SUCCESS / TERMINATE_EPISODE / RERECORD_EPISODE 为边沿触发（按下瞬间 True，下一帧恢复 False）。
        """
        if self._expert is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        _, buttons = self._expert.get_action()
        buttons = list(buttons)

        is_intervention = bool(self._safe_button(buttons, self.config.intervention_button))

        # 边沿检测：当前帧按下而上一帧未按下
        terminate = self._edge(buttons, self._prev_buttons, self.config.terminate_button)
        success = (
            self._edge(buttons, self._prev_buttons, self.config.success_button)
            if self.config.success_button is not None
            else False
        )
        rerecord = (
            self._edge(buttons, self._prev_buttons, self.config.rerecord_button)
            if self.config.rerecord_button is not None
            else False
        )

        self._prev_buttons = buttons

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord,
        }

    def get_se3_delta(self, action_dict: dict | None = None) -> np.ndarray:
        """把当前（或传入的）动作字典转换为 4×4 SE(3) 增量矩阵。

        供 lerobot_collect_franka_multicam.py 做 delta → absolute pose 积分使用。
        若 action_dict 为 None 则自动调用 get_action()。

        返回：
            (4, 4) SE(3) numpy 数组，表示在当前 EE 坐标系下的位姿增量。
        """
        if action_dict is None:
            action_dict = self.get_action()

        dx = action_dict.get("delta_x", 0.0)
        dy = action_dict.get("delta_y", 0.0)
        dz = action_dict.get("delta_z", 0.0)
        dr = action_dict.get("delta_roll", 0.0)
        dp = action_dict.get("delta_pitch", 0.0)
        dy_ = action_dict.get("delta_yaw", 0.0)

        delta = np.eye(4)
        delta[:3, :3] = R.from_rotvec([dr, dp, dy_]).as_matrix()
        delta[:3, 3] = [dx, dy, dz]
        return delta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_button(buttons: list[int], idx: int) -> int:
        """安全读取按钮列表，越界时返回 0。"""
        if idx < len(buttons):
            return int(buttons[idx])
        return 0

    @staticmethod
    def _edge(curr: list[int], prev: list[int], idx: int) -> bool:
        """检测指定按键的上升沿（当前帧按下，上一帧未按下）。"""
        curr_val = SpaceMouseTeleop._safe_button(curr, idx)
        prev_val = SpaceMouseTeleop._safe_button(prev, idx)
        return bool(curr_val and not prev_val)
