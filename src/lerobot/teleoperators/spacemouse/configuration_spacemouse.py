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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    """SpaceMouse 遥操作器配置。

    6-DoF 3Dconnexion SpaceMouse（位移 + 旋转），可选夹爪与旋转通道。
    """

    use_gripper: bool = True
    """是否在动作向量中加入夹爪通道。"""

    use_rotation: bool = False
    """是否在动作向量中暴露旋转通道（delta_roll/pitch/yaw）。
    RL 管线只读 dx/dy/dz[/gripper]，故默认关闭；真机采集时可置 True。"""

    position_scale: float = 0.009
    """位移缩放因子（米/tick），将 SpaceMouse 归一化值 [-1,1] 映射到物理增量。"""

    orientation_scale: float = 0.02
    """旋转缩放因子（弧度/tick），将 SpaceMouse 归一化值 [-1,1] 映射到角度增量。"""

    channel_signs: tuple[int, int, int, int, int, int] = field(
        default_factory=lambda: (1, 1, 1, 1, 1, 1)
    )
    """各通道符号翻转（±1），顺序为 [x, y, z, roll, pitch, yaw]。
    注意：spacemouse_expert 已对原始输出做了 [-y, x, z, -roll, -pitch, -yaw] 的轴重映射，
    此处的 channel_signs 是在重映射之后叠加的额外翻转，用于适配不同安装方向。"""

    gripper_open_button: int = 0
    """触发"夹爪全开"的按键索引（左键，index 0）。"""

    gripper_close_button: int = 1
    """触发"夹爪全合"的按键索引（右键，index 1）。"""

    intervention_button: int = 0
    """触发"人工干预"事件（IS_INTERVENTION）的按键索引。"""

    terminate_button: int = 1
    """触发"终止 episode"事件（TERMINATE_EPISODE）的按键索引。"""

    success_button: int | None = None
    """触发"成功"事件（SUCCESS）的按键索引；None 表示不绑定。"""

    rerecord_button: int | None = None
    """触发"重录 episode"事件（RERECORD_EPISODE）的按键索引；None 表示不绑定。"""
