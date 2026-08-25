#!/usr/bin/env python

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

# Channel -> binding expression over SDL game-controller control names.
# "a-b" reads as "a minus b"; a leading "-" is the same with an empty positive
# side, so "-lefty" means pushing the stick up produces +1. An empty string
# disables the channel. `lerobot-find-gamepad` prints these live so a layout can
# be checked without launching a whole collection run.
#
# The signs below are in the robot base frame, which is also how the env reads
# the assembled command. RLT's sim teleop can reinterpret the same channels in
# the end-effector frame instead (`teleop.frame=tcp`, see rlt/teleop/frames.py);
# the bindings do not change, but delta_z then runs along the tool's approach
# axis and delta_x/delta_y span the plane across it.
#
# The signs below are in the robot base frame: stick-up is +x, stick-left is +y,
# right-trigger is +z. Note that this is *not* the same as pushing the gripper
# around the screen. LIBERO displays agentview as `img[::-1, ::-1]`, which leaves
# the picture upright but mirrored left-right, and the camera faces the robot
# from the far side, so on screen +z is up, +y is left, and +x is toward the
# viewer. Measured on libero_goal task 0 by stepping one channel at a time and
# projecting the eef back into the image. If steering feels inverted, flip the
# two translation channels rather than the rotations:
#   bindings: {delta_x: lefty, delta_y: -leftx}
DEFAULT_GAMEPAD_BINDINGS: dict[str, str] = {
    "delta_x": "leftx",
    "delta_y": "-lefty",
    "delta_z": "righttrigger-lefttrigger",
    "delta_roll": "dpright-dpleft",
    "delta_pitch": "-righty",
    "delta_yaw": "rightx",
    "gripper_close": "a",
    "gripper_open": "b",
    "intervention": "rightshoulder",
    "success": "y",
    "failure": "x",
    "rerecord": "back",
}


@TeleoperatorConfig.register_subclass("gamepad")
@dataclass
class GamepadTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    # Expose delta_roll/pitch/yaw, matching SpaceMouseTeleopConfig. Off by
    # default: the HIL-SERL pipeline reads only dx/dy/dz/gripper. Turn it on for
    # 6-DoF delta-EE envs such as LIBERO.
    use_rotation: bool = False
    deadzone: float = 0.1
    # Only the channels given here override DEFAULT_GAMEPAD_BINDINGS, so a config
    # that just wants yaw the other way round can say {"delta_yaw": "-rightx"}.
    bindings: dict[str, str] = field(default_factory=dict)
