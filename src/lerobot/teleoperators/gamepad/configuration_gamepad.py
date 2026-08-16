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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gamepad")
@dataclass
class GamepadTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    # Expose delta_roll/pitch/yaw (right stick + shoulder buttons), matching
    # SpaceMouseTeleopConfig. Off by default: the HIL-SERL pipeline reads only
    # dx/dy/dz/gripper. Turn it on for 6-DoF delta-EE envs such as LIBERO.
    use_rotation: bool = False
    # pygame axis indices for (roll, pitch, yaw). Axis numbering is not
    # standardised across controllers and the pad may not even have this many,
    # so verify these before trusting a demonstration; a missing axis reads 0.
    # The defaults avoid 0/1/3, which the translation channels already use.
    rotation_axes: tuple[int, int, int] = (2, 4, 5)
