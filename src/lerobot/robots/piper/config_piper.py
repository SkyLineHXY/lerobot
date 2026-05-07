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
from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
@RobotConfig.register_subclass("piper")
@dataclass
class PIPERConfig(RobotConfig):
    can_port: str = 'can_left'
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i + 1}" for i in range(7)])
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_left": RealSenseCameraConfig(
                serial_number_or_name="148522072680",
                fps=60,
                width=320,
                height=180,
            ),
            "cam_top": RealSenseCameraConfig(
                serial_number_or_name="327122074756",
                fps=60,
                width=320,
                height=180,
            ),
        }
    )