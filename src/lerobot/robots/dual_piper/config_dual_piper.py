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


@RobotConfig.register_subclass("dual_piper")
@dataclass
class DualPIPERConfig(RobotConfig):
    """Configuration for Dual Piper robot with two arms.
    
    Each arm has 7 joints, for a total of 14 joints.
    The naming convention follows the pattern:
    - Left arm: joint_1 to joint_7
    - Right arm: joint_8 to joint_14
    """
    # CAN ports for each arm
    can_port_left: str = 'can_left'
    can_port_right: str = 'can_right'
    
    # Joint names for both arms (14 joints total: 7 per arm)
    joint_names: list[str] = field(
        default_factory=lambda: [f"joint_{i + 1}" for i in range(14)]
    )
    
    # Camera configurations
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_left": RealSenseCameraConfig(
                serial_number_or_name="148522072680",
                fps=60,
                width=320,
                height=180,
            ),
            # "cam_right": RealSenseCameraConfig(
            #     serial_number_or_name="241222073777",
            #     fps=60,
            #     width=320,
            #     height=180,
            # ),
            "cam_top": RealSenseCameraConfig(
                serial_number_or_name="327122074756",
                fps=60,
                width=320,
                height=180,
            ),
        }
    )

