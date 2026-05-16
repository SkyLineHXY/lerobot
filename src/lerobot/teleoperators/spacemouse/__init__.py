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

# 导入时触发 @TeleoperatorConfig.register_subclass("spacemouse") 装饰器，
# 使 draccus 能够通过 --teleop.type=spacemouse 解析此配置。
from .configuration_spacemouse import SpaceMouseTeleopConfig
from .teleop_spacemouse import SpaceMouseTeleop

__all__ = ["SpaceMouseTeleopConfig", "SpaceMouseTeleop"]
