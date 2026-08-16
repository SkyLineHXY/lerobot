#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""LIBERO camera keys must reach the policy under the names it was trained with."""

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.utils import check_env_policy_image_keys, env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import OBS_IMAGES

WRIST_MAPPING = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}


def _policy_cfg(*image_keys: str) -> ACTConfig:
    cfg = ACTConfig()
    cfg.input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for key in image_keys
    }
    return cfg


def _image_keys(env_cfg: LiberoEnv) -> set[str]:
    return {k for k, ft in env_to_policy_features(env_cfg).items() if ft.type is FeatureType.VISUAL}


def test_default_mapping_keeps_upstream_keys():
    assert _image_keys(LiberoEnv()) == {f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2"}


def test_mapping_renames_the_wrist_camera():
    env_cfg = LiberoEnv(camera_name_mapping=WRIST_MAPPING)

    assert _image_keys(env_cfg) == {f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.wrist_image"}
    # The gym env has to be built with the same mapping, otherwise the features declared here
    # describe keys the env never emits.
    assert env_cfg.gym_kwargs["camera_name_mapping"] == WRIST_MAPPING


def test_single_camera_drops_the_unused_feature():
    env_cfg = LiberoEnv(camera_name="agentview_image", camera_name_mapping=WRIST_MAPPING)

    assert _image_keys(env_cfg) == {f"{OBS_IMAGES}.image"}


def test_unmapped_camera_raises():
    with pytest.raises(ValueError, match="camera_name_mapping"):
        LiberoEnv(camera_name="agentview_image,robot0_eye_in_hand_image,birdview_image")


def test_mismatched_keys_raise():
    policy_cfg = _policy_cfg(f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.wrist_image")

    with pytest.raises(ValueError, match="wrist_image"):
        check_env_policy_image_keys(LiberoEnv(), policy_cfg)

    check_env_policy_image_keys(LiberoEnv(camera_name_mapping=WRIST_MAPPING), policy_cfg)


def test_rename_map_counts_as_a_fix():
    policy_cfg = _policy_cfg(f"{OBS_IMAGES}.camera1", f"{OBS_IMAGES}.camera2")
    rename_map = {
        f"{OBS_IMAGES}.image": f"{OBS_IMAGES}.camera1",
        f"{OBS_IMAGES}.image2": f"{OBS_IMAGES}.camera2",
    }

    check_env_policy_image_keys(LiberoEnv(), policy_cfg, rename_map)


def test_fewer_cameras_only_warns():
    env_cfg = LiberoEnv(camera_name="agentview_image", camera_name_mapping=WRIST_MAPPING)

    with pytest.warns(UserWarning, match="wrist_image"):
        check_env_policy_image_keys(env_cfg, _policy_cfg(f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.wrist_image"))
