#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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



from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.acfql.configuration_acfql import ACFQLConfig, is_image_feature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


class ACFQLPolicy(
    PreTrainedPolicy,
):
    config_class = ACFQLConfig
    name = "acfql"

    def __init__(
        self,
        config: ACFQLConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config


    def get_optim_params(self) -> dict:
        optim_params = {
            "actor_bc_flow": [
                p
                for n, p in self.actor_bc_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "actor_onestep_flow": [
                p
                for n, p in self.actor_onestep_flow.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
        }

        return optim_params


    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("ACFQLPolicy does not support action chunking. It returns single actions!")

    def compute_flow_actions(self, observations, observations_features, noises: Tensor) -> Tensor:
        actions = noises
        flow_steps = self.config.flow_steps

        # Euler method.
        for i in range(flow_steps):
            t_val = float(i) / flow_steps
            t = torch.full((actions.shape[0], 1), t_val, device=noises.device)
            vels = self.actor_bc_flow(observations, observations_features, actions, t)
            actions = actions + vels / flow_steps

        actions = torch.clamp(actions, -1.0, 1.0)

        return actions


    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation"""
        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            observations_features = batch.get("observation_feature")
            batch_shape = batch["observation.state"].shape[0]
            action_dim = self.actor_onestep_flow.action_dim
            device = batch["observation.state"].device

            # Generate actions using distill-ddpg approach
            noises = torch.randn(batch_shape, action_dim, device=device)
            actions = self.actor_onestep_flow(batch, observations_features, noises)
            actions = torch.clamp(actions, -1.0, 1.0)

            # Reshape actions for chunking: [batch_size, chunk_size, action_dim_per_step]
            action_dim_per_step = action_dim // self.config.chunk_size
            actions = actions.reshape(batch_shape, self.config.chunk_size, action_dim_per_step)
            # Add actions to queue (transpose to get [chunk_size, batch_size, action_dim_per_step])
            self._action_queue.extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        actions = self._action_queue.popleft()

        return actions

    @torch.no_grad()
    def select_action_chunk(self, observations: dict[str, Tensor]) -> Tensor:
        """Select a full action chunk for QC-FQL open-loop execution.

        Returns:
            Tensor: Action chunk of shape [chunk_size, action_dim_per_step]
        """
        observations = observations
        observations_features = None

        batch_shape = observations["observation.state"].shape[0]
        action_dim = self.actor_onestep_flow.action_dim
        device = observations["observation.state"].device

        # Generate actions using one-step flow actor
        noises = torch.randn(batch_shape, action_dim, device=device)
        actions = self.actor_onestep_flow(observations, observations_features, noises)
        actions = torch.clamp(actions, -1.0, 1.0)

        # Reshape actions for chunking: [batch_size, chunk_size, action_dim_per_step]
        action_dim_per_step = action_dim // self.config.chunk_size
        actions = actions.reshape(batch_shape, self.config.chunk_size, action_dim_per_step)

        return actions[0]

    # def critic_forward(
    #     self,
    #     observations: dict[str, Tensor],
    #     actions: Tensor,
    #     use_target: bool = False,
    #     observation_features: Tensor | None = None,
    # ) -> Tensor:
