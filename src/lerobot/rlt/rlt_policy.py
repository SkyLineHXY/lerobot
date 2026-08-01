"""RLT controller: frozen SmolVLA + RL token + actor for rollouts (Sec. V).

At each chunk boundary the controller
  1. runs the frozen VLA prefix once (images + language + state token),
  2. reads out the RL token z_rl and forms x = (z_rl, s^p),
  3. samples the VLA reference chunk a~_{1:H} from the cached prefix,
  4. queries the actor a_{1:C} ~ pi_theta(.|x, a~_{1:C}) (or returns the
     reference itself during warmup / for the base policy).

All actions live in the *normalized* action space used by SmolVLA;
un-normalize at the environment boundary with the SmolVLA postprocessor if
needed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.utils.constants import OBS_STATE

from .actor_critic import RLTAgent
from .rl_token import RLTokenModule, SmolVLAPrefixExtractor

class RLTController:
    def __init__(
        self,
        smolvla_policy,
        rl_token: RLTokenModule,
        agent: RLTAgent | None,
        chunk_len: int,
        action_dim: int,
        proprio_dim: int,
        image_only: bool = True,
        num_inference_steps: int | None = None,
    ):
        self.policy = smolvla_policy
        self.extractor = SmolVLAPrefixExtractor(smolvla_policy)
        self.rl_token = rl_token
        self.agent = agent
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.image_only = image_only
        self.num_inference_steps = num_inference_steps

    def _proprio(self, batch: dict[str, Tensor]) -> Tensor:
        sp = batch[OBS_STATE]
        if sp.ndim == 3:  # (B, 1, D) from delta_timestamps=[0.0]
            sp = sp.squeeze(1)
        return sp[:, : self.proprio_dim].float()

    @torch.no_grad()
    def compute_x(self, batch: dict[str, Tensor]) -> Tensor:
        """RL state x = (z_rl, s^p) for a (batched) observation."""
        feats = self.extractor.extract(batch)
        z, m = self.extractor.select_tokens(feats, self.image_only)
        z_rl = self.rl_token.rl_token(z, m)
        return torch.cat([z_rl, self._proprio(batch)], dim=-1)


    @torch.no_grad()
    def plan_chunk(
        self,
        batch: dict[str, Tensor],
        use_actor: bool = True,
        deterministic: bool = False,
    ) -> dict[str, Tensor]:
        """One chunk-boundary planning step.
        Returns x (B, D_x), ref_full (B, H, d) in normalized action space,
        and action_chunk (B, C, d) to execute.
        """
        feats = self.extractor.extract(batch)
        z, m = self.extractor.select_tokens(feats, self.image_only)
        z_rl = self.rl_token.rl_token(z, m)
        x = torch.cat([z_rl, self._proprio(batch)], dim=-1)

        ref_full = self.extractor.sample_reference_chunk(
            feats, num_steps=self.num_inference_steps
        )[:, :, : self.action_dim]
        ref_chunk = ref_full[:, : self.chunk_len]

        if use_actor and self.agent is not None:
            action_chunk = self.agent.act(x, ref_chunk, deterministic=deterministic)
        else:
            action_chunk = ref_chunk

        return {"x": x, "ref_full": ref_full, "action_chunk": action_chunk}