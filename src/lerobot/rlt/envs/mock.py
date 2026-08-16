"""A mock chunk env for smoke tests.

A 6-DoF abstract reaching task (SO-100-like action space, matching
``smolvla_base``) with three synthetic camera views. It exists purely to
exercise the plumbing — frozen VLA forward, RL token, actor-critic, replay,
updates — without a robot.

It is **not** a validation signal for the algorithm: its reward depends only on
the first 2 of 6 state dims, `done == success` so episodes never terminate on
failure, and it treats normalized actions as physical quantities (no
un-normalisation boundary at all). Use LIBERO for evidence that RLT works.
"""
from __future__ import annotations

import torch
from torch import Tensor

from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..teleop.base import InterventionManager, InterventionResult
from ..teleop.keys import KeyboardEventListener

SMOLVLM_TOKENIZER = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


def _tokenize_prompt(prompt: str, max_length: int = 48) -> tuple[Tensor, Tensor]:
    """Tokenize with the SmolVLM2 tokenizer if cached, else dummy ids.

    Mirrors the SmolVLA preprocessor: trailing newline + right padding to
    ``tokenizer_max_length``.
    """
    if not prompt.endswith("\n"):
        prompt = f"{prompt}\n"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SMOLVLM_TOKENIZER)
        enc = tok(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"][0], enc["attention_mask"][0].bool()
    except Exception:
        ids = torch.zeros(max_length, dtype=torch.long)
        ids[:8] = torch.arange(2, 10)
        mask = torch.zeros(max_length, dtype=torch.bool)
        mask[:8] = True
        return ids, mask

class MockManipEnv:
    """Abstract 6-DoF 'reach the origin' task with synthetic cameras.

    State: 6-dim joint vector; action: 6-dim delta command in [-1, 1].
    Sparse reward: +1 (and done) when ||first 2 dims|| < eps. Camera images
    are procedurally derived from the state so the visual pathway carries
    information about the task. Camera keys match ``smolvla_base``.
    """

    camera_keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
    )

    def __init__(
        self,
        action_dim: int = 6,
        max_episode_steps: int = 120,
        image_size: int = 256,
        success_eps: float = 0.15,
        prompt: str = "reach the target",
        seed: int = 0,
        keys: KeyboardEventListener | None = None,
        intervention: InterventionManager | None = None,
    ):
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self.success_eps = success_eps
        self.rng = torch.Generator().manual_seed(seed)
        self.tokens, self.token_mask = _tokenize_prompt(prompt)
        self.keys = keys or KeyboardEventListener(backend="none")
        self.intervention = intervention or InterventionManager()
        self._state: Tensor | None = None
        self._t = 0

    @property
    def action_names(self) -> list[str]:
        return [f"delta_{i}" for i in range(self.action_dim)]

    def reset(self) -> dict:
        self._state = torch.rand(self.action_dim, generator=self.rng) * 1.6 - 0.8
        self._t = 0
        self.intervention.on_reset()
        return self._obs()

    def apply_action(self, action: Tensor) -> tuple[dict, float, bool, bool]:
        action = action.detach().cpu().clamp(-1, 1)
        self._state = (self._state + 0.05 * action).clamp(-1, 1)
        self._t += 1
        dist = self._state[:2].norm().item()
        success = dist < self.success_eps
        truncated = (not success) and self._t >= self.max_episode_steps
        return self._obs(), 1.0 if success else 0.0, success, truncated

    def step(self, action: Tensor) -> tuple[dict, float, bool]:
        obs, reward, done, _ = self.apply_action(action)
        return obs, reward, done

    def env_action_to_normalized(self, raw) -> Tensor:
        """Identity: the mock has no normalisation boundary at all."""
        return torch.as_tensor(raw, dtype=torch.float32).reshape(-1)[: self.action_dim]

    def run_intervention(self, chunk_len: int) -> InterventionResult | None:
        return self.intervention.run_chunk(chunk_len)

    def intervention_pending(self) -> bool:
        return self.intervention.check()

    def close(self) -> None:
        self.intervention.close()

    def _obs(self) -> dict:
        return {"state": self._state.clone(), "t": self._t}

    def _render(self, state: Tensor, cam: int) -> Tensor:
        """Cheap deterministic 'render': colored blocks encoding the state."""
        s = self.image_size
        img = torch.zeros(3, s, s)
        img[cam % 3] = 0.2
        px = int((state[0].item() + 1) / 2 * (s - 32))
        py = int((state[1].item() + 1) / 2 * (s - 32))
        img[:, py : py + 32, px : px + 32] = 0.9
        vals = (state + 1) / 2
        band = max(s // len(vals), 1)
        for i, v in enumerate(vals):
            img[2, :16, i * band : (i + 1) * band] = v
        return img


    def obs_to_batch(self, obs_list: list[dict], device) -> dict[str, Tensor]:
        states = torch.stack([o["state"] for o in obs_list])
        b = states.shape[0]
        batch: dict[str, Tensor] = {}
        for c, key in enumerate(self.camera_keys):
            imgs = torch.stack([self._render(o["state"], c) for o in obs_list])
            batch[key] = imgs.to(device)
        # Raw 6-dim state; SmolVLA's prepare_state pads to max_state_dim.
        batch[OBS_STATE] = states.to(device)
        batch[OBS_LANGUAGE_TOKENS] = self.tokens.expand(b, -1).to(device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = self.token_mask.expand(b, -1).to(device)
        return batch
