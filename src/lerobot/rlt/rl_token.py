"""RL-token extraction and encoder/decoder (paper Sec. IV-A, Eq. 1-2).

Two pieces:

* :class:`SmolVLAPrefixExtractor` — runs the frozen SmolVLA VLM prefix
  (SigLIP images + language tokens + state token) once, returning the
  final-layer token embeddings ``z_{1:M}`` *and* the prefix KV cache so the
  same forward pass can also be reused to sample the VLA reference action
  chunk (flow-matching denoising loop of the action expert).
* :class:`RLTokenModule` — a lightweight encoder that appends a learned
  ``e_rl`` embedding and reads out the RL token ``z_rl`` (Eq. 1), plus a
  causal decoder trained with teacher forcing to autoregressively reconstruct
  the (stop-gradient) VLA embeddings from ``z_rl`` (Eq. 2).
"""

from __future__ import annotations
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from .configs import RLTokenConfig

class SmolVLAPrefixExtractor:
    """Frozen-VLA feature extraction mirroring ``VLAFlowMatching.sample_actions``.

    The prefix forward is done exactly once per observation; its outputs are
    reused both for the RL token (final-layer embeddings) and for sampling the
    reference action chunk (KV cache for the denoising loop).

    SmolVLA prefix layout (``embed_prefix``): per-camera connector tokens
    (64 each for 512x512 inputs), then language tokens, then one state token.
    """
    def __init__(self, policy):
        self.policy = policy
        self.model = policy.model  # VLAFlowMatching
        self.config = policy.config

    @torch.no_grad()
    def extract(self, batch: dict[str, Tensor]) -> dict:
        model = self.model
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks, state=state
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Same call as the prefill in sample_actions, but keeping the prefix
        # final-layer hidden states in addition to the KV cache.
        outputs_embeds, past_key_values = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        prefix_out = outputs_embeds[0]
        # Prefix = [img tokens x cameras, lang tokens, state token]; language
        # and state tokens sit at the end (no prefix_length padding in
        # smolvla_base: prefix_length <= 0).
        n_state = 1
        n_lang = tokens.shape[1]
        n_img = prefix_out.shape[1] - n_lang - n_state

        return {
            "z": prefix_out.to(torch.float32),  # (B, M_total, vla_width)
            "pad_mask": prefix_pad_masks.bool(),  # (B, M_total)
            "n_img_tokens": n_img,
            "past_key_values": past_key_values,
            "prefix_pad_masks": prefix_pad_masks,
        }

    def select_tokens(self, feats: dict, image_only: bool) -> tuple[Tensor, Tensor]:
        """Return (z, mask) restricted to image tokens if requested."""
        z, mask = feats["z"], feats["pad_mask"]
        if image_only:
            n = feats["n_img_tokens"]
            z, mask = z[:, :n], mask[:, :n]
        return z, mask


    @torch.no_grad()
    def sample_reference_chunk(self, feats: dict, num_steps: int | None = None) -> Tensor:
        """Sample the VLA reference chunk a~_{1:H} reusing the prefix KV cache.

        Returns padded actions of shape (B, chunk_size, max_action_dim).
        """
        model = self.model
        if num_steps is None:
            num_steps = self.config.num_steps


        prefix_pad_masks = feats["prefix_pad_masks"]
        past_key_values = feats["past_key_values"]
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device

        x_t = model.sample_noise(
            (bsize, self.config.chunk_size, self.config.max_action_dim), device
        )
        dt = -1.0 / num_steps

        for step in range(num_steps):
            time = 1 + step*dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = model.denoise_step(prefix_pad_masks, past_key_values, x_t, time_tensor)
            x_t = x_t + dt * v_t
        return x_t

def _causal_mask(size: int, device) -> Tensor:
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

class RLTokenModule(nn.Module):
    """Encoder-decoder bottleneck producing the RL token (Eq. 1-2)."""
    def __init__(self, cfg: RLTokenConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        ff = int(cfg.d_model * cfg.mlp_ratio)

        self.enc_in_proj = nn.Linear(cfg.vla_width, d)
        self.e_rl = nn.Parameter(torch.randn(d) * 0.02)
        self.enc_pos = nn.Parameter(torch.zeros(cfg.max_recon_tokens + 1, d))

        enc_layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, ff, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, cfg.n_encoder_layers, norm=nn.LayerNorm(d)
        )
        self.dec_in_proj = nn.Linear(cfg.vla_width, d)
        self.dec_pos = nn.Parameter(torch.zeros(cfg.max_recon_tokens + 1, d))
        dec_layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, ff, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            dec_layer, cfg.n_decoder_layers, norm=nn.LayerNorm(d)
        )
        self.out_proj = nn.Linear(d, cfg.vla_width)  # h_phi

        nn.init.trunc_normal_(self.enc_pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)

    def _subsample(self, z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        m = z.shape[1]
        if m <= self.cfg.max_recon_tokens:
            return z, mask
        idx = torch.linspace(0, m - 1, self.cfg.max_recon_tokens, device=z.device).long()
        return z[:,idx] , mask[:,idx]

    def encode(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """Eq. 1: z_rl = g_phi([z_{1:M}, e_rl])_{M+1}.
        z: (B, M, vla_width); mask: (B, M) bool, True = valid token.
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z, mask = self._subsample(z, mask)
        b, m, _ = z.shape
        h = self.enc_in_proj(z)
        h = torch.cat([h, self.e_rl.expand(b, 1, -1)], dim=1)
        h = h + self.enc_pos[: m + 1]
        pad = torch.cat(
            [~mask, torch.zeros(b, 1, dtype=torch.bool, device=z.device)], dim=1
        )
        out = self.encoder(h, src_key_padding_mask=pad)

        return out[:, -1]

    def reconstruction_loss(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """Eq. 2 with teacher forcing (parallel autoregressive training).
        Targets and decoder inputs use stop-gradient embeddings z_bar; the
        encoder also consumes z_bar since the VLA is frozen w.r.t. L_ro.
        Returns (loss, z_rl).
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z_bar = z.detach()
        z_bar, mask = self._subsample(z_bar, mask)

        b, m, _ = z_bar.shape

        z_rl = self.encode(z_bar, mask)
        # Decoder input: [z_rl, proj(z_bar_1), ..., proj(z_bar_{M-1})]
        dec_in = torch.cat(
            [z_rl.unsqueeze(1), self.dec_in_proj(z_bar[:, :-1])], dim=1
        )
        dec_in = dec_in + self.dec_pos[:m]
        out = self.decoder(dec_in, mask=_causal_mask(m, z.device))
        pred = self.out_proj(out)  # predicts z_bar_1 .. z_bar_M

        err = (pred - z_bar).pow(2).mean(dim=-1)  # (B, M)
        w = mask.float()
        loss = (err * w).sum() / w.sum().clamp(min=1.0)
        return loss, z_rl


    @torch.no_grad()
    def rl_token(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """Inference helper: RL token for a batch of VLA embeddings."""
        return self.encode(z, mask)