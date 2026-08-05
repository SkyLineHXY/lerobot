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

import torch
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

from .configuration_rlt import RLTokenConfig

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
        n_lang = tokens.shape[1]
        att = prefix_att_masks[0].bool()
        state_pos = att.nonzero()
        if state_pos.numel() == 0:
            raise RuntimeError(
                "No state token found in the SmolVLA prefix attention mask; "
                "the prefix layout assumed by the RL token extractor changed."
            )
        state_start = int(state_pos[0])
        n_state = int(att.sum())
        lang_start = state_start - n_lang
        if lang_start < 0:
            raise RuntimeError(
                f"Inconsistent SmolVLA prefix: state token at {state_start} but "
                f"{n_lang} language tokens expected before it."
            )

        return {
            "z": prefix_out.to(torch.float32),  # (B, M_total, vla_width)
            "pad_mask": prefix_pad_masks.bool(),  # (B, M_total)
            "lang_start": lang_start,  # == number of image-region tokens
            "state_start": state_start,
            "n_state": n_state,
            "past_key_values": past_key_values,
            "prefix_pad_masks": prefix_pad_masks,
        }

    def select_tokens(
        self, feats: dict, drop_language: bool = True, image_only: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Select which VLA embeddings the RL token has to compress.

        Paper footnote 1 drops the *language* embeddings when the task
        instruction is fixed. Everything else is kept — for SmolVLA that means
        the image tokens plus the single state token, which is the only
        proprioceptive entry point in the VLM prefix. `image_only` reproduces
        the stricter variant that drops the state token too.
        """
        z, mask = feats["z"], feats["pad_mask"]
        n_img, s0, ns = feats["lang_start"], feats["state_start"], feats["n_state"]
        if image_only:
            return z[:, :n_img], mask[:, :n_img]
        if drop_language:
            z = torch.cat([z[:, :n_img], z[:, s0 : s0 + ns]], dim=1)
            mask = torch.cat([mask[:, :n_img], mask[:, s0 : s0 + ns]], dim=1)
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

        # SmolVLA multiplies image embeddings by sqrt(width) (~31) inside
        # embed_prefix, so raw-MSE reconstruction is dominated by whichever
        # channels happen to be large. Standardising per channel with running
        # dataset statistics makes L_ro measure information instead of scale.
        # The buffers travel with the checkpoint, so stage 2 sees the same
        # normalisation as stage 1.
        self.register_buffer("z_mean", torch.zeros(cfg.vla_width))
        self.register_buffer("z_var", torch.ones(cfg.vla_width))
        self.register_buffer("z_stats_count", torch.zeros(()))

    @torch.no_grad()
    def _update_z_stats(self, z: Tensor, mask: Tensor, momentum: float = 0.01) -> None:
        flat = z[mask]  # (N_valid, vla_width)
        if flat.numel() == 0:
            return
        mean, var = flat.mean(0), flat.var(0, unbiased=False)
        if self.z_stats_count == 0:  # first batch: initialise rather than blend
            self.z_mean.copy_(mean)
            self.z_var.copy_(var)
        else:
            self.z_mean.lerp_(mean, momentum)
            self.z_var.lerp_(var, momentum)
        self.z_stats_count += 1

    def normalize_z(self, z: Tensor) -> Tensor:
        if not self.cfg.normalize_targets:
            return z
        return (z - self.z_mean) / (self.z_var + 1e-6).sqrt()

    def _subsample(self, z: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        m = z.shape[1]
        if m <= self.cfg.max_recon_tokens:
            return z, mask
        idx = torch.linspace(0, m - 1, self.cfg.max_recon_tokens, device=z.device).long()
        return z[:,idx] , mask[:,idx]

    def encode(
        self, z: Tensor, mask: Tensor | None = None, already_prepared: bool = False
    ) -> Tensor:
        """Eq. 1: z_rl = g_phi([z_{1:M}, e_rl])_{M+1}.

        z: (B, M, vla_width); mask: (B, M) bool, True = valid token.
        `already_prepared` skips subsampling/normalisation for callers that
        did it themselves (the reconstruction loss shares both with the
        decoder targets).
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if not already_prepared:
            z, mask = self._subsample(z, mask)
            z = self.normalize_z(z)
        b, m, _ = z.shape
        h = self.enc_in_proj(z)
        h = torch.cat([h, self.e_rl.expand(b, 1, -1)], dim=1)
        h = h + self.enc_pos[: m + 1]
        pad = torch.cat(
            [~mask, torch.zeros(b, 1, dtype=torch.bool, device=z.device)], dim=1
        )
        out = self.encoder(h, src_key_padding_mask=pad)

        return out[:, -1]

    def reconstruction_loss(
        self, z: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Eq. 2 with teacher forcing (parallel autoregressive training).

        Targets and decoder inputs use stop-gradient embeddings z_bar; the
        encoder also consumes z_bar since the VLA is frozen w.r.t. L_ro.
        Returns (loss, z_rl, metrics).
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z_bar = z.detach()
        z_bar, mask = self._subsample(z_bar, mask)
        if self.training and self.cfg.normalize_targets:
            self._update_z_stats(z_bar, mask)
        z_bar = self.normalize_z(z_bar)

        b, m, _ = z_bar.shape

        z_rl = self.encode(z_bar, mask, already_prepared=True)
        # Decoder input: [z_rl, proj(z_bar_1), ..., proj(z_bar_{M-1})]
        dec_in = torch.cat(
            [z_rl.unsqueeze(1), self.dec_in_proj(z_bar[:, :-1])], dim=1
        )
        dec_in = dec_in + self.dec_pos[:m]
        out = self.decoder(dec_in, mask=_causal_mask(m, z.device))
        pred = self.out_proj(out)  # predicts z_bar_1 .. z_bar_M

        w = mask.float()
        denom = w.sum().clamp(min=1.0)
        err = (pred - z_bar).pow(2).mean(dim=-1)  # (B, M)
        loss = (err * w).sum() / denom

        with torch.no_grad():
            # Relative error tells you whether the bottleneck is too tight far
            # better than the raw loss does, and z_rl's spread tells you
            # whether the token has collapsed to a constant.
            target_energy = (z_bar.pow(2).mean(dim=-1) * w).sum() / denom
            metrics = {
                "recon_rel_err": (loss / target_energy.clamp(min=1e-8)).item(),
                "z_rl_std": z_rl.std(dim=0).mean().item(),
            }
        return loss, z_rl, metrics


    @torch.no_grad()
    def rl_token(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """Inference helper: RL token for a batch of VLA embeddings."""
        return self.encode(z, mask)