"""RL-token extraction and encoder/decoder (paper Sec. IV-A, Eq. 1-2).

Two pieces:

* :class:`SmolVLAPrefixExtractor` — runs the frozen SmolVLA VLM prefix
  (SigLIP images + language tokens + state token) once, returning the
  final-layer token embeddings ``z_{1:M}`` *and* the prefix KV cache so the
  same forward pass can also be reused to sample the VLA reference action
  chunk (flow-matching denoising loop of the action expert).
* :class:`RLTokenModule` — a lightweight cross-attention encoder whose learned
  queries read out the RL token ``z_rl`` (Eq. 1), plus a decoder that
  reconstructs the (stop-gradient) VLA embeddings from ``z_rl`` alone (Eq. 2).
"""

from __future__ import annotations

import math

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

def sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
    """openpi-RLT `fsq_tokenizer.sinusoidal_pe_init`: sin and cos concatenated."""
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    return torch.cat([torch.sin(position * div), torch.cos(position * div)], dim=-1)


class _GeGLU(nn.Module):
    """openpi-RLT `GeGLU`: Dense(2d) then `x * gelu(gate)` over the split."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)


class CrossAttentionLayer(nn.Module):
    """Pre-norm [self-attn, cross-attn, MLP], each residual.

    Port of openpi-RLT `fsq_tokenizer.CrossAttentionLayer`. The cross-attention
    block is what makes the bottleneck load-bearing: in the decoder it is the
    only path from an output position to the RL token.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d_model)
        d_ff = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.Dropout(dropout), _GeGLU(d_ff), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: Tensor, y: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm_self(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_cross(x)
        x = x + self.cross_attn(h, y, y, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class RLTokenModule(nn.Module):
    """Encoder-decoder bottleneck producing the RL token (Eq. 1-2).

    Ported from openpi-RLT `src/openpi/models/rl_token.py`. Both halves are
    cross-attention stacks over learned queries:

    * encoder — `num_rl_tokens` learned queries cross-attend to the VLA prefix
      embeddings and are read out as `z_rl`;
    * decoder — one learned query per output position, cross-attending to
      `z_rl` **only**.

    The decoder deliberately gets no teacher forcing. An autoregressive decoder
    fed the true `z_bar_1..z_bar_{t-1}` can reconstruct M-1 of M positions by
    interpolating neighbours and never read the bottleneck at all: L_ro falls,
    `recon_rel_err` falls, and `z_rl` stays empty. Measured on this repo's
    LIBERO setup, that variant held `bottleneck_gain` at 0.001 for a whole run
    while this one reaches ~1.4. `bottleneck_usage` is the probe that tells the
    two apart; the loss curve cannot.
    """
    def __init__(self, cfg: RLTokenConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.in_proj = nn.Linear(cfg.vla_width, d) if cfg.vla_width != d else nn.Identity()
        self.out_proj = nn.Linear(d, cfg.vla_width) if cfg.vla_width != d else nn.Identity()

        self.enc_query = nn.Parameter(sinusoidal_pe(cfg.num_rl_tokens, d))
        self.enc_pos = nn.Parameter(sinusoidal_pe(cfg.max_recon_tokens, d))
        self.dec_query = nn.Parameter(sinusoidal_pe(cfg.max_recon_tokens, d))
        self.dec_pos = nn.Parameter(sinusoidal_pe(cfg.num_rl_tokens, d))

        def stack(n: int) -> nn.ModuleList:
            return nn.ModuleList(
                [CrossAttentionLayer(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(n)]
            )

        self.encoder = stack(cfg.n_encoder_layers)
        self.decoder = stack(cfg.n_decoder_layers)

        # SmolVLA multiplies image embeddings by sqrt(width) (~31) inside
        # embed_prefix, so a raw MSE on z is dominated by whichever channels
        # happen to be large. Standardising per channel with running dataset
        # statistics makes L_ro measure information instead of scale. openpi-RLT
        # reconstructs raw embeddings; its VLA does not have this scaling.
        self.register_buffer("z_mean", torch.zeros(cfg.vla_width))
        self.register_buffer("z_var", torch.ones(cfg.vla_width))
        self.register_buffer("z_stats_count", torch.zeros(()))

    @property
    def z_rl_dim(self) -> int:
        """Width of the flattened RL token, i.e. the state dim stage 2 sees."""
        return self.cfg.d_model * self.cfg.num_rl_tokens

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
        return z[:, idx], mask[:, idx]

    def encode(self, z: Tensor, mask: Tensor | None = None, already_prepared: bool = False) -> Tensor:
        """Eq. 1: `num_rl_tokens` learned queries cross-attend to z_{1:M}.

        z: (B, M, vla_width); mask: (B, M) bool, True = valid token. Returns the
        flattened token (B, d_model * num_rl_tokens) — openpi-RLT flattens too,
        and with the default `num_rl_tokens = 1` this is just a squeeze.
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        if not already_prepared:
            z, mask = self._subsample(z, mask)
            z = self.normalize_z(z)
        b, m, _ = z.shape
        y = self.in_proj(z) + self.enc_pos[:m]
        x = self.enc_query.expand(b, -1, -1)
        for layer in self.encoder:
            x = layer(x, y, key_padding_mask=~mask)
        return x.reshape(b, -1)

    def decode(self, z_rl: Tensor, target_len: int) -> Tensor:
        """Eq. 2: one learned query per output position, attending to z_rl only."""
        b = z_rl.shape[0]
        y = z_rl.reshape(b, self.cfg.num_rl_tokens, self.cfg.d_model) + self.dec_pos
        x = self.dec_query[:target_len].expand(b, -1, -1)
        for layer in self.decoder:
            x = layer(x, y)
        return self.out_proj(x)

    def reconstruction_loss(
        self, z: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, dict[str, float]]:
        """Eq. 2. Targets are stop-gradient embeddings z_bar; the encoder also
        consumes z_bar since the VLA is frozen w.r.t. L_ro.

        Returns (loss, z_rl, metrics).
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z_bar, mask = self._subsample(z.detach(), mask)
        if self.training and self.cfg.normalize_targets:
            self._update_z_stats(z_bar, mask)
        z_bar = self.normalize_z(z_bar)

        z_rl = self.encode(z_bar, mask, already_prepared=True)
        pred = self.decode(z_rl, z_bar.shape[1])

        w = mask.float()
        denom = w.sum().clamp(min=1.0)
        err = (pred - z_bar).pow(2).mean(dim=-1)  # (B, M)
        loss = (err * w).sum() / denom

        with torch.no_grad():
            # Relative error tells you whether the bottleneck is too tight far
            # better than the raw loss does, and z_rl's spread across the batch
            # is a (noisy) collapse signal.
            target_energy = (z_bar.pow(2).mean(dim=-1) * w).sum() / denom
            metrics = {
                "recon_rel_err": (loss / target_energy.clamp(min=1e-8)).item(),
                "z_rl_std": z_rl.std(dim=0).mean().item(),
            }
        return loss, z_rl, metrics

    @torch.no_grad()
    def reconstruct(self, z: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(pred, z_bar, mask, z_rl) in the normalised space L_ro is measured in.

        `reconstruction_loss` collapses everything to one scalar, which cannot
        answer "was it the *language* tokens that failed to reconstruct" — and
        that question is the whole point of the stage-1 diagnostics, since image
        tokens outnumber language tokens ~5:1 and dominate the average. Doing
        the subsample/normalise dance in the caller instead would duplicate the
        two steps that have to stay in lockstep with training.
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z_bar, mask = self._subsample(z.detach(), mask)
        z_bar = self.normalize_z(z_bar)
        z_rl = self.encode(z_bar, mask, already_prepared=True)
        return self.decode(z_rl, z_bar.shape[1]), z_bar, mask, z_rl

    @torch.no_grad()
    def bottleneck_usage(self, z: Tensor, mask: Tensor | None = None) -> dict[str, float]:
        """How much of the reconstruction actually flows through z_rl.

        Re-runs the decoder with z_rl paired with the wrong observation.
        `bottleneck_gain` is how much worse that is, relative to the true loss.
        Near 0 means the token carries nothing and stage 2 would be RL on a
        constant state — which `loss_ro` alone cannot tell you, since a decoder
        with any shortcut drives it down regardless.

        Mismatching is done by rolling the batch rather than by `randperm`: a
        random permutation of 8 leaves one element in place on average, and each
        fixed point contributes a zero to the average, biasing the number toward
        "unused" exactly when the batch is small. Rolling is a derangement by
        construction, and averaging a few shifts cuts the single-batch variance
        that otherwise makes this metric swing an order of magnitude per log.
        """
        if mask is None:
            mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        z_bar, mask = self._subsample(z.detach(), mask)
        z_bar = self.normalize_z(z_bar)
        b, m, _ = z_bar.shape
        if b < 2:
            return {}

        z_rl = self.encode(z_bar, mask, already_prepared=True)
        w = mask.float()
        denom = w.sum().clamp(min=1.0)

        def loss_for(token: Tensor) -> Tensor:
            err = (self.decode(token, m) - z_bar).pow(2).mean(dim=-1)
            return (err * w).sum() / denom

        true_loss = loss_for(z_rl).clamp(min=1e-8)
        shifts = range(1, min(b, 4))
        gains = [((loss_for(z_rl.roll(s, 0)) - true_loss) / true_loss) for s in shifts]
        return {"bottleneck_gain": (sum(gains) / len(gains)).item()}

    @torch.no_grad()
    def rl_token(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """Inference helper: RL token for a batch of VLA embeddings."""
        return self.encode(z, mask)
