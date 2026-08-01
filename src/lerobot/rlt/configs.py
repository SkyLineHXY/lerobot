"""Configuration dataclasses for the RLT reproduction on SmolVLA.

Defaults follow the paper (App. B) where specified; unspecified values use
sensible defaults documented inline. VLA-dependent widths/dims default to
the `lerobot/smolvla_base` checkpoint (SmolVLM2-500M backbone, 3 cameras,
6-dim SO-100 action space).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RLTokenConfig:
    """Stage 1: RL-token encoder/decoder (Eq. 1-2)."""

    # Width of the VLA final-layer embeddings (SmolVLM2-500M text -> 960).
    vla_width: int = 960
    # Transformer width for encoder/decoder. The paper depicts a 1x2048 RL
    # token on pi0.6 (= VLA width); 512 keeps the module light (~15M params)
    # on the 960-wide SmolVLA. Use --d-model 960 to match "RL token = VLA
    # width" exactly.
    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Paper footnote 1: with a fixed per-task instruction, language tokens are
    # dropped and only image-token embeddings are compressed/reconstructed.
    use_image_tokens_only: bool = True
    # Max number of prefix tokens kept for reconstruction (subsample if more;
    # SmolVLA: 64 connector tokens per camera x 3 cameras = 192 image tokens,
    # so 256 also covers the language+state tokens if image_only is disabled).
    max_recon_tokens: int = 256

    # Stage-1 training (paper: 2000-10000 steps on task demos)
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    steps: int = 5000
    batch_size: int = 8
    # Weight alpha for optional joint VLA supervised fine-tuning (Algorithm 1
    # line 3). alpha = 0 keeps the VLA fully frozen.
    vla_sft_alpha: float = 0.0


@dataclass
class ActorCriticConfig:
    """Stage 2 networks (paper App. B)."""
    # RL chunk length C (paper: 10) and per-step action dim d
    # (smolvla_base / SO-100: 6).
    chunk_len: int = 10
    action_dim: int = 6
    # Dim of proprioceptive state s^p appended to the RL token.
    proprio_dim: int = 6
    rl_token_dim: int = 512

    # 2-layer MLP hidden 256 for most tasks; 3-layer 512 for screw task.
    hidden_dim: int = 256
    n_layers: int = 2

    # Fixed Gaussian std of the actor ("small fixed standard deviation").
    action_std: float = 0.05
    # Reference-action dropout probability during training (paper: 50%).
    ref_dropout: float = 0.5

    n_critics: int = 2  # TD3-style double Q, min for targets

@dataclass
class OnlineRLConfig:
    """Stage 2 training loop (Algorithm 1 + App. B)."""
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    discount: float = 0.97  # per control step; chunk backup uses gamma^C
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    tau: float = 0.005  # polyak for target critics (TD3)
    # BC-regularizer weight beta in Eq. 5 (task-dependent in the paper).
    bc_beta: float = 1.0

    batch_size: int = 256
    # Update-to-data ratio: gradient steps per environment *chunk* collected
    # (paper: UTD 5 counted in transitions; stride-2 subsampling stores
    # C/stride transitions per chunk, so per-chunk updates = utd * C/stride).
    utd: int = 5
    critic_updates_per_actor_update: int = 2

    # Replay buffer
    buffer_capacity: int = 200_000
    subsample_stride: int = 2

    # Rollout
    warmup_env_steps: int = 2_000  # N_warm: prefill with base VLA rollouts
    total_env_steps: int = 100_000
    max_episode_steps: int = 400

    seed: int = 0
    device: str = "cuda"