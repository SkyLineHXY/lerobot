"""Stage 1: train the RL token encoder/decoder on demonstration data (Eq. 2).

Follows Algorithm 1 lines 1-3: with the VLA frozen (alpha = 0 by default),
minimize the autoregressive reconstruction loss L_ro on task demos; with
--vla-sft-alpha > 0 the VLA is jointly fine-tuned with its flow-matching loss.

Example (LIBERO-10 demos, frozen VLA):
  python -m rlt.train_rl_token \
    --checkpoint smolvla_base \
    --dataset lerobot/libero_10 --dataset-root ~/pi05_reproduce/data/lerobot/libero_10 \
    --steps 5000 --batch-size 16 --out outputs/rl_token
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from .configs import RLTokenConfig
from .rl_token import RLTokenModule, SmolVLAPrefixExtractor
from .smolvla_compat import load_smolvla_policy

def build_dataset_and_processors(policy_config, dataset_repo: str, dataset_root: str | None):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    policy_config.input_features = input_features
    policy_config.output_features = output_features
    policy_config.validate_features()

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy_config, dataset_stats=meta.stats
    )
    delta_timestamps = {k: [0.0] for k in policy_config.input_features if k.startswith("observation.")}
    for k in policy_config.output_features:
        if k.startswith("action"):
            delta_timestamps[k] = [i / meta.fps for i in policy_config.action_delta_indices]

    dataset = LeRobotDataset(
        dataset_repo, root=dataset_root, delta_timestamps=delta_timestamps, video_backend="pyav"
    )
    return dataset, preprocessor, postprocessor

def train(args):
    device = args.device
    cfg = RLTokenConfig(
        d_model=args.d_model,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        vla_sft_alpha=args.vla_sft_alpha,
    )

    print(f"[stage1] loading SmolVLA from {args.checkpoint} ...")
    policy = load_smolvla_policy(args.checkpoint, device=device, dtype=args.dtype)
    dataset, preprocessor, _ = build_dataset_and_processors(
        policy.config, args.dataset, args.dataset_root
    )
    extractor = SmolVLAPrefixExtractor(policy)
    finetune_vla = cfg.vla_sft_alpha > 0
    policy.requires_grad_(finetune_vla)
    policy.train(finetune_vla)

    rl_token = RLTokenModule(cfg).to(device)
    n_params = sum(p.numel() for p in rl_token.parameters())
    print(f"[stage1] RL token module: {n_params / 1e6:.1f}M params, d_model={cfg.d_model}")

    params = list(rl_token.parameters())
    if finetune_vla:
        params += [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    step, t0 = 0, time.time()
    data_iter = iter(loader)
    while step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = preprocessor(batch)
        # L_ro treats VLA embeddings with stop-gradient (Eq. 2), so extraction
        # never needs the VLA graph even when alpha > 0; the SFT gradient path
        # goes through policy.forward below.
        feats = extractor.extract(batch)
        z, mask = extractor.select_tokens(feats, cfg.use_image_tokens_only)
        loss_ro, _ = rl_token.reconstruction_loss(z, mask)

        loss = loss_ro
        log = {"loss_ro": loss_ro.item()}
        if finetune_vla:
            loss_vla, _ = policy.forward(batch)
            loss = loss + cfg.vla_sft_alpha * loss_vla
            log["loss_vla"] = loss_vla.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
        opt.step()
        step += 1

        if step % args.log_freq == 0 or step == 1:
            speed = step / (time.time() - t0)
            msg = " ".join(f"{k}={v:.5f}" for k, v in log.items())
            print(f"[stage1] step {step}/{cfg.steps} {msg} ({speed:.2f} it/s)", flush=True)

        if step % args.save_freq == 0 or step == cfg.steps:
            ckpt = {"rl_token": rl_token.state_dict(), "config": vars(cfg), "step": step}
            torch.save(ckpt, out_dir / "rl_token.pt")
            if finetune_vla:
                torch.save(policy.state_dict(), out_dir / "smolvla_sft.pt")

        print(f"[stage1] done; saved to {out_dir / 'rl_token.pt'}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="smolvla_base")
    p.add_argument("--dataset", default="lerobot/libero_10")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--out", default="outputs/rl_token")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--vla-sft-alpha", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=20)
    p.add_argument("--save-freq", type=int, default=500)
    train(p.parse_args())

if __name__ == "__main__":
    main()

