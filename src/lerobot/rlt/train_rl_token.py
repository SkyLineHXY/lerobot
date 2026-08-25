"""Stage 1: train the RL token encoder/decoder on demonstration data (Eq. 2).

Follows Algorithm 1 lines 1-3: with the VLA frozen (alpha = 0 by default),
minimize the autoregressive reconstruction loss L_ro on task demos; with
--vla-sft-alpha > 0 the VLA is jointly fine-tuned with its flow-matching loss.

Configuration is a draccus YAML tree (see `examples/rlt/`); any field can be
overridden on the command line:

  python -m lerobot.rlt.train_rl_token --config_path examples/rlt/_templates/rl_token.yaml \
    --dataset_root ~/data/lerobot/libero_10 --rl_token.steps 8000
"""
import time
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.policies.rlt import (
    RLTokenModule,
    RLTokenTrainConfig,
    SmolVLAPrefixExtractor,
    load_smolvla_policy,
)
from lerobot.rlt.metrics import make_wandb_logger


def select_task_episode_indices(
    meta,
    task_index: int | None,
    episode_start: int = 0,
    num_episodes: int | None = None,
) -> list[int] | None:
    """Resolve a dataset task id to episode ids without decoding any frames."""
    if episode_start < 0 or (num_episodes is not None and num_episodes <= 0):
        raise ValueError("dataset episode start must be >=0 and count must be >0")
    if task_index is None and episode_start == 0 and num_episodes is None:
        return None
    if task_index is None:
        episodes = [int(ep["episode_index"]) for ep in meta.episodes]
    else:
        names = {
            str(name)
            for name, row in meta.tasks.iterrows()
            if int(row["task_index"]) == task_index
        }
        if not names:
            available = sorted(int(v) for v in meta.tasks["task_index"].tolist())
            raise ValueError(f"dataset task_index {task_index} is absent; available={available}")
        episodes = [
            int(ep["episode_index"])
            for ep in meta.episodes
            if names.intersection(str(task) for task in ep["tasks"])
        ]
    if not episodes:
        raise ValueError(f"dataset task_index {task_index} has no episodes")
    stop = None if num_episodes is None else episode_start + num_episodes
    selected = episodes[episode_start:stop]
    if not selected:
        raise ValueError(
            f"dataset episode slice [{episode_start}:{stop}] is empty after task filtering "
            f"({len(episodes)} episodes available)"
        )
    return selected


def build_dataset_and_processors(
    policy_config,
    dataset_repo: str,
    dataset_root: str | None,
    dataset_task_index: int | None = None,
    dataset_episode_start: int = 0,
    dataset_num_episodes: int | None = None,
):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    meta = LeRobotDatasetMetadata(dataset_repo, root=dataset_root)
    episodes = select_task_episode_indices(
        meta, dataset_task_index, dataset_episode_start, dataset_num_episodes
    )
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
        dataset_repo,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    return dataset, preprocessor, postprocessor

def save_stage1(out_dir, rl_token, cfg, step, preprocessor, postprocessor) -> None:
    """Write the RL-token weights next to the processors used to train them.

    Stage 2 and the real robot must normalise observations and un-normalise
    actions with exactly these statistics; shipping them alongside the weights
    is what keeps z_rl and the reference chunks from silently drifting apart.
    """
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    torch.save(
        {"rl_token": rl_token.state_dict(), "config": vars(cfg), "step": step},
        out_dir / "rl_token.pt",
    )
    preprocessor.save_pretrained(out_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(out_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")


def save_finetuned_vla(out_dir, policy, preprocessor, postprocessor) -> None:
    """Persist the jointly fine-tuned VLA as a loadable checkpoint directory.

    A bare state_dict would be unusable downstream: `train_online.py` loads the
    VLA through `SmolVLAPolicy.from_pretrained`, so the SFT result has to be a
    real checkpoint directory or it is silently ignored.
    """
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    preprocessor.save_pretrained(out_dir, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json")
    postprocessor.save_pretrained(out_dir, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json")


def train(train_cfg: RLTokenTrainConfig):
    device = train_cfg.device
    cfg = train_cfg.rl_token
    torch.manual_seed(train_cfg.seed)
    wb = make_wandb_logger(train_cfg, job_type="rlt_token", step_key="step")

    print(f"[stage1] loading SmolVLA from {train_cfg.checkpoint} ...")
    policy = load_smolvla_policy(train_cfg.checkpoint, device=device, dtype=train_cfg.dtype)
    dataset, preprocessor, postprocessor = build_dataset_and_processors(
        policy.config,
        train_cfg.dataset,
        train_cfg.dataset_root,
        train_cfg.dataset_task_index,
        train_cfg.dataset_episode_start,
        train_cfg.dataset_num_episodes,
    )
    if dataset.episodes is not None:
        print(
            f"[stage1] task {train_cfg.dataset_task_index}: "
            f"{dataset.num_episodes} episodes, {dataset.num_frames} frames"
        )
    extractor = SmolVLAPrefixExtractor(policy)
    finetune_vla = cfg.vla_sft_alpha > 0
    policy.requires_grad_(finetune_vla)

    policy.eval()

    rl_token = RLTokenModule(cfg).to(device)
    n_params = sum(p.numel() for p in rl_token.parameters())
    print(f"[stage1] RL token module: {n_params / 1e6:.1f}M params, d_model={cfg.d_model}")

    params = list(rl_token.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    # The VLA is optimised separately: L_ro and L_vla have disjoint gradient
    # graphs (Eq. 2 stop-gradients the embeddings), so sharing one optimizer
    vla_params = [p for p in policy.parameters() if p.requires_grad] if finetune_vla else []
    vla_opt = vla_sched = None
    if finetune_vla:
        vla_opt = torch.optim.AdamW(vla_params, lr=cfg.vla_lr, weight_decay=cfg.weight_decay)
        vla_sched = torch.optim.lr_scheduler.CosineAnnealingLR(vla_opt, T_max=cfg.steps)
        n_vla = sum(p.numel() for p in vla_params)
        print(f"[stage1] joint VLA fine-tuning: {n_vla / 1e6:.0f}M params @ lr={cfg.vla_lr}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=True,
    )

    out_dir = Path(train_cfg.out)
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
        feats = extractor.extract(batch)
        z, mask = extractor.select_tokens(
            feats,
            drop_language=cfg.drop_language_tokens,
            image_only=cfg.use_image_tokens_only,
        )
        loss_ro, _, ro_metrics = rl_token.reconstruction_loss(z, mask)

        loss = loss_ro
        log = {"loss_ro": loss_ro.item(), **ro_metrics}
        if finetune_vla:
            policy.train()
            loss_vla, _ = policy.forward(batch)
            policy.eval()
            loss = loss + cfg.vla_sft_alpha * loss_vla
            log["loss_vla"] = loss_vla.item()

        opt.zero_grad(set_to_none=True)
        if vla_opt is not None:
            vla_opt.zero_grad(set_to_none=True)
        loss.backward()

        log["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm))
        opt.step()
        sched.step()
        if vla_opt is not None:
            log["vla_grad_norm"] = float(
                torch.nn.utils.clip_grad_norm_(vla_params, cfg.grad_clip_norm)
            )
            vla_opt.step()
            vla_sched.step()
        step += 1

        if step % train_cfg.log_freq == 0 or step == 1:
            speed = step / (time.time() - t0)
            # Teacher forcing lets the decoder solve M-1 of M positions without
            # reading z_rl, so L_ro alone cannot tell a working run from a
            # collapsed one. This is the probe that can.
            log |= rl_token.bottleneck_usage(z, mask)
            log["lr"] = sched.get_last_lr()[0]
            log["it_per_s"] = speed
            wb.log({f"stage1/{k}": v for k, v in log.items()}, step)
            msg = " ".join(f"{k}={v:.5f}" for k, v in log.items())
            print(f"[stage1] step {step}/{cfg.steps} {msg} ({speed:.2f} it/s)", flush=True)

        if step % train_cfg.save_freq == 0 or step == cfg.steps:
            save_stage1(out_dir, rl_token, cfg, step, preprocessor, postprocessor)
            if finetune_vla:
                save_finetuned_vla(out_dir / "smolvla_sft", policy, preprocessor, postprocessor)

    wb.finish()
    print(f"[stage1] done; saved to {out_dir / 'rl_token.pt'}")

@parser.wrap()
def main(cfg: RLTokenTrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
