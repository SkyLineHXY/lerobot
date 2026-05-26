#!/usr/bin/env python3

import dataclasses
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots.franka_gen_gripper.umi_dataset_transform import apply_umi_sample_relative_transform
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    inside_slurm,
)


# ──────────────────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UMITrainConfig(TrainPipelineConfig):
    """在 TrainPipelineConfig 基础上增加 UMI world_flange 变换开关。

    绝大多数字段与 TrainPipelineConfig 完全相同；唯一新增字段：

    umi_transform (bool):
        True（默认）：在每个 batch 送入 preprocessor 之前调用
        ``apply_umi_sample_relative_transform``，将数据集中的绝对法兰位姿
        action 转为相对推理时刻的增量（sample-relative）。
        False：跳过变换，适用于已经存储 delta 格式的旧数据集（ee_at_t0_flange）。
    """
    umi_transform: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# 验证
# ──────────────────────────────────────────────────────────────────────────────

def _validate_umi_dataset(dataset, cfg: UMITrainConfig) -> None:
    """检查数据集格式是否与 umi_transform 设置相容。"""
    features = dataset.meta.features
    action_shape = features.get("action", {}).get("shape", ())
    state_shape = features.get("observation.state", {}).get("shape", ())
    has_umi_pose = "observation.umi_pose" in features

    if cfg.umi_transform:
        if not has_umi_pose:
            raise ValueError(
                "umi_transform=True 但数据集中不存在 'observation.umi_pose' 字段。\n"
                "请使用 --pose-format=world_flange 重新转换 mcap 数据集，\n"
                "或者通过 --umi_transform=false 关闭 UMI 变换（旧格式数据集）。"
            )
        if action_shape and action_shape[-1] != 7:
            logging.warning(
                f"umi_transform=True 但 action 维度为 {action_shape[-1]}，"
                "world_flange 格式应为 7D（pos3+rotvec3+gripper）。"
            )
        if state_shape and state_shape[-1] != 1:
            logging.warning(
                f"observation.state 维度为 {state_shape[-1]}，"
                "world_flange 格式应为 1D（gripper_width）。"
            )
        logging.info(
            f"[UMI] 数据集格式验证通过：action={action_shape}，"
            f"state={state_shape}，observation.umi_pose={has_umi_pose}"
        )
    else:
        if has_umi_pose:
            logging.warning(
                "umi_transform=False，但数据集包含 'observation.umi_pose'。"
                "该字段将被忽略（不会送入策略）。"
            )
        logging.info(
            f"[UMI] umi_transform=False，跳过 sample-relative 变换。"
            f" action={action_shape}，state={state_shape}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 单步更新（与 lerobot_train.py 完全相同，便于独立使用）
# ──────────────────────────────────────────────────────────────────────────────

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """执行单步前向 + 反向 + 优化器更新，返回 metrics 和 policy 输出字典。"""
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)

    accelerator.backward(loss)

    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# ──────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ──────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def train(cfg: UMITrainConfig, accelerator: Accelerator | None = None):
    """UMI 训练主流程，在标准 lerobot 训练循环基础上注入 sample-relative 变换。"""
    cfg.validate()

    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ── 数据集 ──
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main_process:
        dataset = make_dataset(cfg)

    # UMI 专项验证
    if is_main_process:
        _validate_umi_dataset(dataset, cfg)

    # ── 策略 ──
    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    # ── Preprocessor ──
    processor_kwargs: dict = {}
    postprocessor_kwargs: dict = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # ── 优化器 ──
    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"UMI sample-relative transform: {cfg.umi_transform}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        effective_bs = cfg.batch_size * accelerator.num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {accelerator.num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # ── DataLoader ──
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    # ── Metrics ──
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        progbar = tqdm(
            total=cfg.steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info(
            f"Start UMI offline training on a fixed dataset, "
            f"effective batch size: {cfg.batch_size * accelerator.num_processes}"
        )

    # ── 训练循环 ──
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)

        # UMI sample-relative 变换：绝对法兰位姿 → 相对当前 EE 的增量
        # 必须在 preprocessor（归一化）之前执行，保证 SE(3) 运算基于原始尺度
        if cfg.umi_transform:
            batch = apply_umi_sample_relative_transform(batch)

        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        if is_main_process:
            progbar.update(1)
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        progbar.close()
        logging.info("End of UMI training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
