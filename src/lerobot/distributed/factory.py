#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""The `Accelerator` factory — the only place accelerate gets configured.

`torchrun` is the launcher; every accelerate parameter comes from `TrainPipelineConfig`
(`cfg.parallelism` + `cfg.accelerator`) so a run is reproducible from its `train_config.json`
alone. `accelerate launch` without a `--config_file` remains equivalent (it only sets rendezvous
env vars in that mode); the yaml flow is superseded.
"""

import logging
import os
from typing import TYPE_CHECKING

from lerobot.configs.parallelism import world_size_from_env
from lerobot.configs.train import TrainPipelineConfig

if TYPE_CHECKING:
    from accelerate import Accelerator

    from lerobot.policies.pretrained import PreTrainedPolicy

#: accelerate's own hook for pinning `state.device`, read by `PartialState.__init__` before
#: `set_device()` would otherwise fall back to the default card.
_TORCH_DEVICE_ENV = "ACCELERATE_TORCH_DEVICE"

# Env vars through which `accelerate launch --config_file` (or a stray shell) would configure
# accelerate behind the config system's back, making train_config.json lie about what ran.
_ACCELERATE_ENV_VARS = (
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_PARALLELISM_CONFIG",
    "ACCELERATE_GRADIENT_ACCUMULATION_STEPS",
    # Set by `pin_device_from_config` below, from `--policy.device` / `--reward_model.device`.
    # Inherited from a shell it would silently move a run to another card.
    _TORCH_DEVICE_ENV,
)
_ENV_OVERRIDE = "LEROBOT_ALLOW_ACCELERATE_ENV"


def guard_against_env_interference() -> None:
    """Hard-error when accelerate-configuring env vars are set.

    A silently env-overridden "reproducible" config is worse than a stop: users migrating from
    the old `accelerate launch --config_file fsdp.yaml` flow get a precise error instead of a
    config that lies. Set LEROBOT_ALLOW_ACCELERATE_ENV=1 to acknowledge and proceed.

    Raises:
        RuntimeError: If any accelerate-configuring environment variable is set and the
            LEROBOT_ALLOW_ACCELERATE_ENV override is not.
    """
    if os.environ.get(_ENV_OVERRIDE):
        return
    offending = sorted(name for name in _ACCELERATE_ENV_VARS if name in os.environ)
    if offending:
        raise RuntimeError(
            f"Accelerate-configuring environment variables are set: {', '.join(offending)}. "
            "LeRobot manages accelerate exclusively through TrainPipelineConfig "
            "(--parallelism.* / --accelerator.*); launch with plain torchrun and remove these "
            "variables (the `accelerate launch --config_file` flow is superseded), or set "
            f"{_ENV_OVERRIDE}=1 to acknowledge that they may override your config."
        )


def pin_device_from_config(cfg: TrainPipelineConfig) -> None:
    """Make an indexed `--policy.device=cuda:1` actually decide where the run lands.

    `make_policy` honours the setting (`policy.to(cfg.device)`), but `accelerator.prepare()`
    immediately moves the model to `accelerator.device`, and accelerate's own default for a
    single-process run is the index-less `torch.device("cuda")` — card 0. So the config was
    obeyed for a few milliseconds and then overruled, and a run pinned away from a busy GPU
    OOMed on it anyway. `ACCELERATE_TORCH_DEVICE` is the documented hook: `PartialState`
    reads it in `__init__`, which makes `set_device()` return early instead of falling back.

    Only for single-process runs. Under `torchrun` the device is per-rank (LOCAL_RANK), and
    a config-level index would collapse every rank onto one card; the index is ignored with
    a warning there. Select the *set* of cards for a distributed run with
    `CUDA_VISIBLE_DEVICES`, which is what maps local ranks onto physical GPUs.

    Args:
        cfg (TrainPipelineConfig): The training config; `cfg.trainable_config.device` is the
            requested device, whether the run trains a policy or a reward model.
    """
    import torch

    requested = getattr(cfg.trainable_config, "device", None)
    if not requested:
        return
    device = torch.device(requested)
    if device.index is None:
        # "cuda" / "cpu" / "mps": no card was named, so accelerate's default is the answer.
        return

    if world_size_from_env() > 1:
        # Rank 0 only: the same misconfiguration would otherwise print once per rank. This
        # runs before `init_logging()`, so it reaches stderr through logging's lastResort
        # handler — which is why it is a warning and the success path stays silent.
        if os.environ.get("LOCAL_RANK", "0") != "0":
            return
        logging.warning(
            f"Ignoring device index in device='{requested}': under a multi-process launch each "
            "rank takes its own card from LOCAL_RANK, so a config-level index would put every "
            "rank on the same GPU. Use CUDA_VISIBLE_DEVICES to choose which cards the run sees."
        )
        return

    if device.type == "cuda" and torch.cuda.is_available():
        count = torch.cuda.device_count()
        if device.index >= count:
            raise ValueError(
                f"device='{requested}' asks for CUDA device {device.index}, but only {count} "
                f"{'is' if count == 1 else 'are'} visible to this process. Indices are relative "
                "to CUDA_VISIBLE_DEVICES, so a masked run renumbers its cards from 0."
            )
        # Also make it the *current* device, so the index-less "cuda" that torch and
        # safetensors resolve lazily (see resolve_safetensors_device) lands here too.
        torch.cuda.set_device(device)

    # Not logged here: `init_logging()` runs *after* the accelerator is built and clears the
    # root handlers, so anything said now would vanish. train() reports the resolved device.
    os.environ[_TORCH_DEVICE_ENV] = str(device)


def make_accelerator(cfg: TrainPipelineConfig) -> "Accelerator":
    """Resolve the topology against the launched world and build the `Accelerator`.

    Must run once per process, before any other component needs the device or the process
    group (`Accelerator.__init__` initializes both and builds the device mesh).

    Args:
        cfg (TrainPipelineConfig): The full training config; `cfg.parallelism` is resolved in
            place against the launched world size and `cfg.accelerator` builds the result.

    Returns:
        Accelerator: The configured accelerator, with device and process group initialized.

    Raises:
        ValueError: If `cfg.checkpoint_format` requires DCP but the topology resolved to a
            non-sharded run.
    """
    guard_against_env_interference()
    # Before the Accelerator is built: `PartialState.__init__` reads the pin exactly once,
    # and the state is a process-wide singleton.
    pin_device_from_config(cfg)
    cfg.parallelism.resolve(world_size_from_env())
    # The parse-time format check ran against the declared degrees, where the dp_shard=-1
    # sentinel counts as sharded; it may resolve to an unsharded run (e.g. -1 at world size 1).
    # Re-check against the concrete degrees so the recorded format never lies about the
    # artifacts a checkpoint will actually contain.
    if cfg.checkpoint_format.wants_dcp and not cfg.parallelism.is_sharded:
        raise ValueError(
            f"checkpoint_format={cfg.checkpoint_format.value} requires a sharded run, but the "
            f"topology resolved to a non-sharded one (dp_replicate={cfg.parallelism.dp_replicate}, "
            f"dp_shard={cfg.parallelism.dp_shard}); non-sharded checkpoints are always safetensors."
        )
    return cfg.accelerator.build(
        cfg.parallelism,
        cpu=cfg.trainable_config.device == "cpu",
    )


def set_fsdp_wrap_modules(accelerator: "Accelerator", policy: "PreTrainedPolicy") -> None:
    """Resolve the FSDP wrap-unit class names onto the plugin before `accelerator.prepare()`.

    Resolution order: user override (`--accelerator.fsdp.wrap_modules`, already on the plugin)
    -> the policy's `_fsdp_wrap_modules` declaration -> hard error. Root-only wrapping — the
    silent default when no wrap source exists — is never accepted: it quietly forfeits all
    sharding memory savings.

    No-op for the size-based policy (`--accelerator.fsdp.min_num_params`), which needs no class
    names, and for non-sharded runs (no fsdp plugin).

    Args:
        accelerator (Accelerator): The accelerator whose FSDP plugin receives the wrap-unit
            class names.
        policy (PreTrainedPolicy): The trainable whose class may declare `_fsdp_wrap_modules`.

    Raises:
        ValueError: If sharded class-based wrapping is configured but neither a user override
            nor a policy declaration supplies wrap-unit class names.
    """
    plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if plugin is None or plugin.min_num_params:
        return
    if plugin.transformer_cls_names_to_wrap:  # user override, set at build time
        return
    # getattr, not attribute access: non-policy trainables (no `_fsdp_wrap_modules` attribute)
    # must reach the actionable error below, not an AttributeError.
    declared = getattr(type(policy), "_fsdp_wrap_modules", None)
    if not declared:
        raise ValueError(
            f"Policy '{type(policy).__name__}' declares no FSDP wrap units. Sharded training "
            "requires wrap-unit class names: set --accelerator.fsdp.wrap_modules='[\"MyBlock\"]' "
            "(or --accelerator.fsdp.min_num_params for a size-based policy), or declare "
            "`_fsdp_wrap_modules` on the policy class."
        )
    plugin.transformer_cls_names_to_wrap = list(declared)
