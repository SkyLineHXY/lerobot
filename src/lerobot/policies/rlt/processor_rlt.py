"""SmolVLA checkpoint loading helpers.

Unlike the PI05 reproduction (which vendored a newer lerobot pi05 and needed
an import-compat shim), SmolVLA ships with the installed lerobot 0.5.1, so
this module only wraps checkpoint loading. Note that with
``load_vlm_weights=True`` in the checkpoint config, constructing the policy
downloads/loads the SmolVLM2 backbone from the HF cache first, then overlays
the fine-tuned policy weights from the checkpoint safetensors.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_smolvla_policy(checkpoint: str, device: str = "cuda", dtype: str | None = None):
    """Load a SmolVLA policy checkpoint (e.g. ``smolvla_base``) onto `device`.

    dtype: None keeps the checkpoint's mixed layout (bf16 VLM/expert weights,
    fp32 projections); "bfloat16"/"float32" casts the whole policy.
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    if dtype == "bfloat16":
        policy = policy.to(torch.bfloat16)
    elif dtype == "float32":
        policy = policy.to(torch.float32)
    policy = policy.to(device)
    policy.config.device = str(device)
    policy.eval()
    return policy


def load_stage1_processors(stage1_dir: str | Path, device: str = "cuda"):
    """Load the pre/post-processors saved next to the stage-1 RL token.

    Returns ``(preprocessor, postprocessor)`` or ``(None, None)`` when the
    directory predates processor persistence — callers should treat that as a
    hard error for real-robot runs, where a normalisation mismatch is silent
    and ruinous.
    """
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.utils.constants import (
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )

    stage1_dir = Path(stage1_dir)
    pre_cfg = stage1_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    if not pre_cfg.is_file():
        return None, None

    overrides = {"device_processor": {"device": str(device)}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=stage1_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=stage1_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor

