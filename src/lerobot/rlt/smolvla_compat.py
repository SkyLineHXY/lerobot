"""SmolVLA checkpoint loading helpers.

Unlike the PI05 reproduction (which vendored a newer lerobot pi05 and needed
an import-compat shim), SmolVLA ships with the installed lerobot 0.5.1, so
this module only wraps checkpoint loading. Note that with
``load_vlm_weights=True`` in the checkpoint config, constructing the policy
downloads/loads the SmolVLM2 backbone from the HF cache first, then overlays
the fine-tuned policy weights from the checkpoint safetensors.
"""

from __future__ import annotations

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

