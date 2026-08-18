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
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # `from_pretrained` loads the safetensors straight onto `config.device`, which
    # comes from the checkpoint's own config.json ("cuda" = device 0). Without
    # overriding it first, asking for cuda:1 still allocates on cuda:0 and only
    # moves the weights afterwards — on a busy box that is an OOM, not a slowdown.
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config)
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


# Only the stats that actually steer the robot: the state the VLA is conditioned
# on and the action space its output is mapped back through.
_CHECKED_STATS = (
    "observation.state.mean",
    "observation.state.std",
    "action.mean",
    "action.std",
)


def check_stage1_matches_policy(stage1_dir: str | Path, checkpoint: str | Path, atol: float = 1e-5):
    """Fail when the stage-1 processors were fitted on a different dataset.

    Stage 2 normalises observations and un-normalises actions with the stats saved
    next to the RL token, while the policy was trained with the stats in its own
    checkpoint. Point `rl_token` at a stage-1 run over another dataset and nothing
    errors: the arm moves, the loop logs, and the policy silently scores 0 because
    every action it emits is rescaled and offset on the way out. Measured on a
    libero_10 stage-1 dir driving a libero_goal policy: 0/10 versus 4/10.

    Returns the list of mismatched stat names (empty when consistent); raises when
    they differ beyond `atol`. Skipped with a warning when either side has no local
    normalizer file, e.g. a checkpoint given as a bare Hub repo id.
    """
    import logging

    from safetensors.torch import load_file

    logger = logging.getLogger(__name__)
    pattern = "policy_preprocessor_step_*_normalizer_processor.safetensors"
    stage1_files = sorted(Path(stage1_dir).glob(pattern))
    policy_files = sorted(Path(checkpoint).glob(pattern))
    if not stage1_files or not policy_files:
        logger.warning(
            "Could not compare normalisation stats between %s and %s (no local normalizer file); "
            "a dataset mismatch here is silent and ruinous, so verify them by hand.",
            stage1_dir,
            checkpoint,
        )
        return []

    a, b = load_file(stage1_files[0]), load_file(policy_files[0])
    mismatched = [
        key
        for key in _CHECKED_STATS
        if key in a
        and key in b
        and (a[key].shape != b[key].shape or (a[key] - b[key]).abs().max().item() > atol)
    ]
    if mismatched:
        raise ValueError(
            f"Stage-1 processors in {stage1_dir} were fitted on different data than the policy in "
            f"{checkpoint}: {mismatched} differ. Stage 2 would un-normalize every action with the "
            f"wrong statistics and the policy would score ~0 without any error being raised. Point "
            f"`rl_token` at the stage-1 run trained on this checkpoint."
        )
    return mismatched

