"""Fit the RL action bound from a dataset's action quantiles.

Writes ``rl_action_bounds.json`` into a stage-1 output directory, next to
``rl_token.pt``. Stage 2 loads it and clamps executed chunks elementwise
instead of at a scalar +-1.

    lerobot-rlt-fit-action-bounds \\
        --dataset lerobot/libero_goal_image \\
        --stage1 outputs/train/rl_token/libero_goal

The mean/std used to express the bound in normalized units come from the
stage-1 processors, so the file is only valid for the policy those processors
were fitted with — the same pairing `check_stage1_matches_policy` enforces.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from lerobot.policies.rlt.action_bounds import DEFAULT_MARGIN, fit_action_bounds

logger = logging.getLogger(__name__)

NORMALIZER_GLOB = "policy_preprocessor_step_*_normalizer_processor.safetensors"


def load_action_stats(stage1_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(stage1_dir.glob(NORMALIZER_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No normalizer stats in {stage1_dir}; expected a file matching {NORMALIZER_GLOB}. "
            "Stage 1 saves them next to rl_token.pt."
        )
    stats = load_file(files[0])
    missing = [k for k in ("action.mean", "action.std") if k not in stats]
    if missing:
        raise KeyError(f"{files[0].name} has no {missing}")
    return (
        stats["action.mean"].flatten().numpy(),
        stats["action.std"].flatten().numpy(),
    )


def load_dataset_actions(repo_id: str, root: str | None) -> np.ndarray:
    """Every action frame of the dataset, raw and un-normalized, as (n, d)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    table = dataset.hf_dataset.with_format("numpy")["action"]
    return np.asarray(table, dtype=np.float64).reshape(len(table), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, help="dataset repo_id the policy was trained on")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--stage1", required=True, help="stage-1 output dir holding rl_token.pt")
    parser.add_argument(
        "--quantile",
        type=float,
        default=1.0,
        help="percent trimmed from each tail; openpi-RLT uses 1.0 (q01/q99)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="headroom beyond the quantiles; must exceed 1.0",
    )
    args = parser.parse_args()

    stage1 = Path(args.stage1)
    if not stage1.is_dir():
        raise NotADirectoryError(f"{stage1} is not a directory")

    mean, std = load_action_stats(stage1)
    actions = load_dataset_actions(args.dataset, args.dataset_root)
    if actions.shape[1] != mean.shape[0]:
        raise ValueError(
            f"dataset actions are {actions.shape[1]}-dim but the stage-1 stats are "
            f"{mean.shape[0]}-dim; --dataset does not match --stage1"
        )

    bounds = fit_action_bounds(actions, mean, std, quantile=args.quantile, margin=args.margin)
    path = bounds.save(stage1)

    normalized = (actions - mean) / std
    outside = np.mean((normalized < bounds.low.numpy()) | (normalized > bounds.high.numpy())) * 100
    scalar_outside = np.mean(np.abs(normalized) > 1.0) * 100
    print(f"fitted from {len(actions)} frames of {args.dataset}")
    print(f"{'dim':>4} {'low':>9} {'high':>9}")
    for i, (lo, hi) in enumerate(zip(bounds.low.tolist(), bounds.high.tolist(), strict=True)):
        print(f"{i:>4} {lo:9.3f} {hi:9.3f}")
    print(f"\nclipped by these bounds: {outside:.2f}%  (a scalar +-1 would clip {scalar_outside:.2f}%)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
