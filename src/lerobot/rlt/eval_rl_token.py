"""Diagnose whether a stage-1 RL token is a usable RL state representation.

The reconstruction loss alone does not answer the question the paper actually
cares about (Sec. IV-A): does the bottleneck "preserve task-relevant
information" while being small enough for a lightweight actor-critic? A token
can have low L_ro and still be useless for RL — for instance if it reconstructs
the visual background faithfully while collapsing everything that distinguishes
one moment of the task from another.

So this evaluates z_rl against what the downstream critic and actor actually
need, on held-out episodes:

1. **Reconstruction** — relative error on episodes never seen in training. A
   large train/held-out gap means the encoder memorised rather than compressed.
2. **Collapse** — per-dim std and the participation ratio of the covariance
   spectrum (an effective-rank estimate). A token whose variance lives in 3 of
   512 dims gives the critic almost no state to condition on.
3. **Linear decodability** — ridge probes for proprioceptive state, the VLA's
   own action chunk, and normalised episode progress t/T. Progress is the
   important one: under a sparse terminal reward the critic is essentially
   learning "how close am I to success", so if progress is not linearly
   readable from z_rl, the critic has to learn it from scratch and the sample
   efficiency the method is built around disappears.
4. **Temporal structure** — consecutive frames should be much closer in z_rl
   space than random pairs. A token that jitters step-to-step makes the TD
   target nonstationary.
5. **Baselines** — the same metrics for mean-pooled VLA embeddings and for a
   random projection of the same width. If a random projection probes as well
   as z_rl, the encoder has not learned anything the raw features lacked.

Usage:
  python -m lerobot.rlt.eval_rl_token --config_path examples/rlt/rl_token.yaml \
    --eval.n_episodes 20
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from lerobot.configs import parser
from lerobot.policies.rlt import (
    RLTokenModule,
    RLTokenTrainConfig,
    SmolVLAPrefixExtractor,
    load_smolvla_policy,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .train_rl_token import build_dataset_and_processors


@dataclass
class RLTokenEvalOptions:
    # Episodes held out from the probes' training split.
    n_episodes: int = 20
    batch_size: int = 8
    # Fraction of episodes used to fit the probes; the rest score them. The
    # split is by *episode*, not by frame — neighbouring frames are so
    # correlated that a frame-level split would report near-perfect probes for
    # any representation at all.
    train_frac: float = 0.6
    ridge_lambda: float = 1.0
    max_frames: int = 4000
    seed: int = 0


@dataclass
class RLTokenEvalConfig(RLTokenTrainConfig):
    """Stage-1 config plus the evaluation knobs; reuses `checkpoint`/`dataset`."""

    eval: RLTokenEvalOptions = field(default_factory=RLTokenEvalOptions)


# --------------------------------------------------------------------- probes
def _ridge_r2(x: Tensor, y: Tensor, split: Tensor, lam: float) -> float:
    """R^2 of a ridge probe y ~ x, fit on `split` and scored on its complement.

    Returns the fraction of held-out variance explained; <= 0 means the probe
    does no better than predicting the mean.
    """
    xtr, ytr = x[split], y[split]
    xte, yte = x[~split], y[~split]
    if xte.shape[0] < 2 or xtr.shape[0] < 2:
        return float("nan")

    # Standardise on the fit split only, then solve (X'X + lam I) w = X'Y.
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp(min=1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    xtr = torch.cat([xtr, torch.ones(xtr.shape[0], 1)], dim=1)
    xte = torch.cat([xte, torch.ones(xte.shape[0], 1)], dim=1)

    d = xtr.shape[1]
    a = xtr.T @ xtr + lam * torch.eye(d, dtype=xtr.dtype)
    w = torch.linalg.solve(a, xtr.T @ ytr)

    pred = xte @ w
    ss_res = (yte - pred).pow(2).sum()
    ss_tot = (yte - yte.mean(0, keepdim=True)).pow(2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-8))


def _participation_ratio(z: Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 of the covariance — an effective rank.

    Equals D for isotropic variance and 1 when all variance sits in one
    direction, so it reads directly as "how many dimensions are actually used".
    """
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / max(zc.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp(min=0)
    s1, s2 = ev.sum(), ev.pow(2).sum()
    return float(s1.pow(2) / s2.clamp(min=1e-12))


def _temporal_ratio(z: Tensor, episode: Tensor) -> float:
    """mean ||z_t - z_{t+1}|| / mean ||z_i - z_j|| over random pairs.

    Well below 1 means the token moves smoothly along a trajectory; near 1
    means consecutive states look as different as unrelated ones.
    """
    same = episode[:-1] == episode[1:]
    if same.sum() < 2:
        return float("nan")
    consecutive = (z[1:][same] - z[:-1][same]).norm(dim=-1).mean()
    idx = torch.randperm(z.shape[0])
    random_pairs = (z - z[idx]).norm(dim=-1).mean()
    return float(consecutive / random_pairs.clamp(min=1e-8))


def _report(name: str, z: Tensor, targets: dict[str, Tensor], split, episode, opt) -> dict:
    out = {
        "dim": z.shape[1],
        "std_mean": float(z.std(0).mean()),
        "eff_rank": _participation_ratio(z),
        "temporal_ratio": _temporal_ratio(z, episode),
    }
    for key, y in targets.items():
        out[f"r2_{key}"] = _ridge_r2(z, y, split, opt.ridge_lambda)
    width = max(len(k) for k in out)
    print(f"\n[{name}]")
    for k, v in out.items():
        print(f"  {k:<{width}}  {v:.4f}" if isinstance(v, float) else f"  {k:<{width}}  {v}")
    return out


# ----------------------------------------------------------------- collection
@torch.no_grad()
def collect(cfg: RLTokenEvalConfig):
    device, opt = cfg.device, cfg.eval
    torch.manual_seed(opt.seed)

    print(f"[eval] loading SmolVLA from {cfg.checkpoint} ...")
    policy = load_smolvla_policy(cfg.checkpoint, device=device, dtype=cfg.dtype)
    policy.eval()
    dataset, preprocessor, _ = build_dataset_and_processors(
        policy.config, cfg.dataset, cfg.dataset_root
    )
    extractor = SmolVLAPrefixExtractor(policy)

    ckpt_dir = Path(cfg.out)
    ckpt = torch.load(ckpt_dir / "rl_token.pt", map_location="cpu", weights_only=False)
    tok_cfg = type(cfg.rl_token)(**{
        k: v for k, v in ckpt["config"].items() if k in type(cfg.rl_token).__dataclass_fields__
    })
    rl_token = RLTokenModule(tok_cfg)
    rl_token.load_state_dict(ckpt["rl_token"])
    rl_token.to(device).eval()
    print(f"[eval] RL token from step {ckpt.get('step')}, d_model={tok_cfg.d_model}")

    # Episode-level split, so the probes are never scored on frames adjacent to
    # ones they were fit on.
    ep_idx = dataset.hf_dataset["episode_index"]
    ep_idx = torch.as_tensor(ep_idx) if not isinstance(ep_idx, Tensor) else ep_idx
    episodes = torch.unique(ep_idx)[: opt.n_episodes]
    frame_ids = torch.nonzero(torch.isin(ep_idx, episodes)).flatten()[: opt.max_frames]
    print(f"[eval] {len(episodes)} episodes, {len(frame_ids)} frames")

    ep_len = {int(e): int((ep_idx == e).sum()) for e in episodes}

    z_rl_all, z_pool_all, state_all, action_all, prog_all, ep_all = [], [], [], [], [], []
    recon_errs = []
    for start in range(0, len(frame_ids), opt.batch_size):
        rows = frame_ids[start : start + opt.batch_size]
        items = [dataset[int(i)] for i in rows]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if isinstance(items[0][k], Tensor)}
        batch["task"] = [it.get("task", "") for it in items]
        batch = preprocessor(batch)

        feats = extractor.extract(batch)
        z, mask = extractor.select_tokens(
            feats,
            drop_language=tok_cfg.drop_language_tokens,
            image_only=tok_cfg.use_image_tokens_only,
        )
        loss, z_rl, _ = rl_token.reconstruction_loss(z, mask)
        recon_errs.append(float(loss))

        z_rl_all.append(z_rl.float().cpu())
        # Baseline: mean-pooled raw VLA embeddings over valid tokens.
        m = mask.unsqueeze(-1).float()
        z_pool_all.append(((z * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())

        st = batch[OBS_STATE]
        state_all.append((st.squeeze(1) if st.ndim == 3 else st).float().cpu())
        act = batch.get(ACTION)
        action_all.append(act.flatten(1).float().cpu() if act is not None else torch.zeros(len(rows), 1))

        eps = ep_idx[rows]
        ep_all.append(eps)
        fi = torch.stack([torch.as_tensor(it["frame_index"]) for it in items]).float()
        prog_all.append(
            torch.stack([fi[j] / max(ep_len[int(eps[j])] - 1, 1) for j in range(len(rows))]).unsqueeze(-1)
        )

        if start % (opt.batch_size * 20) == 0:
            print(f"[eval] {start + len(rows)}/{len(frame_ids)} frames", flush=True)

    return {
        "z_rl": torch.cat(z_rl_all),
        "z_pool": torch.cat(z_pool_all),
        "state": torch.cat(state_all),
        "action": torch.cat(action_all),
        "progress": torch.cat(prog_all),
        "episode": torch.cat(ep_all),
        "recon": sum(recon_errs) / len(recon_errs),
        "episodes": episodes,
    }


def evaluate(cfg: RLTokenEvalConfig) -> dict:
    opt = cfg.eval
    data = collect(cfg)
    episode = data["episode"]

    n_fit = max(int(len(data["episodes"]) * opt.train_frac), 1)
    fit_eps = data["episodes"][:n_fit]
    split = torch.isin(episode, fit_eps)
    print(
        f"\n[eval] held-out reconstruction (normalised space): {data['recon']:.5f}"
        f"\n[eval] probe split: {int(split.sum())} fit / {int((~split).sum())} score frames"
    )

    targets = {
        "state": data["state"],
        "action_chunk": data["action"],
        "progress": data["progress"],
    }

    results = {"recon": data["recon"]}
    results["rl_token"] = _report("RL token z_rl", data["z_rl"], targets, split, episode, opt)
    results["mean_pooled"] = _report(
        "baseline: mean-pooled VLA embeddings", data["z_pool"], targets, split, episode, opt
    )
    # Random projection of the raw features to the same width: the honest
    # control for "did the encoder learn anything, or is any linear map fine?"
    g = torch.Generator().manual_seed(opt.seed)
    proj = torch.randn(data["z_pool"].shape[1], data["z_rl"].shape[1], generator=g)
    proj /= proj.shape[0] ** 0.5
    results["random_proj"] = _report(
        "baseline: random projection", data["z_pool"] @ proj, targets, split, episode, opt
    )

    rl, rnd = results["rl_token"], results["random_proj"]
    print("\n[eval] verdict")
    print(f"  progress R^2: z_rl {rl['r2_progress']:.3f} vs random {rnd['r2_progress']:.3f}")
    if rl["r2_progress"] <= rnd["r2_progress"] + 0.05:
        print("  WARNING: the token is no better than a random projection at encoding task")
        print("           progress — the critic will have to learn phase from scratch.")
    if rl["eff_rank"] < 0.05 * rl["dim"]:
        print(f"  WARNING: effective rank {rl['eff_rank']:.1f} of {rl['dim']} dims — near-collapse.")
    if rl["temporal_ratio"] > 0.7:
        print("  WARNING: consecutive frames are almost as far apart as random pairs;")
        print("           the TD target will see a jittery state.")
    return results


@parser.wrap()
def main(cfg: RLTokenEvalConfig):
    evaluate(cfg)


if __name__ == "__main__":
    main()
