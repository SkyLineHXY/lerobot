"""Chunk-level actor-critic for RLT (paper Sec. IV-B, Eq. 3-5, App. B).

Ported field-for-field from openpi-RLT `rlt_online_rl/src/rlt_online_rl/networks.py`
and `trainer.py`, so that a run of this file and a run of the reference differ in
the environment and the VLA, not in the algorithm.

* Actor: Gaussian over action chunks with a small fixed std, conditioned on the
  RL state x = (z_rl, s^p) and the VLA reference chunk. The mean is emitted
  **directly** — the reference is an input feature, not a base to add a residual
  to. What keeps the policy near the VLA is the BC term plus 50% reference
  dropout, not a box constraint.
* Critic: twin Q functions with target networks, min over the pair for targets;
  chunk-level C-step backup with a fixed gamma^C.
"""
from __future__ import annotations

import copy
from enum import IntEnum

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_rlt import ActorCriticConfig, OnlineRLConfig


class TransitionSource(IntEnum):
    """Who drove a step. Mirrors openpi-RLT `replay.py::TransitionSource`."""

    BASE = 0
    RL = 1
    HUMAN = 2
    MIXED = 3


def _xavier_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.GELU(),
        ]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    net = nn.Sequential(*layers)
    _xavier_(net)
    return net


class _ChunkEncoder(nn.Module):
    """concat[LN(Wz z), tanh(LN(Wp p)), tanh(LN(Wc chunk))].

    Shared by actor and critic; the third pathway takes the reference chunk for
    the actor and the action chunk for a Q head. `x` arrives as the single
    concatenated RL state the buffer and the parameter mirror carry, and is
    split here so nothing upstream has to know about the two projections.
    """

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.z_dim = cfg.rl_token_dim
        self.proprio_dim = cfg.proprio_dim
        self.z_proj = nn.Linear(cfg.rl_token_dim, cfg.z_proj_dim)
        self.proprio_proj = nn.Linear(cfg.proprio_dim, cfg.proprio_proj_dim)
        self.chunk_proj = nn.Linear(cfg.chunk_len * cfg.action_dim, cfg.ref_proj_dim)
        self.z_norm = nn.LayerNorm(cfg.z_proj_dim, elementwise_affine=False)
        self.proprio_norm = nn.LayerNorm(cfg.proprio_proj_dim, elementwise_affine=False)
        self.chunk_norm = nn.LayerNorm(cfg.ref_proj_dim, elementwise_affine=False)
        self.out_dim = cfg.z_proj_dim + cfg.proprio_proj_dim + cfg.ref_proj_dim
        _xavier_(self)

    def forward(self, x: Tensor, chunk: Tensor) -> Tensor:
        z = x[:, : self.z_dim]
        proprio = x[:, self.z_dim : self.z_dim + self.proprio_dim]
        chunk_flat = chunk.reshape(chunk.shape[0], -1)
        return torch.cat(
            [
                self.z_norm(self.z_proj(z)),
                torch.tanh(self.proprio_norm(self.proprio_proj(proprio))),
                torch.tanh(self.chunk_norm(self.chunk_proj(chunk_flat))),
            ],
            dim=-1,
        )


class ChunkActor(nn.Module):
    """pi_theta(a_{1:C} | x, a~_{1:C}) = N(mu_theta(x, a~), sigma^2 I)  (Eq. 4).

    The paper and both reference repositories emit `mu` directly. Reference
    dropout can therefore force the actor to reconstruct a complete action from
    the multimodal RL state. `residual_scale > 0` retains the former bounded-edit
    parameterisation strictly as an ablation; it is not the default algorithm.
    """

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _ChunkEncoder(cfg)
        self.trunk = _mlp(
            self.encoder.out_dim, cfg.hidden_dim, cfg.chunk_len * cfg.action_dim, cfg.n_layers
        )
        if cfg.residual_scale > 0:
            nn.init.zeros_(self.trunk[-1].weight)
            nn.init.zeros_(self.trunk[-1].bias)

    def mu(self, x: Tensor, ref_chunk: Tensor) -> Tensor:
        """x: (B, rl_token_dim + proprio_dim); ref_chunk: (B, C, d)."""
        out = self.trunk(self.encoder(x, ref_chunk))
        out = out.reshape(-1, self.cfg.chunk_len, self.cfg.action_dim)
        s = self.cfg.residual_scale
        if s <= 0:
            return out
        return ref_chunk + s * torch.tanh(out / s)

    def sample(
        self,
        x: Tensor,
        ref_chunk: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        mu = self.mu(x, ref_chunk)
        if deterministic:
            return mu
        return mu + self.cfg.action_std * torch.randn_like(mu)

    def apply_ref_dropout(self, ref_chunk: Tensor) -> Tensor:
        """Zero the reference for a random subset of the batch (training only)."""
        if self.cfg.ref_dropout <= 0:
            return ref_chunk
        keep = (
                torch.rand(ref_chunk.shape[0], 1, 1, device=ref_chunk.device)
                >= self.cfg.ref_dropout
        ).float()
        return ref_chunk * keep


class _QHead(nn.Module):
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.encoder = _ChunkEncoder(cfg)
        self.trunk = _mlp(self.encoder.out_dim, cfg.hidden_dim, 1, cfg.n_layers)

    def forward(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        return self.trunk(self.encoder(x, action_chunk)).squeeze(-1)


class ChunkCritic(nn.Module):
    """Ensemble of Q_psi(x, a_{1:C}) heads (twin by default)."""

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.qs = nn.ModuleList([_QHead(cfg) for _ in range(cfg.n_critics)])

    def forward(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        """Returns (n_critics, B)."""
        return torch.stack([q(x, action_chunk) for q in self.qs], dim=0)

    def min_q(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        return self.forward(x, action_chunk).min(dim=0).values


class RLTAgent:
    """Owns actor/critic/targets and implements the Algorithm 1 updates."""
    def __init__(self, cfg: OnlineRLConfig, device: str | torch.device = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = ChunkActor(cfg.ac).to(self.device)
        self.critic = ChunkCritic(cfg.ac).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self._update_count = 0
        self._warmup_end_update: int | None = None
        self._gammas = (cfg.discount ** torch.arange(cfg.ac.chunk_len)).float().to(self.device)
        self._gamma_c = cfg.discount**cfg.ac.chunk_len

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(self, x: Tensor, ref_chunk: Tensor, deterministic: bool = False) -> Tensor:
        """Inference: the reference is always provided (no dropout)."""
        self.actor.eval()
        a = self.actor.sample(x, ref_chunk, deterministic=deterministic)
        self.actor.train()
        return a

    # ------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def diagnostics(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Probes that separate "RL is working" from "the loss curves look fine".

        Sparse-reward chunk RL fails silently in ways the losses do not show:
        a critic that ignores its action argument still drives `critic_loss` to
        zero, and an actor that has drifted off the VLA manifold still reports a
        small `bc_penalty`. Each probe below answers one of those.
        """
        x, ref, a_buf = batch["x"], batch["ref"], batch["action"]
        a_pi = self.actor.mu(x, ref)
        # Reference dropout is only useful if its zero-reference branch can
        # reconstruct a sensible command from (z_rl, proprio).  The ordinary
        # BC curve sees only the stochastic mixture used for optimisation and
        # cannot distinguish "the RL state carries the action" from "the
        # actor copies the reference whenever it is present".  Evaluate both
        # branches deterministically on the same rows so that failure of the
        # dropout objective is visible without changing the training loss.
        a_zero_ref = self.actor.mu(x, torch.zeros_like(ref))
        q_pi = self.critic.min_q(x, a_pi)
        q_buf = self.critic.min_q(x, a_buf)
        q_ref = self.critic.min_q(x, ref)
        # Does Q depend on the action at all? Perturb by the actor's own
        # exploration scale: near-zero movement means the policy gradient is
        # noise and any improvement is coming from the BC term alone.
        noise = max(self.cfg.ac.action_std, 1e-3) * torch.randn_like(a_pi)
        q_sensitivity = (q_pi - self.critic.min_q(x, a_pi + noise)).abs().mean()

        delta = a_pi - ref
        zero_ref_delta = a_zero_ref - ref
        out = {
            "q_actor": q_pi.mean().item(),
            "q_behavior": q_buf.mean().item(),
            "q_ref": q_ref.mean().item(),
            # The critic's own verdict on whether the RL policy beats the frozen
            # VLA. If this never rises above 0, online RL has found nothing.
            "adv_actor_vs_ref": (q_pi - q_ref).mean().item(),
            "adv_actor_vs_behavior": (q_pi - q_buf).mean().item(),
            "q_action_sensitivity": q_sensitivity.item(),
            "actor_ref_dist": delta.pow(2).sum(-1).sqrt().mean().item(),
            "actor_ref_max": delta.abs().max().item(),
            # Evo-RLT calls the first quantity `ref_dropped_mse`.  Keep the
            # more explicit names here and also report how much supplying the
            # reference changes the actor output.  A large zero-ref error with
            # a small full-ref error means the actor ignored z_rl despite the
            # nominal 50% dropout rate.
            "actor_ref_mse": delta.pow(2).mean().item(),
            "actor_zero_ref_mse": zero_ref_delta.pow(2).mean().item(),
            "actor_ref_conditioning_mse": (a_pi - a_zero_ref).pow(2).mean().item(),
        }

        # With a terminal-only reward the critic's whole job is telling
        # "near success" from "not". If these two never separate, `z_rl` carries
        # no task progress and no amount of UTD will help.
        rewarded = batch["rewards"].sum(dim=-1) > 0
        if rewarded.any() and not rewarded.all():
            out["q_rewarded"] = q_buf[rewarded].mean().item()
            out["q_unrewarded"] = q_buf[~rewarded].mean().item()
            out["q_reward_gap"] = out["q_rewarded"] - out["q_unrewarded"]

        # The critic against the return the data actually earned. `critic_loss`
        # goes to zero on a Q that has propagated nothing — every non-terminal
        # target is then just gamma^C times its own near-zero successor, which
        # is self-consistent and completely wrong. This is the probe that tells
        # the two apart, so it is worth the extra column it costs.
        valid = batch["mc_valid"] > 0
        if valid.sum() > 1:
            q_v, mc_v = q_buf[valid], batch["mc_return"][valid]
            out["mc_mean"] = mc_v.mean().item()
            out["q_vs_mc"] = (q_v - mc_v).mean().item()
            std = q_v.std(unbiased=False) * mc_v.std(unbiased=False)
            if std > 0:
                out["q_mc_corr"] = (
                    ((q_v - q_v.mean()) * (mc_v - mc_v.mean())).mean() / std
                ).item()
        return out

    # -------------------------------------------------------------- updates
    def update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        cfg = self.cfg
        x, a, rewards, x_next, ref_next, done = (
            batch["x"],
            batch["action"],
            batch["rewards"],  # (B, C), zero-padded past the end of the window
            batch["x_next"],
            batch["ref_next"],
            batch["done"],
        )

        with torch.no_grad():
            # a' ~ pi_target(. | x', a~'). The actor's own fixed std doubles as
            # TD3 target smoothing; the reference is always provided, since that
            # is the policy actually deployed (App. B).
            a_next = self.actor_target.sample(x_next, ref_next, deterministic=False)
            q_next = self.critic_target.min_q(x_next, a_next)
            if cfg.target_q_clip > 0:
                # The only reward is +1 at a terminal step, so no state can be
                # worth more than 1; anything above that is bootstrap drift, not
                # signal. Evo-RLT clamps for the same reason (`target_q_clip`),
                # just at a value loose enough to never bind.
                q_next = q_next.clamp(-cfg.target_q_clip, cfg.target_q_clip)
            r = (rewards * self._gammas).sum(dim=-1)
            target = r + (1.0 - done) * self._gamma_c * q_next
            if cfg.critic_mc_weight > 0:
                w = cfg.critic_mc_weight * batch["mc_valid"]
                target = (1.0 - w) * target + w * batch["mc_return"]

        q = self.critic(x, a)  # (n_critics, B)
        critic_loss = sum(F.mse_loss(q[i], target) for i in range(q.shape[0]))

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": q.mean().item(),
            "target_mean": target.mean().item(),
        }

    def _awr_weights(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Advantage weights exp(beta * A_norm) for the self-imitation term.

        The advantage baseline is `Q(x, a~)` — the critic's value of *following
        the VLA* at this state — so `A = G - V` asks the only question this data
        can answer: did the chunk that was actually executed here end better
        than the reference policy's own average outcome from the same state.

        That restriction is deliberate. Measured on the 2026-08-19 buffer, a
        ridge probe of the return from the state alone scores R^2 = 0.34 and
        *adding the executed action chunk moves it to 0.3378* — the action is
        not identifiable from this data, so `-q_weight * Q1` is noise. Returns
        are identifiable, which is what makes advantage-weighted regression the
        objective that fits: it never asks the critic to rank two actions at one
        state, only to say how good the state was.

        Rows whose episode was truncated carry no usable return (`mc_valid`),
        and are held at weight 1 rather than dropped, so they still receive the
        plain BC anchor.
        """
        cfg = self.cfg
        with torch.no_grad():
            v = self.critic.min_q(batch["x"], batch["ref"])
            adv = batch["mc_return"] - v
            valid = batch["mc_valid"] > 0
            scale = adv[valid].std().clamp(min=1e-3) if valid.sum() > 1 else adv.new_tensor(1.0)
            w = torch.exp(cfg.awr_beta * adv / scale).clamp(max=cfg.awr_weight_max)
            w = torch.where(valid, w, torch.ones_like(w))
            # Normalising to mean 1 decouples the weights from `awr_weight`: the
            # term's overall pull stays put when the buffer's success rate moves.
            w = w / w.mean().clamp(min=1e-6)
        return w, {
            "awr_adv_mean": adv.mean().item(),
            "awr_weight_max": w.max().item(),
            "awr_weight_p90": w.quantile(0.9).item(),
        }

    def update_actor(
        self, batch: dict[str, Tensor], bc_weight: float, q_weight: float
    ) -> dict[str, float]:
        cfg = self.cfg
        x, ref = batch["x"], batch["ref"]
        # Reparameterised sample, matching openpi-RLT. With the direct actor,
        # dropped rows have no hidden path through which the VLA reference can
        # leak back into the output.
        a = self.actor.sample(x, self.actor.apply_ref_dropout(ref), deterministic=False)
        self.critic.requires_grad_(False)
        q1 = self.critic.qs[0](x, a)
        self.critic.requires_grad_(True)
        source = batch["source_chunk"]
        human = (source == int(TransitionSource.HUMAN)) | (source == int(TransitionSource.MIXED))
        human_f = human.to(a.dtype)
        # The reference remains the VLA proposal that conditions the deployed
        # actor. On an intervention step the supervised target is instead the
        # action actually executed by the operator, matching openpi-RLT without
        # changing the actor input distribution.
        bc_target = torch.where(human.unsqueeze(-1), batch["action"], ref)
        step_w = 1.0 + (cfg.human_bc_weight - 1.0) * human_f

        # `aligned` only tells whether the *executed behaviour chunk* came from
        # one planning decision.  The fresh VLA reference stored at every anchor
        # is a valid Eq. (5) BC target regardless, so it must not mask this term.
        # It is retained below for AWR, whose target is the executed chunk.
        row_w = batch.get("aligned")
        if row_w is None:
            row_w = a.new_ones(a.shape[0])
        sq = (a - bc_target).pow(2).mean(dim=-1)
        bc = (sq * step_w).sum() / step_w.sum().clamp(min=1.0)

        extra: dict[str, float] = {}
        awr = a.new_zeros(())
        if cfg.awr_weight > 0:
            # The target is the action that was executed — for a takeover that is
            # the operator's command, which is the whole point.
            w, extra = self._awr_weights(batch)
            w = w * row_w
            awr = ((a - batch["action"]).pow(2).mean(dim=(1, 2)) * w).sum() / w.sum().clamp(min=1e-6)

        # Paper Eq. (5): L_actor = alpha * L_BC - beta * Q. The previous
        # TD3+BC-style division by mean|Q| made beta stateful and could amplify
        # critic noise precisely when Q happened to be near zero.
        actor_q = q1.mean()
        actor_loss = bc_weight * bc + cfg.awr_weight * awr - q_weight * actor_q

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_opt.step()

        with torch.no_grad():
            per_step = (a.detach() - ref).pow(2).mean(dim=-1)
            human_share = human_f.sum().clamp(min=1.0)
            policy_share = (1.0 - human_f).sum().clamp(min=1.0)
            human_err = (a.detach() - batch["action"]).pow(2).mean(dim=-1)
            metrics = {
                "aligned_ratio": row_w.mean().item(),
                "actor_loss": actor_loss.item(),
                "actor_q": actor_q.item(),
                "actor_q_weight": q_weight,
                # Keep the three terms in their actual weighted units. Raw BC
                # and Q magnitudes are not comparable and previously hid the
                # fact that the Q term dominated some runs.
                "actor_bc_term": (bc_weight * bc).item(),
                "actor_awr_term": float(cfg.awr_weight * awr),
                "actor_q_term": (q_weight * actor_q).item(),
                "bc_penalty": bc.item(),
                "awr_penalty": float(awr),
                "bc_ref_penalty": ((per_step * (1.0 - human_f)).sum() / policy_share).item(),
                "bc_human_penalty": ((human_err * human_f).sum() / human_share).item(),
                "human_mask_ratio": human_f.mean().item(),
                **extra,
            }
        return metrics

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            for online, target in (
                (self.actor, self.actor_target),
                (self.critic, self.critic_target),
            ):
                for p, pt in zip(online.parameters(), target.parameters(), strict=True):
                    pt.lerp_(p, self.cfg.tau)

    def update(self, batch: dict[str, Tensor], warmup: bool = False) -> dict[str, float]:
        """One gradient step; actor updated every `actor_update_period`.

        Critic and actor share the batch, as in TD3, and both target networks
        move only on the steps the actor moves. `warmup` selects the BC/Q weight
        pair: the actor keeps training during warmup, it just leans harder on
        imitation while the critic is still worthless.
        """
        cfg = self.cfg
        metrics = self.update_critic(batch)
        self._update_count += 1
        if self._update_count % cfg.actor_update_period == 0:
            bc_weight, q_weight = self._actor_weights(warmup)
            metrics.update(self.update_actor(batch, bc_weight, q_weight))
            metrics |= {"bc_weight": bc_weight, "q_weight": q_weight}
            self._soft_update_targets()
        return metrics

    def _actor_weights(self, warmup: bool) -> tuple[float, float]:
        """Warmup pair, online pair, or a linear interpolation between them.

        Both reference implementations switch in one step. On the LIBERO run of
        2026-08-19 that step took the BC:Q ratio from 100:1 to 1:1 between two
        consecutive gradient steps, and `actor_ref_dist` went 0.20 -> 1.10 while
        `success20` fell 0.85 -> 0.55 and never recovered. `weight_ramp_updates`
        spreads the same endpoints over a window instead; 0 restores the step.
        """
        cfg = self.cfg
        if warmup:
            self._warmup_end_update = None
            return cfg.warmup_bc_weight, cfg.warmup_q_weight
        if cfg.weight_ramp_updates <= 0:
            return cfg.online_bc_weight, cfg.online_q_weight
        if self._warmup_end_update is None:
            self._warmup_end_update = self._update_count
        t = min((self._update_count - self._warmup_end_update) / cfg.weight_ramp_updates, 1.0)
        return (
            cfg.warmup_bc_weight + t * (cfg.online_bc_weight - cfg.warmup_bc_weight),
            cfg.warmup_q_weight + t * (cfg.online_q_weight - cfg.warmup_q_weight),
        )

    # ------------------------------------------------------------------- io
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_target.load_state_dict(sd["actor_target"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self._update_count = sd.get("update_count", 0)
