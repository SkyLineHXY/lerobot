"""Chunk-level actor-critic for RLT (paper Sec. IV-B, Eq. 3-5, App. B).

* Actor: Gaussian over action chunks with small fixed std, conditioned on the
  RL state x = (z_rl, s^p) and the VLA reference chunk (pass-through), with
  50% reference dropout during training (zeros on the input pathway only).
  The mean is parameterised as ``mu = a~ + clamp(residual, +-max_residual)``
  with a zero-initialised output layer, so before any gradient step the RL
  policy *is* the base VLA. This keeps the very first real-robot chunk safe
  and turns online RL into the local action editing the paper describes.
* Critic: ensemble of two Q functions; TD3-style target networks, min over
  the ensemble for target values; chunk-level C-step backup with an exponent
  of gamma^k where k is the number of steps actually executed in the window.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .configuration_rlt import ActorCriticConfig, OnlineRLConfig


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class ChunkActor(nn.Module):
    """pi_theta(a_{1:C} | x, a~_{1:C}) = N(mu_theta(x, a~), sigma^2 I)  (Eq. 4).

    ``mu = a~ + clamp(f(x, a~_masked), +-max_residual)``. Reference dropout
    masks the *input* pathway only: the residual is still added to the full
    reference, so a dropped sample forces the network to produce a correction
    from the RL token alone rather than to regenerate the whole chunk from
    scratch.

    There is deliberately no bound on the action itself, matching openpi-RLT
    and Evo-RLT. `max_residual` already confines the action to a box around the
    VLA reference, which is the property that matters, and it says so in the
    VLA's own normalised units — so nothing here assumes what those units mean.
    An absolute bound would have to, and a wrong guess is silent: a scalar +-1
    is right for a policy normalised to a bounded range (openpi-RLT
    re-normalises with q01/q99 for exactly that reason) but is one standard
    deviation under SmolVLA's mean/std, where it truncates 27% of every LIBERO
    chunk. Physical limits belong downstream, where they are unambiguous:
    robosuite clips to +-1 raw in `Controller.scale_action`, and the Piper env
    has `rate_limit_joints`.
    """

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg
        chunk_dim = cfg.chunk_len * cfg.action_dim
        in_dim = cfg.rl_token_dim + cfg.proprio_dim + chunk_dim
        self.net = _mlp(in_dim, cfg.hidden_dim, chunk_dim, cfg.n_layers)

        # Zero-init the output layer => residual == 0 => mu == reference.
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def residual(self, x: Tensor, ref_in: Tensor) -> Tensor:
        """Bounded correction predicted from x and the (possibly masked) ref."""
        ref_flat = ref_in.reshape(ref_in.shape[0], -1)
        out = self.net(torch.cat([x, ref_flat], dim=-1))
        out = out.reshape(-1, self.cfg.chunk_len, self.cfg.action_dim)
        if self.cfg.max_residual > 0:
            out = out.clamp(-self.cfg.max_residual, self.cfg.max_residual)
        return out

    def mu(self, x: Tensor, ref_chunk: Tensor, ref_in: Tensor | None = None) -> Tensor:
        """x: (B, rl_token_dim + proprio_dim); ref_chunk: (B, C, d).

        `ref_in` is the (dropout-masked) reference fed to the network; it
        defaults to `ref_chunk`, i.e. inference behaviour where the reference
        is always provided (paper App. B).
        """
        res = self.residual(x, ref_chunk if ref_in is None else ref_in)
        return ref_chunk + res

    def sample(
        self,
        x: Tensor,
        ref_chunk: Tensor,
        deterministic: bool = False,
        ref_in: Tensor | None = None,
    ) -> Tensor:
        mu = self.mu(x, ref_chunk, ref_in=ref_in)
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

class ChunkCritic(nn.Module):
    """Ensemble of Q_psi(x, a_{1:C}) heads."""
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        chunk_dim = cfg.chunk_len * cfg.action_dim
        in_dim = cfg.rl_token_dim + cfg.proprio_dim + chunk_dim
        self.qs = nn.ModuleList(
            [_mlp(in_dim, cfg.hidden_dim, 1, cfg.n_layers) for _ in range(cfg.n_critics)]
        )
    def forward(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        """Returns (n_critics, B)."""
        a_flat = action_chunk.reshape(action_chunk.shape[0], -1)
        inp = torch.cat([x, a_flat], dim=-1)
        return torch.stack([q(inp).squeeze(-1) for q in self.qs], dim=0)
    def min_q(self, x: Tensor, action_chunk: Tensor) -> Tensor:
        return self.forward(x, action_chunk).min(dim=0).values


class RLTAgent:
    """Owns actor/critic/targets and implements the Algorithm 1 updates."""
    def __init__(self, cfg: OnlineRLConfig, device: str | torch.device = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = ChunkActor(cfg.ac).to(self.device)
        self.critic = ChunkCritic(cfg.ac).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self._update_count = 0
        gamma_c = cfg.discount**cfg.ac.chunk_len
        self.q_max = cfg.target_q_clip_scale / max(1.0 - gamma_c, 1e-6)

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(self, x: Tensor, ref_chunk: Tensor, deterministic: bool = False) -> Tensor:
        """Inference: the reference is always provided (no dropout)."""
        self.actor.eval()
        a = self.actor.sample(x, ref_chunk, deterministic=deterministic)
        self.actor.train()
        return a

    # -------------------------------------------------------------- updates
    def update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        cfg = self.cfg
        x, a, r, x_next, ref_next, done = (
            batch["x"],
            batch["action"],
            batch["reward_disc"],  # sum_{t'=1}^{C} gamma^{t'-1} r_{t'}
            batch["x_next"],
            batch["ref_next"],
            batch["done"],
        )
        # Windows that ran fewer than C steps must bootstrap with gamma^k, not
        # gamma^C, or the critic discounts a gap that never elapsed.
        steps = batch.get("actual_steps")

        with torch.no_grad():
            # a' ~ pi_theta(. | x', a~') with the reference *always* provided:
            # that is the policy actually deployed (App. B), so it is the one
            # the backup must evaluate. TD3 target smoothing replaces the
            # actor's own exploration noise here.
            a_next = self.actor.mu(x_next, ref_next)
            noise = (cfg.target_noise_std * torch.randn_like(a_next)).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            a_next = a_next + noise

            q_next = self.critic_target.min_q(x_next, a_next).clamp(0.0, self.q_max)
            if steps is None:
                gamma_k = cfg.discount**cfg.ac.chunk_len
            else:
                gamma_k = cfg.discount**steps
            target = r + gamma_k * (1.0 - done) * q_next

        q = self.critic(x, a)  # (n_critics, B)
        critic_loss = F.mse_loss(q, target.unsqueeze(0).expand_as(q))

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        with torch.no_grad():
            for p, pt in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                pt.lerp_(p, cfg.tau)


        return {
            "critic_loss": critic_loss.item(),
            "q_mean": q.mean().item(),
            "target_mean": target.mean().item(),
        }

    def update_actor(self, batch: dict[str, Tensor]) -> dict[str, float]:
        cfg = self.cfg
        x, ref = batch["x"], batch["ref"]

        # Reference dropout on the input pathway only (Eq. 5 still uses the
        # true reference as the BC target, and the residual is still applied
        # on top of the unmasked reference).
        ref_in = self.actor.apply_ref_dropout(ref)
        a = self.actor.mu(x, ref, ref_in=ref_in)

        # Only the actor is being optimised; freezing the critic here avoids a
        # pointless gradient accumulation over its parameters.
        self.critic.requires_grad_(False)
        q = self.critic.min_q(x, a)
        self.critic.requires_grad_(True)

        # Paper Eq. 5 uses the squared L2 *norm* over the whole chunk. Summing
        # (not averaging) over C x d keeps beta on the paper's scale.
        bc = (a - ref).pow(2).sum(dim=(1, 2))
        actor_loss = (-q + cfg.bc_beta * bc).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
        self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "actor_q": q.mean().item(),
            "bc_dist": bc.mean().item(),
        }

    def update(self, batch: dict[str, Tensor], allow_actor: bool = True) -> dict[str, float]:
        """One gradient step; actor updated every `critic_updates_per_actor_update`.

        Critic and actor share the batch, as in TD3. `allow_actor=False` keeps
        the critic learning while the actor is held at the VLA reference (used
        during the warmup phase).
        """
        metrics = self.update_critic(batch)
        self._update_count += 1
        if allow_actor and self._update_count % self.cfg.critic_updates_per_actor_update == 0:
            metrics.update(self.update_actor(batch))
        return metrics

    # ------------------------------------------------------------------- io
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "update_count": self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self._update_count = sd.get("update_count", 0)
