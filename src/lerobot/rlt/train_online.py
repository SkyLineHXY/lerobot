"""Stage 2: online RL with the RL token (Algorithm 1).

Rollout-and-update loop:
  * warmup: execute the base VLA reference chunks for N_warm env steps,
  * afterwards the actor refines the VLA chunks; every executed chunk is
    stored with stride-2 subsampling, and `utd * (C / stride)` gradient steps
    are performed per chunk (2 critic updates per actor update),
  * optional human interventions replace both the executed action and the
    stored reference (paper Sec. V).

Example (mock env smoke run):
  python -m rlt.train_online --checkpoint smolvla_base \
    --rl-token outputs/rl_token/rl_token.pt --mock-env \
    --total-env-steps 2000 --warmup-env-steps 400
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .actor_critic import RLTAgent
from .configs import ActorCriticConfig,OnlineRLConfig,RLTokenConfig
from .envs import MockManipEnv
from .replay_buffer import ChunkRecord, ChunkReplayBuffer
from .rl_token import RLTokenModule
from .rlt_policy import RLTController
from .smolvla_compat import load_smolvla_policy

def load_rl_token(path: str | Path, device: str) -> tuple[RLTokenModule, RLTokenConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RLTokenConfig(**{
        k: v for k, v in ckpt["config"].items() if k in RLTokenConfig.__dataclass_fields__
    })
    module = RLTokenModule(cfg)
    module.load_state_dict(ckpt["rl_token"])
    module.to(device).eval().requires_grad_(False)
    return module, cfg


class RolloutWorker:
    """Executes chunks in a single env and assembles ChunkRecords."""
    def __init__(self, env, controller: RLTController, buffer: ChunkReplayBuffer, stride: int):
        self.env = env
        self.controller = controller
        self.buffer = buffer
        self.stride = stride
        self.device = next(controller.rl_token.parameters()).device
        self.obs = None
        self.ep_steps = 0
        self.ep_return = 0.0

    def reset(self) -> None:
        self.obs = self.env.reset()
        self.ep_steps = 0
        self.ep_return = 0.0
        self.buffer.start_episode()

    def run_chunk(self, use_actor: bool, deterministic: bool = False):
        """Plan at the chunk boundary, execute up to C steps, store the record.

        Returns (n_steps, episode_done, episode_success).
        """
        c = self.controller.chunk_len
        batch = self.env.obs_to_batch([self.obs], self.device)
        plan = self.controller.plan_chunk(batch, use_actor=use_actor, deterministic=deterministic)

        intervention = self.env.get_intervention()

        if intervention is not None:
            actions = intervention.to(plan["action_chunk"][0])
            ref_full = actions[-1:].repeat(plan["ref_full"].shape[1], 1).clone()
            ref_full[:c] = actions
        else:
            actions = plan["action_chunk"][0]
            ref_full = plan["ref_full"][0]

        rewards = torch.zeros(c)
        inter_obs: list[dict] = []

        success, truncated, done_step = False, False, None
        for j in range(c):
            self.obs, r, success = self.env.step(actions[j])
            rewards[j] = r
            self.ep_steps += 1
            self.ep_return += r
            if success:
                done_step = j + 1
                break
            if self.ep_steps >= self.env.max_episode_steps:
                truncated = True
                done_step = j + 1
                break
            if (j + 1) % self.stride == 0 and (j + 1) < c:
                inter_obs.append(self.obs)

        xs = [plan["x"][0].cpu()]
        if inter_obs:
            xb = self.env.obs_to_batch(inter_obs, self.device)
            xs.extend(self.controller.compute_x(xb).cpu())


        rec = ChunkRecord(
            xs=torch.stack(xs),
            actions=actions.detach().cpu(),
            rewards=rewards,
            ref_full=ref_full.detach().cpu(),
            done=success,
            done_step=done_step,
        )
        self.buffer.add_chunk(rec)
        if truncated and not success:
            self.buffer.end_episode()

        n_steps = done_step if done_step is not None else c
        return n_steps, success or truncated, success

def train(args):
    device = args.device
    cfg = OnlineRLConfig(
        ac=ActorCriticConfig(
            chunk_len=args.chunk_len,
            action_dim=args.action_dim,
            proprio_dim=args.proprio_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            action_std=args.action_std,
        ),
        bc_beta=args.bc_beta,
        utd=args.utd,
        batch_size=args.batch_size,
        warmup_env_steps=args.warmup_env_steps,
        total_env_steps=args.total_env_steps,
        max_episode_steps=args.max_episode_steps,
        device=device,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)

    print(f"[stage2] loading SmolVLA from {args.checkpoint} ...")
    policy = load_smolvla_policy(args.checkpoint, device=device, dtype=args.dtype)
    rl_token, rt_cfg = load_rl_token(args.rl_token, device)
    cfg.ac.rl_token_dim = rt_cfg.d_model

    agent = RLTAgent(cfg, device=device)
    controller = RLTController(
        policy,
        rl_token,
        agent,
        chunk_len=cfg.ac.chunk_len,
        action_dim=cfg.ac.action_dim,
        proprio_dim=cfg.ac.proprio_dim,
        num_inference_steps=args.num_inference_steps,
    )

    if not args.mock_env:
        raise NotImplementedError(
            "Only --mock-env is wired up in this repo; implement ChunkEnv for a "
            "real robot or simulator and pass it here."
        )

    env = MockManipEnv(
        action_dim=cfg.ac.action_dim,
        max_episode_steps=cfg.max_episode_steps,
        seed=cfg.seed,
    )
    x_dim = cfg.ac.rl_token_dim + cfg.ac.proprio_dim
    buffer = ChunkReplayBuffer(
        capacity=cfg.buffer_capacity,
        x_dim=x_dim,
        chunk_len=cfg.ac.chunk_len,
        action_dim=cfg.ac.action_dim,
        discount=cfg.discount,
        stride=cfg.subsample_stride,
        device=device,
    )
    worker = RolloutWorker(env, controller, buffer, stride=cfg.subsample_stride)

    updates_per_chunk = cfg.utd * (cfg.ac.chunk_len // cfg.subsample_stride)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    env_steps, episodes, ep_results = 0, 0, []
    worker.reset()
    t0 = time.time()
    metrics: dict[str, float] = {}
    
    while env_steps < cfg.total_env_steps:
        warmup = env_steps < cfg.warmup_env_steps
        prev_steps = env_steps
        n_steps, ep_done, ep_success = worker.run_chunk(use_actor=not warmup)
        env_steps += n_steps
        
        if ep_done:
            episodes += 1
            ep_results.append(1.0 if ep_success else 0.0)
            worker.reset()

        if not warmup and len(buffer) >= cfg.batch_size:
            for _ in range(updates_per_chunk):
                metrics = agent.update(lambda: buffer.sample(cfg.batch_size))

        
        if env_steps // args.log_freq != prev_steps // args.log_freq:
            recent = ep_results[-20:]
            sr = sum(recent) / max(len(recent), 1)
            speed = env_steps / (time.time() - t0)
            m = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(
                f"[stage2] steps={env_steps} eps={episodes} buffer={len(buffer)} "
                f"success20={sr:.2f} {m} ({speed:.1f} steps/s)",
                flush=True,
            )
            
        if env_steps // args.save_freq != prev_steps // args.save_freq:
            torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")

        
    torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")
    print(f"[stage2] done; {episodes} episodes, saved to {out_dir/'rlt_agent.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="smolvla_base")
    p.add_argument("--rl-token", default="outputs/rl_token/rl_token.pt")
    p.add_argument("--out", default="outputs/rlt_online")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16"])
    p.add_argument("--mock-env", action="store_true")
    p.add_argument("--chunk-len", type=int, default=10)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--proprio-dim", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--action-std", type=float, default=0.05)
    p.add_argument("--bc-beta", type=float, default=1.0)
    p.add_argument("--utd", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--warmup-env-steps", type=int, default=2000)
    p.add_argument("--total-env-steps", type=int, default=100_000)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=2000)
    train(p.parse_args())


if __name__ == "__main__":
    main()