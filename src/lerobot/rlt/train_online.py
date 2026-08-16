"""Stage 2: online RL with the RL token (Algorithm 1).

Rollout-and-update loop:
  * warmup: execute the base VLA reference chunks for N_warm env steps while
    the learner already trains the critic on that data,
  * afterwards the actor refines the VLA chunks; every executed chunk is
    stored with stride-2 subsampling and the learner paces itself at
    `utd` gradient steps per stored transition (2 critic updates per actor
    update),
  * a human may take over at any chunk boundary; the taken-over actions
    replace both the executed action and the stored reference (paper Sec. V),
  * with `--critical-phase` the episode starts under the base VLA and the
    operator presses `r` to hand control to the RL policy, so data collection
    concentrates on the precise segment that decides success.

Rollout and learning run concurrently: the learner owns the agent in its own
thread and publishes actor weights, the rollout thread reads them through a
mirror at chunk boundaries. A synchronous loop would stall a real arm for the
whole gradient burst at every boundary. See `README.md` for what that
concurrency does and does not buy — the two threads still share one CUDA
context, so the learner is throttled rather than free.

This module is the wiring only: the env factory is in `envs/`, the chunk
execution loop in `rollout.py`, the gradient loop in `learner.py`.

Configuration is a draccus YAML tree (see `examples/rlt/`); any field can be
overridden on the command line:

  # mock smoke run
  python -m lerobot.rlt.train_online --config_path examples/rlt/mock_online.yaml

  # LIBERO simulation, pre-hardware validation
  python -m lerobot.rlt.train_online --config_path examples/rlt/libero_online.yaml \
    --env.task_id=3

  # real Piper arm, leader-arm interventions
  python -m lerobot.rlt.train_online --config_path examples/rlt/piper_online.yaml \
    --env.dry_run=true
"""
import logging
import time
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.policies.rlt import (
    RLTAgent,
    RLTController,
    RLTokenConfig,
    RLTokenModule,
    RLTOnlineTrainConfig,
    load_smolvla_policy,
)

from .envs import make_chunk_env
from .learner import ActorMirror, LearnerThread
from .replay_buffer import ChunkReplayBuffer
from .rollout import RolloutWorker
from .teleop.keys import KeyboardEventListener

logger = logging.getLogger(__name__)


def load_rl_token(path: str | Path, device: str) -> tuple[RLTokenModule, RLTokenConfig]:
    """Load the stage-1 RL token from either its directory or the .pt itself."""
    path = Path(path)
    if path.is_dir():
        path = path / "rl_token.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = RLTokenConfig(**{
        k: v for k, v in ckpt["config"].items() if k in RLTokenConfig.__dataclass_fields__
    })
    module = RLTokenModule(cfg)
    module.load_state_dict(ckpt["rl_token"])
    module.to(device).eval().requires_grad_(False)
    return module, cfg


def build_agent_and_controller(cfg: RLTOnlineTrainConfig):
    device = cfg.device
    print(f"[stage2] loading SmolVLA from {cfg.checkpoint} ...")
    policy = load_smolvla_policy(cfg.checkpoint, device=device, dtype=cfg.dtype)
    rl_token, rt_cfg = load_rl_token(cfg.rl_token, device)
    cfg.rl.ac.rl_token_dim = rt_cfg.d_model

    agent = RLTAgent(cfg.rl, device=device)
    controller = RLTController(
        policy,
        rl_token,
        agent,
        chunk_len=cfg.rl.ac.chunk_len,
        action_dim=cfg.rl.ac.action_dim,
        proprio_dim=cfg.rl.ac.proprio_dim,
        drop_language=rt_cfg.drop_language_tokens,
        image_only=rt_cfg.use_image_tokens_only,
        num_inference_steps=cfg.num_inference_steps,
    )
    return policy, agent, controller


def train(cfg: RLTOnlineTrainConfig):
    device = cfg.device
    rl = cfg.rl
    torch.manual_seed(cfg.seed)

    _policy, agent, controller = build_agent_and_controller(cfg)

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_dim = rl.ac.rl_token_dim + rl.ac.proprio_dim
    buffer = ChunkReplayBuffer(
        capacity=rl.buffer_capacity,
        x_dim=x_dim,
        chunk_len=rl.ac.chunk_len,
        action_dim=rl.ac.action_dim,
        discount=rl.discount,
        stride=rl.subsample_stride,
        device=device,
        seed=cfg.seed,
    )
    if cfg.resume_buffer:
        buffer.load(cfg.resume_buffer)
        print(f"[stage2] resumed buffer with {len(buffer)} transitions")

    keys = KeyboardEventListener(backend=cfg.keyboard_backend, discard_key=cfg.discard_key)
    env = None
    try:
        # Build the env *before* putting stdin in cbreak mode: connecting the
        # leader can drop into the interactive calibration flow, whose `input()`
        # and Enter-detection do not work once the terminal is raw.
        env = make_chunk_env(cfg, keys, policy=_policy)
        keys.start()

        mirror = ActorMirror(rl, device)
        controller.agent = mirror  # rollout reads published weights, not live ones
        worker = RolloutWorker(env, controller, buffer, rl.subsample_stride, keys=keys)

        learner = LearnerThread(agent, buffer, rl)
        mirror.sync(learner)
        learner.start()

        env_steps, episodes, ep_results = 0, 0, []
        worker.reset(critical_phase=cfg.critical_phase)
        t0 = time.time()

        while env_steps < rl.total_env_steps:
            learner.raise_if_failed()
            if keys.should_quit():
                print("[stage2] operator requested stop.")
                break

            warmup = env_steps < rl.warmup_env_steps
            learner.allow_actor_updates = not warmup

            # Critical-phase handover: run the base VLA until the operator
            # presses `r`, then let the RL policy take over for the rest of the
            # episode (paper Sec. V, "Targeted improvement of critical phases").
            if cfg.critical_phase and not worker.rl_engaged and keys.poll_handover():
                worker.rl_engaged = True
                print("[stage2] handover -> RL policy")

            use_actor = (not warmup) and worker.rl_engaged
            mirror.sync(learner)
            prev_steps = env_steps
            n_steps, ep_done, ep_success, intervened = worker.run_chunk(use_actor=use_actor)
            env_steps += n_steps

            if keys.poll_discard():
                # `env_steps` is not rolled back: the arm really did move, and
                # the budget is a wear/time budget, not a data counter.
                dropped = buffer.discard_episode()
                print(f"[stage2] episode discarded by operator ({dropped} transitions dropped)")
                worker.reset(critical_phase=cfg.critical_phase)
                continue

            if ep_done:
                episodes += 1
                ep_results.append(1.0 if ep_success else 0.0)
                worker.reset(critical_phase=cfg.critical_phase)

            if env_steps // cfg.log_freq != prev_steps // cfg.log_freq:
                recent = ep_results[-20:]
                sr = sum(recent) / max(len(recent), 1)
                speed = env_steps / (time.time() - t0)
                m = " ".join(f"{k}={v:.4f}" for k, v in learner.metrics().items())
                print(
                    f"[stage2] steps={env_steps} eps={episodes} buffer={len(buffer)} "
                    f"success20={sr:.2f} {'(warmup) ' if warmup else ''}"
                    f"{'(interv) ' if intervened else ''}{m} ({speed:.1f} steps/s)",
                    flush=True,
                )

            if env_steps // cfg.save_freq != prev_steps // cfg.save_freq:
                _checkpoint(agent, buffer, out_dir, learner)
    finally:
        # Each step gets its own try/except: an exception raised while tearing
        # down masks whatever actually ended the run.
        for label, teardown in (
            ("keyboard", keys.stop),
            ("learner", lambda: (learner.stop(), learner.join(timeout=5.0))),
            ("env", (env.close if env is not None else lambda: None)),
        ):
            try:
                teardown()
            except Exception:
                logger.exception("[stage2] %s teardown failed", label)

    _checkpoint(agent, buffer, out_dir)
    print(f"[stage2] done; {episodes} episodes, saved to {out_dir / 'rlt_agent.pt'}")


def _checkpoint(agent, buffer, out_dir: Path, learner=None) -> None:
    """Write the agent and buffer with the learner held between gradient steps.

    `agent.state_dict()` runs on the main thread; without pausing it can land
    mid-optimizer-step and persist an actor and critic that disagree.
    """
    if learner is not None and not learner.pause():
        logger.warning("[stage2] learner did not pause in time; checkpoint may be torn")
    try:
        torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")
        buffer.save(out_dir / "replay_buffer.pt")
    finally:
        if learner is not None:
            learner.resume()


@parser.wrap()
def main(cfg: RLTOnlineTrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
