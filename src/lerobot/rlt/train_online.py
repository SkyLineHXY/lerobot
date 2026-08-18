"""Stage 2: online RL with the RL token (Algorithm 1).

Rollout-and-update loop:
  * warmup: execute the base VLA reference chunks for N_warm env steps while
    the learner already trains the critic on that data,
  * afterwards the actor refines the VLA chunks; every executed chunk is
    stored with stride-2 subsampling and the learner paces itself at
    `utd` gradient steps per stored transition (2 critic updates per actor
    update),
  * once warmup is over a human may take over at any point, including
    mid-chunk; the taken-over actions replace both the executed action and the
    stored reference (paper Sec. V),
  * with `--critical-phase` the episode starts under the base VLA and the
    operator presses `r` to hand control to the RL policy, so data collection
    concentrates on the precise segment that decides success,
  * in a sim env, digit keys 0-9 switch the LIBERO task of the current suite
    between episodes,
  * `view.enabled` opens an OpenCV window showing the cameras plus the RL state
    (phase, who is driving, success rate, buffer, learner metrics).

Both the window and takeover stay off for the whole warmup phase. Warmup exists
to fit the critic on what the frozen base VLA actually does, so a human-driven
chunk there would teach it the value of an action the deployed policy cannot
produce; and nobody needs to watch a policy they are not allowed to correct.
The window opens and the controls arm together, at the warmup boundary.

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
    check_stage1_matches_policy,
    load_smolvla_policy,
)
from lerobot.rlt.backends import make_backend
from lerobot.rlt.envs import make_chunk_env
from lerobot.rlt.rollout import RolloutWorker
from lerobot.rlt.teleop.keys import KeyboardEventListener
from lerobot.rlt.view import RolloutView

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
    check_stage1_matches_policy(cfg.rl_token, cfg.checkpoint)
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


def poll_task_switch(env, keys) -> int | None:
    """Digit keys 0-9 select a task of the current suite; returns the new id.

    Only the sim envs expose `set_task_id`; on hardware a digit is just an
    unclaimed key. `poll_extra` is the only legal way to read one — the listener
    drains its backend in a single pass, so a second reader would see nothing.
    """
    if not hasattr(env, "set_task_id"):
        return None
    wanted = None
    for key in keys.poll_extra():
        if len(key) == 1 and key.isdigit():
            wanted = int(key)
    if wanted is None:
        return None
    try:
        if not env.set_task_id(wanted):
            print(f"[stage2] already on task {wanted}")
            return None
    except ValueError as exc:
        print(f"[stage2] {exc}")
        return None
    print(f"[stage2] task -> {wanted}: {env.task_description}")
    return wanted


def train(cfg: RLTOnlineTrainConfig):
    rl = cfg.rl
    torch.manual_seed(cfg.seed)

    _policy, agent, controller = build_agent_and_controller(cfg)

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_dim = rl.ac.rl_token_dim + rl.ac.proprio_dim

    keys = KeyboardEventListener(backend=cfg.keyboard_backend, discard_key=cfg.discard_key)
    env = None
    backend = None
    view = None
    try:
        # Build the env *before* putting stdin in cbreak mode: connecting the
        # leader can drop into the interactive calibration flow, whose `input()`
        # and Enter-detection do not work once the terminal is raw.
        env = make_chunk_env(cfg, keys, policy=_policy)
        keys.start()

        backend = make_backend(cfg, agent, x_dim)
        controller.agent = backend.mirror  # plan with published weights, not live ones
        view = RolloutView(
            env,
            enabled=cfg.view.enabled,
            max_width=cfg.view.max_width,
            panel_height=cfg.view.panel_height,
            tile_height=cfg.view.tile_height,
            min_period_s=cfg.view.min_period_s,
        )
        worker = RolloutWorker(
            env, controller, backend.buffer, rl.subsample_stride, keys=keys, view=view
        )
        backend.start()
        print(f"[stage2] concurrency mode: {backend.mode}")
        hints = ["space=takeover", "s=success", "f=failure", f"{cfg.discard_key}=discard", "esc=quit"]
        print(
            f"[stage2] warmup: base VLA only for {rl.warmup_env_steps} env steps "
            f"(no view, no takeover)"
        )
        if hasattr(env, "set_task_id"):
            hints.append(f"0-9=task (0..{env.n_tasks - 1})")
            print(f"[stage2] task {env.task_id}: {env.task_description}")
        print(f"[stage2] operator keys: {' | '.join(hints)}")
        view.set(
            task=getattr(env, "task_description", None) or getattr(env, "task", None),
            task_id=getattr(env, "task_id", None),
            total_env_steps=rl.total_env_steps,
            max_episode_steps=env.max_episode_steps,
            hint=" | ".join(hints),
        )

        env_steps, episodes, ep_results = 0, 0, []

        def reset_episode() -> None:
            worker.reset(critical_phase=cfg.critical_phase)

        reset_episode()
        t0 = time.time()

        while env_steps < rl.total_env_steps:
            backend.check_health()
            if keys.should_quit():
                print("[stage2] operator requested stop.")
                break
            if view.quit_requested:
                print("[stage2] operator closed the view.")
                break

            if poll_task_switch(env, keys) is not None:
                backend.buffer.discard_episode()
                view.set(task=env.task_description, task_id=env.task_id)
                reset_episode()
                continue

            warmup = env_steps < rl.warmup_env_steps
            backend.set_warmup(warmup)
            worker.allow_intervention = not warmup
            view.set_active(not warmup)

            # Critical-phase handover: run the base VLA until the operator
            # presses `r`, then let the RL policy take over for the rest of the
            # episode (paper Sec. V, "Targeted improvement of critical phases").
            if cfg.critical_phase and not worker.rl_engaged and keys.poll_handover():
                worker.rl_engaged = True
                print("[stage2] handover -> RL policy")

            use_actor = (not warmup) and worker.rl_engaged
            recent = ep_results[-20:]
            view.set(
                phase="WARMUP" if warmup else ("RL" if use_actor else "VLA"),
                env_steps=env_steps,
                episodes=episodes,
                success_rate=sum(recent) / max(len(recent), 1),
                successes=int(sum(ep_results)),
                buffer=len(backend.buffer),
                fps=env_steps / max(time.time() - t0, 1e-6),
                elapsed=time.time() - t0,
                metrics=backend.metrics(),
            )
            backend.sync_mirror()
            prev_steps = env_steps
            n_steps, ep_done, ep_success, intervened = worker.run_chunk(use_actor=use_actor)
            env_steps += n_steps

            if keys.poll_discard():
                # `env_steps` is not rolled back: the arm really did move, and
                # the budget is a wear/time budget, not a data counter.
                dropped = backend.buffer.discard_episode()
                print(f"[stage2] episode discarded by operator ({dropped} dropped)")
                reset_episode()
                continue

            if ep_done:
                episodes += 1
                ep_results.append(1.0 if ep_success else 0.0)
                reset_episode()

            if env_steps // cfg.log_freq != prev_steps // cfg.log_freq:
                recent = ep_results[-20:]
                sr = sum(recent) / max(len(recent), 1)
                speed = env_steps / (time.time() - t0)
                m = " ".join(f"{k}={v:.4f}" for k, v in backend.metrics().items())
                print(
                    f"[stage2] steps={env_steps} eps={episodes} buffer={len(backend.buffer)} "
                    f"success20={sr:.2f} {'(warmup) ' if warmup else ''}"
                    f"{'(interv) ' if intervened else ''}{m} ({speed:.1f} steps/s)",
                    flush=True,
                )

            if env_steps // cfg.save_freq != prev_steps // cfg.save_freq:
                backend.checkpoint(out_dir)
    finally:
        # Each step gets its own try/except: an exception raised while tearing
        # down masks whatever actually ended the run.
        for label, teardown in (
            ("view", (view.stop if view is not None else lambda: None)),
            ("keyboard", keys.stop),
            ("backend", (backend.stop if backend is not None else lambda: None)),
            ("env", (env.close if env is not None else lambda: None)),
        ):
            try:
                teardown()
            except Exception:
                logger.exception("[stage2] %s teardown failed", label)

    if backend is not None:
        backend.checkpoint(out_dir)
    print(f"[stage2] done; {episodes} episodes, saved under {out_dir}")


@parser.wrap()
def main(cfg: RLTOnlineTrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
