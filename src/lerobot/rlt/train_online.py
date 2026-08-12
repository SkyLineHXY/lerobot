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
whole gradient burst at every boundary.

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
import time
from pathlib import Path

import torch

from lerobot.configs import parser
from lerobot.policies.rlt import (
    LiberoEnvConfig,
    MockEnvConfig,
    PiperEnvConfig,
    RLTAgent,
    RLTController,
    RLTokenConfig,
    RLTokenModule,
    RLTOnlineTrainConfig,
    load_smolvla_policy,
    load_stage1_processors,
)

from .envs import MockManipEnv
from .intervention import KeyboardEventListener
from .learner import ActorMirror, LearnerThread
from .replay_buffer import ChunkRecord, ChunkReplayBuffer


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


class RolloutWorker:
    """Executes chunks in a single env and assembles ChunkRecords."""

    def __init__(
        self,
        env,
        controller: RLTController,
        buffer: ChunkReplayBuffer,
        stride: int,
        keys: KeyboardEventListener | None = None,
    ):
        self.env = env
        self.controller = controller
        self.buffer = buffer
        self.stride = stride
        self.keys = keys
        self.device = next(controller.rl_token.parameters()).device
        self.obs = None
        self.ep_steps = 0
        self.ep_return = 0.0
        self.ep_interventions = 0
        # Critical-phase mode starts every episode under the base VLA and only
        # switches to the RL policy when the operator says so.
        self.rl_engaged = True

    def reset(self, critical_phase: bool = False) -> None:
        self.obs = self.env.reset()
        self.ep_steps = 0
        self.ep_return = 0.0
        self.ep_interventions = 0
        self.rl_engaged = not critical_phase
        self.buffer.start_episode()

    def _states_for(self, obs_list: list[dict]) -> list[torch.Tensor]:
        if not obs_list:
            return []
        batch = self.env.obs_to_batch(obs_list, self.device)
        return list(self.controller.compute_x(batch).cpu())

    def _plan(self, batch, use_actor: bool, deterministic: bool):
        return self.controller.plan_chunk(
            batch, use_actor=use_actor, deterministic=deterministic
        )

    def run_chunk(self, use_actor: bool, deterministic: bool = False):
        """Plan at the chunk boundary, execute up to C steps, store the record.

        Returns (n_steps, episode_done, episode_success, intervened).
        """
        c = self.controller.chunk_len
        batch = self.env.obs_to_batch([self.obs], self.device)
        # If the operator is already holding the leader, the VLA's action chunk
        # is guaranteed to be discarded; compute only the RL state x and skip
        # the flow-matching sampling the human would otherwise wait through.
        taking_over = self.env.intervention_pending()
        plan = (
            {"x": self.controller.compute_x(batch)}
            if taking_over
            else self._plan(batch, use_actor, deterministic)
        )

        intervention = self.env.run_intervention(c)

        if intervention is not None:
            # Human corrections replace the VLA reference too, so the actor's
            # BC term pulls toward the correction rather than toward the failed
            # VLA attempt (paper Sec. V, "Rollout").
            actions = intervention.action_chunk.to(plan["x"])
            # The buffer only ever reads ref_full[o : o+C] for o < C, so a 2C
            # horizon is all a human-authored reference needs.
            ref_full = actions[-1:].repeat(2 * c, 1).clone()
            ref_full[:c] = actions
            rewards = intervention.rewards
            step_obs = intervention.obs_list
            n_exec = intervention.n_steps
            terminated, truncated = intervention.done, intervention.truncated
            # Sparse reward model: +1 only on operator-judged success. `done`
            # also fires on the failure key, which must NOT count as a success.
            success = bool(rewards.sum() > 0)
            if step_obs:
                self.obs = step_obs[-1]
            self.ep_steps += n_exec
            self.ep_return += float(rewards.sum())
            self.ep_interventions += 1
        else:
            if taking_over:
                # Pre-check said "human", but they let go before the chunk
                # started; fall back to a full plan.
                plan = self._plan(batch, use_actor, deterministic)
            actions = plan["action_chunk"][0]
            ref_full = plan["ref_full"][0]
            rewards = torch.zeros(c)
            step_obs = []
            terminated = truncated = False
            n_exec = 0
            for j in range(c):
                self.obs, r, terminated = self.env.step(actions[j])
                rewards[j] = r
                step_obs.append(self.obs)
                n_exec = j + 1
                self.ep_steps += 1
                self.ep_return += r
                if terminated:
                    break
                if self.ep_steps >= self.env.max_episode_steps:
                    truncated = True
                    break
            success = bool(rewards.sum() > 0)

        done_step = n_exec if (terminated or truncated or n_exec < c) else None

        # RL states at offsets 0, stride, 2*stride, ... The offset-0 state came
        # for free with the plan; the rest are recomputed in a single batched
        # VLM forward after execution so they never sit in the control loop.
        # step_obs[j] is the observation *after* step j+1, so offset o lives at
        # index o-1 and is only available if the chunk ran that far.
        inter_obs = [
            step_obs[o - 1] for o in range(self.stride, c, self.stride) if o <= len(step_obs)
        ]
        xs = [plan["x"][0].cpu()] + self._states_for(inter_obs)

        rec = ChunkRecord(
            xs=torch.stack(xs),
            actions=actions.detach().cpu(),
            rewards=rewards,
            ref_full=ref_full.detach().cpu(),
            # Terminal — success *or* operator-declared failure. Both end the
            # episode with no further reward available, so both mask the
            # bootstrap; only `success` carries the +1.
            done=terminated,
            done_step=done_step,
        )
        self.buffer.add_chunk(rec)
        if truncated and not terminated:
            # A truncated window still bootstraps, so it needs the real state
            # observed after the last executed step.
            x_last = self._states_for([step_obs[-1]]) if step_obs else []
            self.buffer.end_episode(x_last[0] if x_last else None)

        return n_exec, terminated or truncated, success, intervention is not None


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


def build_env(cfg: RLTOnlineTrainConfig, keys: KeyboardEventListener):
    env_cfg, ac = cfg.env, cfg.rl.ac

    if isinstance(env_cfg, MockEnvConfig):
        return MockManipEnv(
            action_dim=ac.action_dim,
            max_episode_steps=env_cfg.max_episode_steps,
            image_size=env_cfg.image_size,
            success_eps=env_cfg.success_eps,
            prompt=env_cfg.prompt,
            seed=cfg.seed,
        )

    # Everything below runs the policy for real, so it must use exactly the
    # normalisation stage 1 was trained with.
    preprocessor, postprocessor = load_stage1_processors(cfg.rl_token, device=cfg.device)

    if isinstance(env_cfg, LiberoEnvConfig):
        from .libero_env import LiberoChunkEnv

        return LiberoChunkEnv(
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task_suite_name=env_cfg.suite,
            task_id=env_cfg.task_id,
            action_dim=ac.action_dim,
            max_episode_steps=env_cfg.max_episode_steps,
            control_mode=env_cfg.control_mode,
            observation_size=env_cfg.observation_size,
            camera_name=env_cfg.camera_name,
            seed=cfg.seed,
        )

    if not isinstance(env_cfg, PiperEnvConfig):
        raise NotImplementedError(f"Unsupported env type {env_cfg.type!r}.")

    from lerobot.robots.piper.config_piper import PIPERConfig
    from lerobot.robots.piper.piper import Piper

    from .intervention import PiperLeaderIntervention
    from .piper_env import PiperChunkEnv, build_piper_cameras

    robot_cfg = PIPERConfig(can_port=env_cfg.can_port)
    if env_cfg.cameras:
        robot_cfg.cameras = build_piper_cameras(env_cfg.cameras, env_cfg.control_hz)
    robot_cfg.max_joint_step_rad = env_cfg.max_joint_step_rad
    robot = Piper(robot_cfg)

    env = PiperChunkEnv(
        robot=robot,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        task=env_cfg.task,
        action_dim=ac.action_dim,
        max_episode_steps=env_cfg.max_episode_steps,
        control_hz=env_cfg.control_hz,
        max_joint_step_rad=env_cfg.max_joint_step_rad,
        reset_pose=env_cfg.reset_pose,
        reset_noise_rad=env_cfg.reset_noise_rad,
        keys=keys,
        dry_run=env_cfg.dry_run,
        seed=cfg.seed,
    )

    if env_cfg.teleop is not None:
        from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

        leader = PiperLeader(
            PiperLeaderConfig(port=env_cfg.teleop.port, id=env_cfg.teleop.id)
        )
        leader.connect()
        env.intervention = PiperLeaderIntervention(
            leader,
            env,
            keys,
            use_calibrated_offsets=env_cfg.teleop.use_calibrated_offsets,
            max_takeover_delta_rad=env_cfg.teleop.max_takeover_delta_rad,
        )
        # The leader entered gravity compensation on connect; park it on the
        # follower's pose so the first takeover has a defined starting point.
        env.intervention.align()
    return env


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

    keys = KeyboardEventListener()
    env = None
    try:
        # Build the env *before* putting stdin in cbreak mode: connecting the
        # leader can drop into the interactive calibration flow, whose `input()`
        # and Enter-detection do not work once the terminal is raw.
        env = build_env(cfg, keys)
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
            if keys.should_quit():
                print("[stage2] operator requested stop.")
                break

            warmup = env_steps < cfg.warmup_env_steps
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
                torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")
                buffer.save(out_dir / "replay_buffer.pt")
    finally:
        keys.stop()
        try:
            learner.stop()
            learner.join(timeout=5.0)
        except (NameError, RuntimeError):
            pass
        if env is not None:
            env.close()

    torch.save(agent.state_dict(), out_dir / "rlt_agent.pt")
    buffer.save(out_dir / "replay_buffer.pt")
    print(f"[stage2] done; {episodes} episodes, saved to {out_dir / 'rlt_agent.pt'}")


@parser.wrap()
def main(cfg: RLTOnlineTrainConfig):
    train(cfg)


if __name__ == "__main__":
    main()
