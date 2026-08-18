"""The rollout half of stage 2: execute one chunk, assemble one `ChunkRecord`.

Runs on the main thread. Everything expensive it does — two VLM forwards and a
flow-matching sample per chunk boundary — shares a CUDA context with the learner
thread, so keep anything addable to the control loop out of `run_chunk`.
"""
from __future__ import annotations

import torch

from lerobot.policies.rlt import RLTController

from .replay_buffer import ChunkRecord, ChunkReplayBuffer
from .teleop.keys import KeyboardEventListener


class RolloutWorker:
    """Executes chunks in a single env and assembles ChunkRecords."""

    def __init__(
        self,
        env,
        controller: RLTController,
        buffer: ChunkReplayBuffer,
        stride: int,
        keys: KeyboardEventListener | None = None,
        view=None,
    ):
        self.env = env
        self.controller = controller
        self.buffer = buffer
        self.stride = stride
        self.keys = keys
        self.view = view
        self.device = next(controller.rl_token.parameters()).device
        self.obs = None
        self.ep_steps = 0
        self.ep_return = 0.0
        self.ep_interventions = 0
        # Display-only step counter for the stretch the human is driving; the
        # authoritative count still comes from InterventionResult.n_steps.
        self._live_steps = 0
        # Critical-phase mode starts every episode under the base VLA and only
        # switches to the RL policy when the operator says so.
        self.rl_engaged = True
        # Cleared during warmup: that phase runs the frozen base VLA to fit the
        # critic on what the VLA actually does, so a human-driven chunk would
        # teach it the value of an action the deployed policy cannot produce.
        self.allow_intervention = True
        if view is not None and hasattr(env, "intervention"):
            env.intervention.on_step = self._intervention_step

    def reset(self, critical_phase: bool = False) -> None:
        self.obs = self.env.reset()
        self.ep_steps = 0
        self.ep_return = 0.0
        self.ep_interventions = 0
        self.rl_engaged = not critical_phase
        self.buffer.start_episode()
        self._draw(human=False)

    def _draw(self, *, human: bool, extra_steps: int = 0) -> None:
        if self.view is None:
            return
        self.view.on_step(
            self.obs,
            ep_step=self.ep_steps + extra_steps,
            ep_return=self.ep_return,
            ep_interventions=self.ep_interventions + (1 if human else 0),
            human=human,
        )

    def _intervention_step(self, obs) -> None:
        """Called by the intervention manager after every step it executes."""
        self.obs = obs
        self._live_steps += 1
        self._draw(human=True, extra_steps=self._live_steps)

    def _states_for(self, obs_list: list[dict]) -> list[torch.Tensor]:
        if not obs_list:
            return []
        batch = self.env.obs_to_batch(obs_list, self.device)
        return list(self.controller.compute_x(batch).cpu())

    def _plan(self, batch, use_actor: bool, deterministic: bool):
        return self.controller.plan_chunk(batch, use_actor=use_actor, deterministic=deterministic)

    def run_chunk(self, use_actor: bool, deterministic: bool = False):
        """Plan at the chunk boundary, execute up to C steps, store the record.

        Returns (n_steps, episode_done, episode_success, intervened).
        """
        c = self.controller.chunk_len
        batch = self.env.obs_to_batch([self.obs], self.device)
        # If the operator is already holding the leader, the VLA's action chunk
        # is guaranteed to be discarded; compute only the RL state x and skip
        # the flow-matching sampling the human would otherwise wait through.
        taking_over = self.allow_intervention and self.env.intervention_pending()
        plan = (
            {"x": self.controller.compute_x(batch)}
            if taking_over
            else self._plan(batch, use_actor, deterministic)
        )

        self._live_steps = 0
        intervention = self.env.run_intervention(c) if self.allow_intervention else None

        if intervention is not None:
            actions = intervention.action_chunk.to(plan["x"])
            ref_full = actions[-1:].repeat(2 * c, 1).clone()
            ref_full[:c] = actions
            rewards = intervention.rewards
            step_obs = intervention.obs_list
            n_exec = intervention.n_steps
            terminated, truncated = intervention.done, intervention.truncated
            success = bool(rewards.sum() > 0)
            if step_obs:
                self.obs = step_obs[-1]
            self.ep_steps += n_exec
            self.ep_return += float(rewards.sum())
            self.ep_interventions += 1
        else:
            if taking_over:
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
                self._draw(human=False)
                if terminated:
                    break
                if self.ep_steps >= self.env.max_episode_steps:
                    truncated = True
                    break
                if self.allow_intervention and self.env.intervention_pending():
                    # Hand over mid-chunk. Waiting for the boundary costs the
                    # operator up to C steps (~1 s at 10 Hz) between seeing the
                    # policy go wrong and being able to grab it — long enough to
                    # lose the object. A short chunk is already a representable
                    # record: `done_step` covers n_exec < c.
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
            from_human=intervention is not None,
        )
        self.buffer.add_chunk(rec)
        if truncated and not terminated:
            # A truncated window still bootstraps, so it needs the real state
            # observed after the last executed step.
            x_last = self._states_for([step_obs[-1]]) if step_obs else []
            self.buffer.end_episode(x_last[0] if x_last else None)

        return n_exec, terminated or truncated, success, intervention is not None
