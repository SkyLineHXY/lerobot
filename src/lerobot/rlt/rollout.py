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
