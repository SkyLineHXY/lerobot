"""The rollout half of stage 2: execute one chunk, assemble one `ChunkRecord`.

Runs on the main thread — and whenever the operator has taken over, the main
thread *is* the human's control loop. So the expensive part of a chunk boundary
(a VLM prefix forward plus a full flow-matching sample, twice: once for the
plan, once for the subsample anchors) is pushed onto a background assembler
thread. Only the plan whose actions the policy is about to execute has to be
computed synchronously; everything else the buffer needs is derived from
observations that are already in hand, so it can be finished later without
changing a single stored value.

That split is what keeps the arm from freezing at every chunk boundary of a
takeover. The assembler still shares a CUDA context with the learner thread, so
it is throttled rather than free — but it is no longer in front of the human.
"""
from __future__ import annotations

import functools
import queue
import threading
import time
from collections import deque

import torch

from lerobot.policies.rlt import RLTController, TransitionSource

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
        truncation_is_failure: bool = False,
        async_records: bool = True,
    ):
        self.env = env
        self.controller = controller
        self.buffer = buffer
        self.stride = stride
        # In a simulator with a completion checker, running out the clock is a
        # failure whose return really is 0, so the Monte Carlo target is valid.
        self.truncation_is_failure = truncation_is_failure
        self.keys = keys
        self.view = view
        self.device = next(controller.rl_token.parameters()).device
        self.obs = None
        self.ep_steps = 0
        self.ep_return = 0.0
        self.ep_interventions = 0
        self.ep_intervention_steps = 0
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

        # Wall-clock gap between the last executed step of one chunk and the
        # first of the next. This is the number the operator feels, and the only
        # honest way to tell whether deferring the VLA work actually helped.
        self.boundary_gaps: deque[float] = deque(maxlen=256)
        self._last_exec_end = time.perf_counter()

        # Bounded so a GPU that falls behind stalls the rollout instead of
        # growing without limit; the episode trace it feeds is per-episode
        # anyway, so a deep queue would buy nothing.
        # Guards the stage-1 preprocessor, which the assembler and the control
        # thread both enter. See `_obs_to_batch`.
        self._preprocess_lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self._job_error: BaseException | None = None
        self._assembler: threading.Thread | None = None
        if async_records:
            self._assembler = threading.Thread(
                target=self._assemble_loop, name="rlt-record-assembler", daemon=True
            )
            self._assembler.start()

        # Wired unconditionally, not only when a view exists: the hook is also
        # how `self.obs` and the boundary-gap clock follow the human's steps.
        if hasattr(env, "intervention"):
            env.intervention.on_step = self._intervention_step

    # ------------------------------------------------------- record assembly
    def _assemble_loop(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                job()
            except BaseException as exc:  # re-raised on the control thread
                self._job_error = exc
            finally:
                self._jobs.task_done()

    def _raise_deferred(self) -> None:
        exc, self._job_error = self._job_error, None
        if exc is not None:
            raise exc

    def _submit(self, job) -> None:
        self._raise_deferred()
        if self._assembler is None:
            job()
            return
        self._jobs.put(job)

    def drain(self) -> None:
        """Block until every submitted record has reached the buffer.

        Mandatory before anything that touches the buffer's episode trace from
        the control thread — `start_episode`, `discard_episode`, `save` — since
        an in-flight chunk would otherwise land in the wrong episode.
        """
        if self._assembler is not None:
            self._jobs.join()
        self._raise_deferred()

    def stop(self) -> None:
        if self._assembler is None:
            return
        self._jobs.put(None)
        self._assembler.join(timeout=5.0)
        self._assembler = None

    # -------------------------------------------------------------- rollout
    def reset(self, critical_phase: bool = False, warmup: bool = False) -> None:
        self.drain()
        self.obs = self.env.reset()
        self.ep_steps = 0
        self.ep_return = 0.0
        self.ep_interventions = 0
        self.ep_intervention_steps = 0
        self.rl_engaged = not critical_phase
        self.buffer.start_episode(warmup=warmup)
        self._last_exec_end = time.perf_counter()
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
        self._last_exec_end = time.perf_counter()
        self._draw(human=True, extra_steps=self._live_steps)

    def _states_and_refs_for(self, obs_list: list[dict]):
        """RL state *and* a freshly sampled VLA reference for each observation.

        The reference has to be sampled here rather than sliced out of the chunk
        boundary's plan: at an anchor k steps into the chunk, that plan's row k
        is an open-loop continuation, while the actions stored alongside it are
        what the VLA re-planned after actually moving k steps. Measured on the
        2026-08-20 LIBERO buffer the two sit 0.41 apart per element against a
        reference std of 0.9, and the re-plan is systematically *smaller*
        (0.39-0.92x per dimension) — so slicing taught the actor to shrink the
        VLA's actions rather than to improve them.
        """
        if not obs_list:
            return [], []
        batch = self._obs_to_batch(obs_list)
        x, ref_full = self.controller.compute_x(batch, with_ref=True)
        return list(x.cpu()), list(ref_full.cpu())

    def _obs_to_batch(self, obs_list: list[dict]):
        """The one place the stage-1 preprocessor is entered, under a lock.

        `NormalizerProcessorStep` is not thread-safe: it keeps its statistics on
        whichever device it last saw and migrates them lazily
        (`_apply_transform` -> `self.to(...)`, which *rebinds* `_tensor_stats`).
        Two threads in that check-then-rebind race, and one of them ends up
        subtracting a CPU mean from a CUDA tensor. Nothing else in this class
        may call `env.obs_to_batch` directly.

        The control thread only lands here on the policy path — a takeover
        defers its plan — so serialising costs the operator nothing.
        """
        with self._preprocess_lock:
            return self.env.obs_to_batch(obs_list, self.device)

    def _plan(self, batch, use_actor: bool, deterministic: bool):
        return self.controller.plan_chunk(batch, use_actor=use_actor, deterministic=deterministic)

    def _plan_boundary(self, obs, use_actor: bool, deterministic: bool):
        return self._plan(self._obs_to_batch([obs]), use_actor, deterministic)

    def _store_chunk(
        self,
        *,
        base: int,
        boundary_obs: dict,
        boundary_state,
        step_obs: list[dict],
        actions,
        rewards,
        n_exec: int,
        source: int,
        terminated: bool,
        truncated: bool,
        success: bool,
    ) -> None:
        """Finish one chunk's transitions and hand them to the buffer.

        Runs on the assembler thread. Everything it reads was captured before
        the chunk executed, so deferring it cannot change what gets stored —
        which holds only because an env hands out a fresh observation per step.
        Verified for the robosuite envs (a step returns new arrays and leaves
        earlier ones alone) and true by construction for `PiperChunkEnv`. An env
        that reuses one observation buffer would silently anchor these
        transitions on frames from later in the chunk.
        """
        interior = [
            i - base
            for i in range(base + 1, base + n_exec)
            if i % self.stride == 0 and i - base - 1 < len(step_obs)
        ]
        boundary_anchor = base % self.stride == 0
        want_last = (terminated or truncated) and bool(step_obs)

        # One batched forward covers everything this chunk still owes the
        # buffer. Splitting it would cost a separate VLM prefix forward and a
        # separate flow-matching sample per group.
        obs_batch: list[dict] = []
        if boundary_anchor and boundary_state is None:
            obs_batch.append(boundary_obs)
        obs_batch += [step_obs[o - 1] for o in interior]
        if want_last:
            obs_batch.append(step_obs[-1])
        xs_all, refs_all = self._states_and_refs_for(obs_batch)

        cursor = 0
        if boundary_anchor and boundary_state is None:
            boundary_state = (xs_all[0], refs_all[0])
            cursor = 1

        xs = xs_all[cursor : cursor + len(interior)]
        refs = refs_all[cursor : cursor + len(interior)]
        offsets = list(interior)
        # Interior anchors get a fresh, deploy-consistent reference, but their
        # executed action window can straddle two planning decisions. They are
        # still valid for reference BC and the off-policy critic; `aligned`
        # only prevents optional behaviour imitation from treating them as one
        # policy sample.
        aligned = [False] * len(interior)
        if boundary_anchor:
            offsets = [0, *offsets]
            xs = [boundary_state[0], *xs]
            refs = [boundary_state[1], *refs]
            aligned = [True, *aligned]

        self.buffer.add_chunk(
            ChunkRecord(
                xs=torch.stack(xs) if xs else torch.zeros(0),
                x_offsets=torch.tensor(offsets, dtype=torch.long),
                refs=torch.stack(refs) if refs else torch.zeros(0),
                aligned=torch.tensor(aligned, dtype=torch.bool),
                actions=actions,
                rewards=rewards,
                source=source,
                done=terminated,
            )
        )

        if terminated or truncated:
            self.buffer.end_episode(
                xs_all[-1] if want_last else None,
                refs_all[-1] if want_last else None,
                success=success,
                truncation_is_failure=self.truncation_is_failure,
            )

    def run_chunk(self, use_actor: bool, deterministic: bool = False):
        """Plan at the chunk boundary, execute up to C steps, store the record.

        Returns (n_steps, episode_done, episode_success, intervened).
        """
        c = self.controller.chunk_len
        base = self.ep_steps
        boundary_obs = self.obs
        self._raise_deferred()

        # A chunk the operator is already driving still has to store the VLA
        # reference for this boundary, but nothing in the control loop waits
        # for it — the actions it would produce are discarded. Sampling it here
        # would hold the arm still for a VLM forward plus the whole
        # flow-matching loop while the human's hand keeps moving, which is the
        # stutter they feel at every boundary of a takeover. The assembler
        # samples it from this same observation instead.
        plan = None
        if not (self.allow_intervention and self.env.intervention_pending()):
            plan = self._plan_boundary(boundary_obs, use_actor, deterministic)

        self._live_steps = 0
        self.boundary_gaps.append(time.perf_counter() - self._last_exec_end)
        intervention = self.env.run_intervention(c) if self.allow_intervention else None

        if intervention is None and plan is None:
            # The takeover was refused — on hardware, because the leader had
            # been dragged away from the follower — so the policy drives this
            # chunk after all and does need a plan.
            plan = self._plan_boundary(boundary_obs, use_actor, deterministic)

        if intervention is not None:
            actions = intervention.action_chunk
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
            self.ep_intervention_steps += n_exec
            source = int(TransitionSource.HUMAN)
        else:
            actions = plan["action_chunk"][0]
            rewards = torch.zeros(c)
            step_obs = []
            terminated = truncated = False
            n_exec = 0
            for j in range(c):
                self.obs, r, terminated = self.env.step(actions[j])
                self._last_exec_end = time.perf_counter()
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
                    break
            success = bool(rewards.sum() > 0)
            source = int(TransitionSource.RL if use_actor else TransitionSource.BASE)

        # Only materialised for a boundary that is actually an anchor; anywhere
        # else this would be a GPU sync whose result nothing reads.
        boundary_state = None
        if plan is not None and base % self.stride == 0:
            boundary_state = (plan["x"][0].detach().cpu(), plan["ref_full"][0].detach().cpu())

        self._submit(
            functools.partial(
                self._store_chunk,
                base=base,
                boundary_obs=boundary_obs,
                boundary_state=boundary_state,
                step_obs=step_obs,
                actions=actions[:n_exec].detach().cpu().float(),
                rewards=rewards[:n_exec].detach().cpu().float(),
                n_exec=n_exec,
                source=source,
                terminated=terminated,
                truncated=truncated,
                success=success,
            )
        )

        return n_exec, terminated or truncated, success, intervention is not None
