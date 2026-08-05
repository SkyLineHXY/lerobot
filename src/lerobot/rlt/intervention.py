"""Human-in-the-loop primitives for RLT online RL on a real robot.

Two independent channels, matching the paper's system (Sec. V):

* **Sparse outcome labels.** A human supervisor ends each episode with a
  success/failure keypress; success is the only source of reward (r_T = 1).
  The same listener carries the *critical-phase handover* key: episodes start
  under the base VLA and the operator hands control to the RL policy at the
  precise moment that matters, so data collection and credit assignment
  concentrate there.
* **Teleoperated interventions.** The operator may take over mid-episode. The
  taken-over actions replace both the executed action *and* the stored VLA
  reference in the replay buffer, which is what lets the actor's BC term pull
  toward human corrections rather than toward the VLA's failed attempt.

Interventions are decided while the chunk is running and the teleoperator has
to step the arm itself to produce commands, so :meth:`InterventionManager.run_chunk`
executes the whole chunk and returns everything the rollout worker would
otherwise have gathered (the `InterventionResult` pattern from rlt-openpi).
"""
from __future__ import annotations

import logging
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Key bindings mirror the Evo-RLT recording wrapper so operators trained on one
# tool are not retrained for the other.
KEY_SUCCESS = "s"
KEY_FAILURE = "f"
KEY_HANDOVER = "r"
KEY_INTERVENE = " "
KEY_DISCARD = "\x1b[D"  # left arrow
KEY_QUIT = "\x1b"  # bare Esc


@dataclass
class InterventionResult:
    """Outcome of a chunk driven by the human, mirroring the worker's own loop."""

    action_chunk: Tensor  # (C, action_dim), normalized action space
    obs_list: list[dict]  # observation after every executed step
    rewards: Tensor  # (C,)
    n_steps: int  # steps actually executed
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class KeyboardEventListener:
    """Non-blocking single-keypress listener (no Enter needed).

    Falls back to a no-op when stdin is not a TTY (headless / nohup runs), so a
    remote session degrades to "no operator input" instead of crashing.
    """

    def __init__(self) -> None:
        self._old: list | None = None
        self._raw = False
        self._success = False
        self._failure = False
        self._handover = False
        self._discard = False
        self._quit = False
        self._intervene = False

    def start(self) -> None:
        try:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._raw = True
        except (termios.error, OSError, ValueError):
            self._raw = False
            logger.warning(
                "stdin is not a TTY: operator keys (success/failure/handover) are disabled. "
                "Episodes will only end on the step limit."
            )

    def stop(self) -> None:
        if self._raw and self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            self._raw = False
            self._old = None

    def __enter__(self) -> KeyboardEventListener:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _read_keys(self) -> None:
        if not self._raw:
            return
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b" and select.select([sys.stdin], [], [], 0.02)[0]:
                ch += sys.stdin.read(2)  # arrow keys arrive as an escape sequence
            if ch == KEY_SUCCESS:
                self._success = True
            elif ch == KEY_FAILURE:
                self._failure = True
            elif ch == KEY_HANDOVER:
                self._handover = True
            elif ch == KEY_INTERVENE:
                self._intervene = not self._intervene
            elif ch == KEY_DISCARD:
                self._discard = True
            elif ch == KEY_QUIT:
                self._quit = True

    # Latched flags: polled once per control step, consumed by the reader.
    def poll_outcome(self) -> tuple[bool, bool]:
        """(success, failure) since the last call; both are cleared on read."""
        self._read_keys()
        s, f = self._success, self._failure
        self._success = self._failure = False
        return s, f

    def poll_handover(self) -> bool:
        self._read_keys()
        h, self._handover = self._handover, False
        return h

    def poll_discard(self) -> bool:
        self._read_keys()
        d, self._discard = self._discard, False
        return d

    def should_quit(self) -> bool:
        self._read_keys()
        return self._quit

    @property
    def intervening(self) -> bool:
        self._read_keys()
        return self._intervene

    def clear_intervention(self) -> None:
        self._intervene = False

    def reset_episode_flags(self) -> None:
        self._success = self._failure = self._handover = self._discard = False


class InterventionManager:
    """No-op manager: never intervenes. Base class for real teleop devices."""

    def check(self) -> bool:
        return False

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        return None

    def close(self) -> None:
        return None


class PiperLeaderIntervention(InterventionManager):
    """Leader-arm teleoperation for a Piper follower.

    The leader is a second Piper arm held under gravity compensation, so its
    joints map 1:1 onto the follower's 7-dim action and no IK is involved. On
    exit the leader is put back into command mode and driven to the follower's
    current pose, so the next takeover does not begin with a jump.
    """

    def __init__(self, leader, env, keys: KeyboardEventListener):
        self.leader = leader
        self.env = env  # PiperChunkEnv; used for stepping and observations
        self.keys = keys
        self._active = False

    def check(self) -> bool:
        return self.keys.intervening

    def _enter(self) -> None:
        if self._active:
            return
        logger.info("Intervention: leader arm released (gravity compensation).")
        self.leader.set_manual_control(True)
        self._active = True

    def _exit(self) -> None:
        if not self._active:
            return
        # Hand the leader back to command mode and align it with the follower
        # so the operator's next grab starts from the current robot pose.
        self.leader.send_feedback(self.env.raw_joint_action())
        self.leader.set_manual_control(False)
        self._active = False
        logger.info("Intervention: leader arm re-engaged.")

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        if not self.check():
            self._exit()
            return None
        self._enter()

        actions, rewards, obs_list = [], [], []
        done = truncated = False
        for _ in range(chunk_len):
            raw = self.leader.get_action()  # {"joint_N.pos": float, ...}
            norm_action = self.env.raw_action_to_normalized(raw)
            obs, r, done, truncated = self.env.apply_action(norm_action)
            actions.append(norm_action)
            rewards.append(r)
            obs_list.append(obs)
            if done or truncated:
                break
            if not self.check():  # operator let go mid-chunk
                break

        n = len(actions)
        chunk = torch.stack(actions)
        if n < chunk_len:  # hold the last human command for the unused tail
            chunk = torch.cat([chunk, chunk[-1:].expand(chunk_len - n, -1)], dim=0)
        rew = torch.zeros(chunk_len)
        rew[:n] = torch.tensor(rewards, dtype=torch.float32)

        if not self.check():
            self._exit()

        return InterventionResult(
            action_chunk=chunk,
            obs_list=obs_list,
            rewards=rew,
            n_steps=n,
            done=done,
            truncated=truncated,
            info={"intervention": True},
        )

    def close(self) -> None:
        self._exit()


def wait_for_key(keys: KeyboardEventListener, prompt: str, poll_s: float = 0.05) -> bool:
    """Block until the operator confirms (any outcome key) or asks to quit."""
    print(prompt, flush=True)
    while True:
        if keys.should_quit():
            return False
        success, failure = keys.poll_outcome()
        if success or failure:
            return True
        time.sleep(poll_s)
