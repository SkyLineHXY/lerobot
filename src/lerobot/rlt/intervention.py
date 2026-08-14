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
import os
import queue
import select
import sys
import termios
import time
import tty
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Key bindings mirror the Evo-RLT recording wrapper so operators trained on one
# tool are not retrained for the other. These are *canonical* names: every
# backend normalizes to them, so the state machine never sees raw escape codes
# or pynput objects.
KEY_SUCCESS = "s"
KEY_FAILURE = "f"
KEY_HANDOVER = "r"
KEY_INTERVENE = "space"
KEY_DISCARD = "left"
KEY_QUIT = "esc"

# The 6 arm joints (the gripper is excluded from the takeover safety check: it
# is an opening in metres, not an angle, and a gripper mismatch is harmless).
JOINT_KEYS_6 = [f"joint_{i}.pos" for i in range(1, 7)]


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


class _TermiosBackend:
    """Read raw keypresses from stdin. Only usable when stdin is a real pty.

    Keys reach the process only while its terminal has focus, which is the safe
    behaviour on a robot: nothing typed into another window can command the arm.
    """

    name = "termios"

    # Raw byte sequence -> canonical key name.
    _SEQUENCES = {
        " ": KEY_INTERVENE,
        "\x1b[D": KEY_DISCARD,
        "\x1b": KEY_QUIT,
    }

    def __init__(self) -> None:
        self._old: list | None = None
        self._active = False

    def start(self) -> bool:
        try:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError, ValueError, AttributeError):
            return False
        self._active = True
        return True

    def poll(self) -> list[str]:
        if not self._active:
            return []
        keys: list[str] = []
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x1b" and select.select([sys.stdin], [], [], 0.02)[0]:
                ch += sys.stdin.read(2)  # arrow keys arrive as an escape sequence
            keys.append(self._SEQUENCES.get(ch, ch))
        return keys

    def stop(self) -> None:
        if self._active and self._old is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
        self._active = False
        self._old = None


class _PynputBackend:
    """Grab keyboard events at the X11 level, bypassing stdin entirely.

    Migrated from the dual_piper collection node
    (``scripts/data_to_lerobot3_node.py``): an IDE run/debug console, ``nohup``
    and ``roslaunch`` all hand the process a *pipe* rather than a pty, which
    kills the termios backend outright — and with it every operator key, so an
    episode can only end on the step limit and no human can ever intervene.
    pynput keeps working there because it never touches stdin.

    The cost is that capture is **global**: whichever window has focus, `s` /
    `f` / space still count as operator commands. Typing `s` into an editor
    while the rig is running will label the episode a success. That is a real
    hazard on hardware, which is why the termios backend wins whenever stdin is
    a proper terminal.
    """

    name = "pynput"

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._listener = None

    def start(self) -> bool:
        # Same guard the keyboard teleoperator uses: importing pynput without a
        # display raises (or worse, hangs) on headless Linux.
        if "DISPLAY" not in os.environ and "linux" in sys.platform:
            return False
        try:
            from pynput import keyboard
        except Exception as exc:  # ImportError, X11 errors, ...
            logger.debug("pynput unavailable: %s", exc)
            return False

        def on_press(event) -> None:
            # Printable keys carry `.char`; special keys (space/esc/left) carry
            # `.name`, which already matches our canonical names.
            char = getattr(event, "char", None)
            key = char if char is not None else getattr(event, "name", None)
            if key is not None:
                self._queue.put(key)

        try:
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.daemon = True  # never keep the process alive
            self._listener.start()
        except Exception as exc:
            logger.debug("could not start pynput listener: %s", exc)
            self._listener = None
            return False
        return True

    def poll(self) -> list[str]:
        keys: list[str] = []
        while True:
            try:
                keys.append(self._queue.get_nowait())
            except queue.Empty:
                return keys

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.debug("pynput listener stop failed", exc_info=True)
            self._listener = None


def key_backend_candidates(backend: str = "auto") -> list:
    """按 backend 名给出要依次尝试的后端实例。

    ``auto`` 优先 termios（要求 stdin 是真终端，好处是只有该终端有焦点时才收键），
    退到 pynput 的全局 X11 钩子，代价是不管焦点在哪个窗口按键都算数。
    """
    if backend not in ("auto", "termios", "pynput", "none"):
        raise ValueError(f"unknown keyboard backend {backend!r}; use auto/termios/pynput/none")
    if backend == "none":
        return []
    if backend == "termios":
        return [_TermiosBackend()]
    if backend == "pynput":
        return [_PynputBackend()]
    return [_TermiosBackend(), _PynputBackend()]


def start_key_backend(backend: str = "auto"):
    """挑一个可用的按键后端并启动它；都不可用时返回 None。

    两个使用方共用它：本模块的 :class:`KeyboardEventListener`，以及采集脚本里键位
    完全不同的状态机 —— 共用的是"先试哪个后端、失败了怎么提示"，而不是键位表。
    """
    for impl in key_backend_candidates(backend):
        if impl.start():
            if impl.name == "pynput":
                logger.warning(
                    "stdin is not a TTY; falling back to the global pynput hook. Keys are "
                    "captured regardless of which window has focus — typing 's'/'f'/space "
                    "anywhere on this desktop will be read as an operator command."
                )
            return impl

    if backend != "none":
        logger.warning(
            "No usable keyboard backend: operator keys are ALL disabled. stdin is not a "
            "TTY and pynput is unavailable (no DISPLAY?). In PyCharm/VSCode enable the run "
            "configuration's terminal emulation, or run from a real terminal."
        )
    return None


class KeyboardEventListener:
    """Non-blocking single-keypress listener (no Enter needed).

    Two backends, tried in order (``backend="auto"``):

    1. ``termios`` — needs stdin to be a real terminal. Preferred, because keys
       only register while that terminal has focus.
    2. ``pynput`` — global X11 hook, works when stdin is a pipe (IDE console,
       nohup, roslaunch). Chosen only as a fallback: it captures keys no matter
       which window is focused.

    If neither is available the listener degrades to "no operator input" rather
    than crashing, and says so loudly — on a real rig that means the whole
    human-in-the-loop channel is gone.
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend not in ("auto", "termios", "pynput", "none"):
            raise ValueError(
                f"unknown keyboard backend {backend!r}; use auto/termios/pynput/none"
            )
        self.backend = backend
        self._impl = None
        # Keys the six built-in bindings don't claim. `_read_keys` drains the
        # backend in one go, so a caller that polls the backend itself would
        # always find it empty — tools with their own hotkeys must read them
        # from here instead. Bounded so stray typing can't grow without limit.
        self._extra: deque[str] = deque(maxlen=64)
        self._success = False
        self._failure = False
        self._handover = False
        self._discard = False
        self._quit = False
        self._intervene = False

    @property
    def backend_name(self) -> str:
        return self._impl.name if self._impl is not None else "none"

    @property
    def available(self) -> bool:
        """Whether operator keys actually work in this run."""
        return self._impl is not None

    def start(self) -> None:
        self._impl = start_key_backend(self.backend)
        if self._impl is None and self.backend != "none":
            logger.warning(
                "Episodes will only end on the step limit and no human intervention "
                "is possible."
            )

    def stop(self) -> None:
        if self._impl is not None:
            self._impl.stop()
            self._impl = None

    def __enter__(self) -> KeyboardEventListener:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _read_keys(self) -> None:
        if self._impl is None:
            return
        for key in self._impl.poll():
            if key == KEY_SUCCESS:
                self._success = True
            elif key == KEY_FAILURE:
                self._failure = True
            elif key == KEY_HANDOVER:
                self._handover = True
            elif key == KEY_INTERVENE:
                self._intervene = not self._intervene
            elif key == KEY_DISCARD:
                self._discard = True
            elif key == KEY_QUIT:
                self._quit = True
            else:
                self._extra.append(key)

    def poll_extra(self) -> list[str]:
        """Keys not claimed by the built-in bindings; cleared on read.

        For tools that add their own hotkeys on top of the operator keys. They
        must come through here rather than off the backend: `_read_keys` drains
        the backend in a single pass, so a second reader would always lose the
        race and see nothing.
        """
        self._read_keys()
        out = list(self._extra)
        self._extra.clear()
        return out

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

    def set_intervening(self, value: bool) -> None:
        """Seed the takeover toggle (e.g. an `engage_on_start` option).

        Callers must go through the listener rather than driving the robot
        directly: the toggle *is* the state the control loop reads every tick,
        so a takeover arranged behind its back is undone on the very next one.
        """
        self._intervene = bool(value)

    def reset_episode_flags(self) -> None:
        self._success = self._failure = self._handover = self._discard = False


class InterventionManager:
    """No-op manager: never intervenes. Base class for real teleop devices."""

    def check(self) -> bool:
        return False

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        return None

    def on_reset(self) -> None:
        """Called at every episode reset, before the first chunk."""
        return None

    def close(self) -> None:
        return None


class PiperLeaderIntervention(InterventionManager):
    """Leader-arm teleoperation for a Piper follower.

    The leader is a second Piper arm held under gravity compensation, so its
    joints map 1:1 onto the follower's 7-dim action and no IK is involved. On
    exit the leader is put back into command mode and driven to the follower's
    current pose, so the next takeover does not begin with a jump.

    The same alignment runs at every episode reset (:meth:`on_reset`), because
    the leader enters gravity compensation the moment it connects: without it
    the *first* takeover of a run would start from wherever the limp leader
    happens to be resting and drag the follower there.
    """

    def __init__(
        self,
        leader,
        env,
        keys: KeyboardEventListener,
        use_calibrated_offsets: bool = False,
        max_takeover_delta_rad: float = 0.15,
    ):
        self.leader = leader
        self.env = env  # PiperChunkEnv; used for stepping and observations
        self.keys = keys
        # `get_action()` returns offsets from the calibrated neutral pose; the
        # stage-1 dataset actions are absolute joint angles. Default to the
        # absolute reading so human corrections land in the policy's own action
        # space instead of being shifted by the leader's neutral pose.
        self.use_calibrated_offsets = use_calibrated_offsets
        self.max_takeover_delta_rad = max_takeover_delta_rad
        self._active = False

    # ------------------------------------------------------------------ leader
    def _read_leader(self) -> dict[str, float]:
        """One leader sample, already renamed to the follower's joint keys."""
        from .piper_env import leader_action_to_follower

        raw = (
            self.leader.get_action()
            if self.use_calibrated_offsets
            else self.leader.get_raw_action()
        )
        return leader_action_to_follower(raw)

    def _takeover_delta(self) -> float:
        """Largest |leader - follower| over the 6 arm joints, in radians."""
        leader = self._read_leader()
        follower = self.env.raw_joint_action()
        return max(abs(leader[k] - follower[k]) for k in JOINT_KEYS_6)

    def check(self) -> bool:
        return self.keys.intervening

    def align(self) -> None:
        """Drive the leader to the follower's pose and hold it in command mode."""
        from .piper_env import follower_action_to_leader

        self.leader.send_feedback(follower_action_to_leader(self.env.raw_joint_action()))
        self.leader.set_manual_control(False)
        self._active = False

    def on_reset(self) -> None:
        self.align()

    def _enter(self) -> bool:
        """Release the leader for the operator. False if it is unsafe to do so."""
        if self._active:
            return True
        # The leader has been holding the follower's pose since the last exit
        # (or since `on_reset`), so a large gap means the operator has already
        # dragged it somewhere else — releasing now would make the follower
        # chase that pose. Refuse and make them re-press after re-aligning.
        delta = self._takeover_delta()
        if delta > self.max_takeover_delta_rad > 0:
            logger.warning(
                "Intervention refused: leader is %.3f rad away from the follower "
                "(limit %.3f). Re-aligning; press space again once the leader is "
                "back on the robot's pose.",
                delta,
                self.max_takeover_delta_rad,
            )
            self.keys.clear_intervention()
            self.align()
            return False
        logger.info("Intervention: leader arm released (gravity compensation).")
        self.leader.set_manual_control(True)
        self._active = True
        return True

    def _exit(self) -> None:
        if not self._active:
            return
        # Hand the leader back to command mode and align it with the follower
        # so the operator's next grab starts from the current robot pose.
        self.align()
        logger.info("Intervention: leader arm re-engaged.")

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        if not self.check():
            self._exit()
            return None
        if not self._enter():
            return None

        actions, rewards, obs_list = [], [], []
        done = truncated = False
        for _ in range(chunk_len):
            raw = self._read_leader()  # {"joint_N.pos": float, ...}
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
        try:
            self._exit()
        finally:
            # The leader owns a CAN handle and a 200 Hz gravity-compensation
            # thread; leaving it connected keeps the arm enabled after the run.
            if getattr(self.leader, "is_connected", False):
                self.leader.disconnect()


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
