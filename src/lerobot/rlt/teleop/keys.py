"""Operator keyboard: raw keypress backends and the latched key state machine.

This is the *sparse outcome label* channel of the paper's system (Sec. V): a
human supervisor ends each episode with a success/failure keypress, success
being the only source of reward (r_T = 1). The same listener carries the
critical-phase handover key, the takeover toggle, and the discard/quit keys.

The teleoperation channel — the device that actually produces actions — lives
in the sibling modules (`base`, `device`, `piper_leader`).
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


class _TermiosBackend:
    """Read raw keypresses from stdin. Only usable when stdin is a real pty.

    Keys reach the process only while its terminal has focus, which is the safe
    behaviour on a robot: nothing typed into another window can command the arm.
    """

    name = "termios"

    # Raw byte sequence -> canonical key name. The four arrows are all decoded
    # even though only `left` is bound by default: a keyboard teleop device
    # steers with the arrows, so the trainer rebinds discard onto `backspace`
    # and the other three still have to arrive under a name it can ignore.
    _SEQUENCES = {
        " ": KEY_INTERVENE,
        "\x1b[A": "up",
        "\x1b[B": "down",
        "\x1b[C": "right",
        "\x1b[D": "left",
        "\x7f": "backspace",
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
    """Backend instances to try in order, for a given backend name.

    ``auto`` prefers termios (needs stdin to be a real terminal; the upside is that
    keys only register while that terminal has focus) and falls back to pynput's
    global X11 hook, which counts keypresses regardless of which window has focus.
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
    """Start the first usable key backend, or return None if none work.

    Shared by :class:`KeyboardEventListener` and the collection script's own state
    machine, whose key bindings are completely different: what is shared is the
    backend preference order and the failure message, not the key table.
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

    def __init__(self, backend: str = "auto", discard_key: str = KEY_DISCARD) -> None:
        if backend not in ("auto", "termios", "pynput", "none"):
            raise ValueError(
                f"unknown keyboard backend {backend!r}; use auto/termios/pynput/none"
            )
        self.backend = backend
        # A keyboard teleop device steers with the arrow keys, which collides
        # with the default `left` = discard. Rebinding is the only way out: both
        # readers see the same global keystream.
        self.discard_key = discard_key
        self._impl = None
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
            elif key == self.discard_key:
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

    def latch_outcome(self, success: bool = False, failure: bool = False) -> None:
        """Raise the outcome flags from something other than the keyboard.

        A gamepad's own buttons are the practical case: with both hands on the
        pad the operator cannot reach `s`/`f`. Forwarding into this listener
        rather than reading the device directly keeps one place where an outcome
        is latched, so the env's `poll_outcome` still sees every source.
        """
        self._success = self._success or success
        self._failure = self._failure or failure

    def reset_episode_flags(self) -> None:
        self._success = self._failure = self._handover = self._discard = False


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
