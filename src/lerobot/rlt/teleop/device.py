"""Teleoperated interventions from a hand-held device (keyboard, gamepad, SpaceMouse).

This is the simulation counterpart of `piper_leader.PiperLeaderIntervention`. A
leader arm maps 1:1 onto the follower's joints; a hand-held device does not, so
the two things this module has to get right are the *action assembly* and the
*takeover gate*.

**Assembly is by name, never by position.** Every LeRobot teleoperator declares
`action_features["names"]` (`delta_x`, `delta_roll`, `gripper`, ...) and the env
declares `action_names` in its own order. Matching them up by name means a
3-DoF keyboard and a 6-DoF SpaceMouse drive the same env through the same code,
with the channels the device lacks left at zero rather than shifted into the
wrong slot.

**Commands may be given in the tool frame.** `teleop.frame="tcp"` rotates the
operator's translation and rotation triples by the current end-effector
orientation before they are executed, so the stick keeps meaning the same thing
relative to the tool no matter how the wrist has turned. The env still receives —
and the buffer still stores — a base-frame action, so the frame is an operator
setting, not a change of action space. See `frames.py`.

**The takeover gate is the trainer's own keyboard listener, not the device's
`get_teleop_events()`.** Routing takeover, success, failure and discard through
`KeyboardEventListener` keeps those operator commands identical in sim and on
hardware, independently of the device-specific motion keys. `use_device_events`
additionally folds the device's own buttons into that same listener — needed for
a gamepad, where both hands are off the keyboard — without ever bypassing it.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.teleoperators.utils import TeleopEvents

from .base import InterventionManager, InterventionResult
from .frames import (
    POSITION_CHANNELS,
    ROTATION_CHANNELS,
    check_frame,
    frame_triples,
    tcp_to_base,
)
from .keys import KeyboardEventListener

logger = logging.getLogger(__name__)

# LeRobot teleoperators encode the gripper as GripperAction: 0 close, 1 stay,
# 2 open. robosuite wants a continuous command where -1 opens and +1 closes.
GRIPPER_CLOSE, GRIPPER_STAY, GRIPPER_OPEN = 0.0, 1.0, 2.0


class DeviceIntervention(InterventionManager):
    """Drive a `ChunkEnv` from any Teleoperator that emits `delta_*` actions."""

    def __init__(
        self,
        teleop,
        env,
        keys: KeyboardEventListener,
        position_scale: float = 1.0,
        rotation_scale: float = 1.0,
        gripper_open_value: float = -1.0,
        gripper_close_value: float = 1.0,
        use_device_events: bool = False,
        frame: str = "base",
    ):
        self.teleop = teleop
        self.env = env
        self.keys = keys
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_open_value = gripper_open_value
        self.gripper_close_value = gripper_close_value
        self.use_device_events = use_device_events and hasattr(teleop, "get_teleop_events")
        if use_device_events and not self.use_device_events:
            logger.warning(
                "%s has no get_teleop_events(); takeover and success/failure stay on the keyboard.",
                type(teleop).__name__,
            )

        self.action_names = list(getattr(env, "action_names", []))
        if not self.action_names:
            raise ValueError(
                f"{type(env).__name__} does not declare `action_names`, so a teleop "
                "device's named channels cannot be mapped onto its action vector."
            )
        device_names = set(teleop.action_features.get("names", {}))
        if not device_names & set(self.action_names):
            raise ValueError(
                f"Teleop device {type(teleop).__name__} emits {sorted(device_names)}, "
                f"none of which the env accepts ({self.action_names})."
            )
        self._unmapped = sorted(device_names - set(self.action_names))
        if self._unmapped:
            logger.warning(
                "Teleop channels %s have no matching env action; they are ignored.",
                self._unmapped,
            )
        self.frame = check_frame(frame, env)
        self._frame_triples = frame_triples(self.action_names) if self.frame == "tcp" else []
        self._gripper_idx = (
            self.action_names.index("gripper") if "gripper" in self.action_names else None
        )
        # Held rather than re-read every step: robosuite's gripper command is a
        # continuous target, so "stay" has to repeat the last command. Sending 0
        # would let a closed gripper drift open mid-grasp.
        self._gripper = gripper_open_value

    def check(self) -> bool:
        return self._poll_device_events() or self.keys.intervening

    def _poll_device_events(self) -> bool:
        """Fold the device's own buttons into the operator key state.

        Returns whether the device is asking for a takeover. Success/failure are
        pushed into the keyboard listener instead of returned, because the env
        reads them from there on every step.
        """
        if not self.use_device_events:
            return False
        events = self.teleop.get_teleop_events()
        if events.get(TeleopEvents.SUCCESS):
            self.keys.latch_outcome(success=True)
        elif events.get(TeleopEvents.TERMINATE_EPISODE):
            self.keys.latch_outcome(failure=True)
        return bool(events.get(TeleopEvents.IS_INTERVENTION))

    def _to_env_action(self, raw: dict[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.action_names), dtype=np.float32)
        for i, name in enumerate(self.action_names):
            if name == "gripper":
                vec[i] = self._gripper_value(raw.get("gripper", GRIPPER_STAY))
            elif name in POSITION_CHANNELS:
                vec[i] = float(raw.get(name, 0.0)) * self.position_scale
            elif name in ROTATION_CHANNELS:
                vec[i] = float(raw.get(name, 0.0)) * self.rotation_scale
            else:
                vec[i] = float(raw.get(name, 0.0))
        if self._frame_triples:
            vec = tcp_to_base(vec, self._frame_triples, self.env.eef_rotation_matrix)
        return vec

    def _gripper_value(self, command: float) -> float:
        if command == GRIPPER_OPEN:
            self._gripper = self.gripper_open_value
        elif command == GRIPPER_CLOSE:
            self._gripper = self.gripper_close_value
        return self._gripper

    def on_reset(self) -> None:
        self._gripper = self.gripper_open_value

    def _sync_gripper_from_env(self) -> None:
        """Adopt the gripper command the env is already holding.

        `_gripper` is a latch, because the device only ever reports open, close
        or stay. Taking over with that latch still on its episode-reset value
        makes the very first human step command "open", which drops whatever the
        policy had just grasped — the operator grabs the controls precisely when
        something is in the gripper, so this fires nearly every time. Envs that
        do not publish their last action (the mock, which has no gripper channel
        at all) keep the latch untouched.
        """
        if self._gripper_idx is None:
            return
        last = getattr(self.env, "last_env_action", None)
        if last is not None:
            self._gripper = float(last[self._gripper_idx])

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        if not self.check():
            return None
        self._sync_gripper_from_env()

        actions, rewards, obs_list = [], [], []
        done = truncated = False
        for _ in range(chunk_len):
            env_action = self._to_env_action(self.teleop.get_action())
            norm_action = self.env.env_action_to_normalized(env_action)
            obs, r, done, truncated = self.env.apply_action(norm_action)
            actions.append(norm_action)
            rewards.append(r)
            obs_list.append(obs)
            self.notify_step(obs)
            if done or truncated:
                break
            if not self.check():  # operator released the takeover toggle
                break

        n = len(actions)
        chunk = torch.stack(actions)
        if n < chunk_len:  # hold the last human command for the unused tail
            chunk = torch.cat([chunk, chunk[-1:].expand(chunk_len - n, -1)], dim=0)
        rew = torch.zeros(chunk_len)
        rew[:n] = torch.tensor(rewards, dtype=torch.float32)

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
        if getattr(self.teleop, "is_connected", False):
            self.teleop.disconnect()


def make_teleop_device(cfg):
    """Build and connect the teleoperator named by a `SimTeleopConfig`."""
    if cfg.device == "keyboard":
        from lerobot.teleoperators.keyboard import (
            KeyboardEndEffectorTeleop,
            KeyboardEndEffectorTeleopConfig,
        )

        teleop = KeyboardEndEffectorTeleop(
            KeyboardEndEffectorTeleopConfig(use_rotation=cfg.use_rotation)
        )
    elif cfg.device == "gamepad":
        from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig

        teleop = GamepadTeleop(
            GamepadTeleopConfig(
                use_rotation=cfg.use_rotation,
                deadzone=cfg.gamepad_deadzone,
                bindings=dict(cfg.gamepad_bindings),
            )
        )
    elif cfg.device == "spacemouse":
        from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig

        teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(use_rotation=cfg.use_rotation))
    else:
        raise ValueError(
            f"unknown teleop device {cfg.device!r}; use keyboard/gamepad/spacemouse"
        )

    teleop.connect()
    if not teleop.is_connected:
        # KeyboardTeleop refuses to start pynput without a DISPLAY, and returns
        # quietly. Degrading to "no human in the loop" here would look like the
        # operator's keys simply not working for the whole run.
        raise RuntimeError(
            f"Teleop device {cfg.device!r} failed to connect. On headless Linux "
            "pynput/pygame need a display — run from a desktop session, or drop "
            "`env.teleop` to train without interventions."
        )
    return teleop


def build_sim_intervention(cfg, env, keys: KeyboardEventListener) -> DeviceIntervention:
    """`SimTeleopConfig` -> a connected `DeviceIntervention` for a sim env."""
    return DeviceIntervention(
        make_teleop_device(cfg),
        env,
        keys,
        position_scale=cfg.position_scale,
        rotation_scale=cfg.rotation_scale,
        use_device_events=cfg.use_device_events,
        frame=cfg.frame,
    )
