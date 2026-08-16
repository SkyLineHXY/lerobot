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

**The takeover gate is the trainer's own keyboard listener, not the device's
`get_teleop_events()`.** `KeyboardEndEffectorTeleop.get_action()` clears the
`current_pressed` set that `get_teleop_events()` reads, so whichever of the two
is called second sees nothing — there is no call order that makes both work.
Routing takeover, success, failure and discard through `KeyboardEventListener`
sidesteps that entirely and gives the operator identical keys in sim and on
hardware.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from .base import InterventionManager, InterventionResult
from .keys import KeyboardEventListener

logger = logging.getLogger(__name__)

POSITION_CHANNELS = ("delta_x", "delta_y", "delta_z")
ROTATION_CHANNELS = ("delta_roll", "delta_pitch", "delta_yaw")

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
    ):
        self.teleop = teleop
        self.env = env
        self.keys = keys
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        self.gripper_open_value = gripper_open_value
        self.gripper_close_value = gripper_close_value

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
        # Held rather than re-read every step: robosuite's gripper command is a
        # continuous target, so "stay" has to repeat the last command. Sending 0
        # would let a closed gripper drift open mid-grasp.
        self._gripper = gripper_open_value

    def check(self) -> bool:
        return self.keys.intervening

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
        return vec

    def _gripper_value(self, command: float) -> float:
        if command == GRIPPER_OPEN:
            self._gripper = self.gripper_open_value
        elif command == GRIPPER_CLOSE:
            self._gripper = self.gripper_close_value
        return self._gripper

    def on_reset(self) -> None:
        self._gripper = self.gripper_open_value

    def run_chunk(self, chunk_len: int) -> InterventionResult | None:
        if not self.check():
            return None

        actions, rewards, obs_list = [], [], []
        done = truncated = False
        for _ in range(chunk_len):
            env_action = self._to_env_action(self.teleop.get_action())
            norm_action = self.env.env_action_to_normalized(env_action)
            obs, r, done, truncated = self.env.apply_action(norm_action)
            actions.append(norm_action)
            rewards.append(r)
            obs_list.append(obs)
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

        teleop = GamepadTeleop(GamepadTeleopConfig(use_rotation=cfg.use_rotation))
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
    )
