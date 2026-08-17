#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from ..utils import TeleopEvents


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def get_rotation_deltas(self):
        """Get the current wrist rotation deltas (roll, pitch, yaw) in radians."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


STICK_CONTROLS = ("leftx", "lefty", "rightx", "righty")
TRIGGER_CONTROLS = ("lefttrigger", "righttrigger")
BUTTON_CONTROLS = (
    "a",
    "b",
    "x",
    "y",
    "leftshoulder",
    "rightshoulder",
    "leftstick",
    "rightstick",
    "back",
    "start",
    "guide",
    "dpup",
    "dpdown",
    "dpleft",
    "dpright",
)
CONTROL_NAMES = frozenset(STICK_CONTROLS + TRIGGER_CONTROLS + BUTTON_CONTROLS)


def parse_binding(expression: str) -> tuple[str | None, str | None]:
    """`"righttrigger-lefttrigger"` -> `("righttrigger", "lefttrigger")`.

    A leading "-" leaves the positive side empty, so `"-lefty"` negates the axis;
    an empty expression disables the channel.
    """
    parts = [part.strip() for part in expression.split("-")]
    if len(parts) > 2:
        raise ValueError(f"binding {expression!r} has more than one '-'; write it as 'positive-negative'")
    positive = parts[0]
    negative = parts[1] if len(parts) == 2 else ""
    for name in (positive, negative):
        if name and name not in CONTROL_NAMES:
            raise ValueError(f"unknown gamepad control {name!r}; pick from {sorted(CONTROL_NAMES)}")
    return positive or None, negative or None


class _ControllerPad:
    """SDL's game-controller layer, which normalises every pad to one layout.

    Raw joystick numbering is per-device and unusable as a default: the pad this
    was developed against reports `lefttrigger` as axis 5 and `righttrigger` as
    axis 4, and its face buttons as 0/1/3/4. Worse, an axis the pad does not
    have reads a constant -1.0 rather than 0.0, so a wrong index does not go
    quiet — it looks like a stick held hard over, which is how the old default
    `rotation_axes=(2, 4, 5)` produced a wrist that span continuously.
    """

    def __init__(self, index: int):
        import pygame
        from pygame._sdl2 import controller

        self._pygame = pygame
        self._controller = controller.Controller(index)
        self.name = self._controller.name
        self._axes = {
            "leftx": pygame.CONTROLLER_AXIS_LEFTX,
            "lefty": pygame.CONTROLLER_AXIS_LEFTY,
            "rightx": pygame.CONTROLLER_AXIS_RIGHTX,
            "righty": pygame.CONTROLLER_AXIS_RIGHTY,
            "lefttrigger": pygame.CONTROLLER_AXIS_TRIGGERLEFT,
            "righttrigger": pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
        }
        self._buttons = {
            "a": pygame.CONTROLLER_BUTTON_A,
            "b": pygame.CONTROLLER_BUTTON_B,
            "x": pygame.CONTROLLER_BUTTON_X,
            "y": pygame.CONTROLLER_BUTTON_Y,
            "leftshoulder": pygame.CONTROLLER_BUTTON_LEFTSHOULDER,
            "rightshoulder": pygame.CONTROLLER_BUTTON_RIGHTSHOULDER,
            "leftstick": pygame.CONTROLLER_BUTTON_LEFTSTICK,
            "rightstick": pygame.CONTROLLER_BUTTON_RIGHTSTICK,
            "back": pygame.CONTROLLER_BUTTON_BACK,
            "start": pygame.CONTROLLER_BUTTON_START,
            "guide": pygame.CONTROLLER_BUTTON_GUIDE,
            "dpup": pygame.CONTROLLER_BUTTON_DPAD_UP,
            "dpdown": pygame.CONTROLLER_BUTTON_DPAD_DOWN,
            "dpleft": pygame.CONTROLLER_BUTTON_DPAD_LEFT,
            "dpright": pygame.CONTROLLER_BUTTON_DPAD_RIGHT,
        }

    def pump(self) -> bool:
        try:
            self._pygame.event.pump()
        except self._pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return False
        return True

    def read(self, control: str) -> float:
        if control in self._axes:
            return max(-1.0, min(1.0, self._controller.get_axis(self._axes[control]) / 32767.0))
        return float(self._controller.get_button(self._buttons[control]))

    def close(self) -> None:
        self._controller.quit()


class _JoystickPad:
    """Raw-index fallback for pads SDL has no game-controller mapping for."""

    # The layout most generic dual-stick pads expose. It is a guess by
    # construction, which is why _ControllerPad is always tried first.
    AXES = {"leftx": 0, "lefty": 1, "rightx": 2, "righty": 3}
    TRIGGERS = {"lefttrigger": 4, "righttrigger": 5}
    BUTTONS = {
        "a": 0,
        "b": 1,
        "x": 2,
        "y": 3,
        "leftshoulder": 4,
        "rightshoulder": 5,
        "back": 6,
        "start": 7,
        "leftstick": 8,
        "rightstick": 9,
        "guide": 10,
    }
    HAT = {"dpup": (1, 1), "dpdown": (1, -1), "dpleft": (0, -1), "dpright": (0, 1)}

    def __init__(self, index: int):
        import pygame

        self._pygame = pygame
        self._joystick = pygame.joystick.Joystick(index)
        self._joystick.init()
        self.name = self._joystick.get_name()

    def pump(self) -> bool:
        try:
            self._pygame.event.pump()
        except self._pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return False
        return True

    def read(self, control: str) -> float:
        if control in self.AXES:
            return self._axis(self.AXES[control])
        if control in self.TRIGGERS:
            # Unipolar triggers rest at -1.0 on the raw interface.
            return (self._axis(self.TRIGGERS[control], resting=-1.0) + 1.0) / 2.0
        if control in self.HAT:
            if self._joystick.get_numhats() == 0:
                return 0.0
            component, sign = self.HAT[control]
            return float(self._joystick.get_hat(0)[component] == sign)
        index = self.BUTTONS[control]
        if index >= self._joystick.get_numbuttons():
            return 0.0
        return float(self._joystick.get_button(index))

    def _axis(self, index: int, resting: float = 0.0) -> float:
        if index >= self._joystick.get_numaxes():
            return resting
        return self._joystick.get_axis(index)

    def close(self) -> None:
        self._joystick.quit()


class GamepadController(InputController):
    """Generate motion deltas from gamepad input.

    Channels are bound to *named* controls (`"righttrigger-lefttrigger"`) rather
    than raw axis indices, so one binding table works across pads. See
    `configuration_gamepad.DEFAULT_GAMEPAD_BINDINGS`.
    """

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
        bindings=None,
    ):
        from .configuration_gamepad import DEFAULT_GAMEPAD_BINDINGS

        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.bindings = {**DEFAULT_GAMEPAD_BINDINGS, **(bindings or {})}
        unknown = sorted(set(self.bindings) - set(DEFAULT_GAMEPAD_BINDINGS))
        if unknown:
            raise ValueError(f"unknown gamepad binding channels {unknown}")
        self._terms = {channel: parse_binding(expr) for channel, expr in self.bindings.items()}
        self.pad = None
        self._pressed: dict[str, bool] = {}

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame
        from pygame._sdl2 import controller

        pygame.init()
        pygame.joystick.init()
        controller.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        if controller.is_controller(0):
            self.pad = _ControllerPad(0)
        else:
            self.pad = _JoystickPad(0)
            logging.warning(
                "SDL has no game-controller mapping for %r, so the raw axis/button layout is a "
                "guess. Verify it with `lerobot-find-gamepad`, and if it is wrong export a mapping "
                "string in SDL_GAMECONTROLLERCONFIG.",
                self.pad.name,
            )
        logging.info("Initialized gamepad: %s", self.pad.name)
        print(f"Gamepad: {self.pad.name}")
        for channel, expression in self.bindings.items():
            print(f"  {channel:<14} {expression or '(disabled)'}")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if self.pad is not None:
            self.pad.close()
            self.pad = None
        if pygame.joystick.get_init():
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        if self.pad is None or not self.pad.pump():
            return

        for channel, status in (
            ("success", TeleopEvents.SUCCESS),
            ("failure", TeleopEvents.FAILURE),
            ("rerecord", TeleopEvents.RERECORD_EPISODE),
        ):
            if self._pressed_edge(channel):
                self.episode_end_status = status

        self.close_gripper_command = self.channel("gripper_close") > 0.5
        self.open_gripper_command = self.channel("gripper_open") > 0.5
        self.intervention_flag = self.channel("intervention") > 0.5

    def channel(self, name: str) -> float:
        """Current value of one bound channel, in [-1, 1]."""
        if self.pad is None:
            return 0.0
        positive, negative = self._terms[name]
        return self._read(positive) - self._read(negative)

    def _read(self, control: str | None) -> float:
        if control is None:
            return 0.0
        value = self.pad.read(control)
        # Buttons and triggers rest at exactly 0, so one deadzone covers both
        # them and the sticks, whose centre drifts by a percent or two.
        return 0.0 if abs(value) < self.deadzone else value

    def _pressed_edge(self, channel: str) -> bool:
        """True only on the frame a channel goes from released to pressed."""
        down = self.channel(channel) > 0.5
        was_down = self._pressed.get(channel, False)
        self._pressed[channel] = down
        return down and not was_down

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        return (
            self.channel("delta_x") * self.x_step_size,
            self.channel("delta_y") * self.y_step_size,
            self.channel("delta_z") * self.z_step_size,
        )

    def get_rotation_deltas(self):
        return (
            self.channel("delta_roll"),
            self.channel("delta_pitch"),
            self.channel("delta_yaw"),
        )


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: Move in X-Y plane")
            logging.info("  Right analog stick: Move in Z axis (vertical)")
            logging.info("  Button 1/B/Circle: Exit")
            logging.info("  Button 2/A/Cross: End episode with SUCCESS")
            logging.info("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_y = (data[1] - 128) / 128.0
                self.left_x = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif buttons & 1 << 5:
                    self.episode_end_status = TeleopEvents.FAILURE
                elif buttons & 1 << 4:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                else:
                    self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_x * self.x_step_size  # Forward/backward
        delta_y = -self.left_y * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z
