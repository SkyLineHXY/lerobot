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

import struct
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.gen_gripper import UmiGripper, UmiGripperConfig
from lerobot.robots.gen_gripper.config_umi_gripper import SIDE_DEFAULTS
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


# ----------------------------------------------------------------------
# Hardware mocks (DataBus / CameraCapture / tactile processing)
# ----------------------------------------------------------------------


def _make_databus_mock() -> MagicMock:
    """A DataBus mock that captures the registered encoder/tactile callbacks
    so tests can drive them synchronously."""
    bus = MagicMock(name="DataBusMock")

    def _init(*_args, **kwargs):
        # The real DataBus stores these and invokes them from background threads;
        # we just keep references so tests can fire them directly.
        bus.encoder_callback = kwargs.get("encoder_callback")
        bus.tactile_callback = kwargs.get("tactile_callback")
        return bus

    bus.side_effect = _init
    return bus


def _make_camera_mock(num_cameras: int = 0) -> MagicMock:
    """A CameraCapture mock with the attributes touched by UmiGripper."""
    cam = MagicMock(name="CameraCaptureMock")

    def _init(*_args, **_kwargs):
        cam.running = True
        cam.cameras = [
            {"id": i, "cap": MagicMock(), "frame_count": 0} for i in range(num_cameras)
        ]
        # Default: cap.read() returns no frame so the capture loop is a no-op
        for entry in cam.cameras:
            entry["cap"].read.return_value = (False, None)
        return cam

    cam.side_effect = _init
    return cam


@pytest.fixture
def patches():
    """Patch SDK entry points used by UmiGripper.connect()."""
    databus_mock = _make_databus_mock()
    camera_mock = _make_camera_mock(num_cameras=0)

    with (
        patch("lerobot.robots.gen_gripper.umi_gripper.DataBus", databus_mock),
        patch("lerobot.robots.gen_gripper.umi_gripper.CameraCapture", camera_mock),
        patch("lerobot.robots.gen_gripper.umi_gripper.time.sleep", lambda *_a, **_k: None),
    ):
        yield {"DataBus": databus_mock, "CameraCapture": camera_mock}


@pytest.fixture
def gripper(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0, enable_tactile=False)
    robot = UmiGripper(cfg)
    yield robot
    if robot.is_connected:
        robot.disconnect()


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------


def test_config_defaults_left():
    cfg = UmiGripperConfig(side="left")
    assert cfg.serial_port == SIDE_DEFAULTS["left"]["serial_port"]
    assert cfg.video_devices == SIDE_DEFAULTS["left"]["video_devices"]


def test_config_defaults_right():
    cfg = UmiGripperConfig(side="right")
    assert cfg.serial_port == SIDE_DEFAULTS["right"]["serial_port"]
    assert cfg.video_devices == SIDE_DEFAULTS["right"]["video_devices"]


def test_config_invalid_side():
    with pytest.raises(ValueError, match="side must be"):
        UmiGripperConfig(side="middle")


def test_config_user_overrides_kept():
    cfg = UmiGripperConfig(side="left", serial_port="/dev/ttyFAKE", video_devices=["/dev/v0"])
    assert cfg.serial_port == "/dev/ttyFAKE"
    assert cfg.video_devices == ["/dev/v0"]


# ----------------------------------------------------------------------
# Feature descriptors
# ----------------------------------------------------------------------


def test_action_features():
    robot = UmiGripper(UmiGripperConfig(side="left"))
    assert robot.action_features == {"gripper.pos": float}


def test_observation_features_without_tactile():
    cfg = UmiGripperConfig(side="left", camera_count=2, camera_width=320, camera_height=240)
    robot = UmiGripper(cfg)
    feats = robot.observation_features
    assert feats["gripper.pos"] is float
    assert feats["cam_0"] == (240, 320, 3)
    assert feats["cam_1"] == (240, 320, 3)
    assert "tactile_left" not in feats


def test_observation_features_with_tactile():
    cfg = UmiGripperConfig(side="left", camera_count=1, enable_tactile=True)
    robot = UmiGripper(cfg)
    feats = robot.observation_features
    assert feats["tactile_left"] == (500,)
    assert feats["tactile_right"] == (500,)


# ----------------------------------------------------------------------
# Connect / disconnect lifecycle
# ----------------------------------------------------------------------


def test_is_calibrated(gripper):
    assert gripper.is_calibrated is True


def test_connect_disconnect(gripper):
    assert not gripper.is_connected

    gripper.connect()
    assert gripper.is_connected

    gripper.disconnect()
    assert not gripper.is_connected


def test_double_connect_raises(gripper):
    gripper.connect()
    with pytest.raises(DeviceAlreadyConnectedError):
        gripper.connect()


def test_disconnect_when_not_connected_raises(gripper):
    with pytest.raises(DeviceNotConnectedError):
        gripper.disconnect()


def test_camera_init_failure_propagates(patches):
    patches["CameraCapture"].side_effect = RuntimeError("no /dev/video0")
    robot = UmiGripper(UmiGripperConfig(side="left", camera_count=0))
    with pytest.raises(ConnectionError, match="Camera init failed"):
        robot.connect()
    assert not robot.is_connected


def test_serial_init_failure_releases_camera(patches):
    patches["DataBus"].side_effect = RuntimeError("serial busy")
    robot = UmiGripper(UmiGripperConfig(side="left", camera_count=0))

    with pytest.raises(ConnectionError, match="Serial init failed"):
        robot.connect()

    # Camera should have been stopped during cleanup. Our CameraCapture mock
    # returns itself as the instance (side_effect returns the mock object), so
    # the .stop() call is recorded on the patch object directly.
    patches["CameraCapture"].stop.assert_called()
    assert robot._camera is None
    assert not robot.is_connected


# ----------------------------------------------------------------------
# Observation / action error paths
# ----------------------------------------------------------------------


def test_get_observation_not_connected_raises(gripper):
    with pytest.raises(DeviceNotConnectedError):
        gripper.get_observation()


def test_send_action_not_connected_raises(gripper):
    with pytest.raises(DeviceNotConnectedError):
        gripper.send_action({"gripper.pos": 0.05})


# ----------------------------------------------------------------------
# Observation content
# ----------------------------------------------------------------------


def test_observation_returns_zero_frames_before_callbacks(patches):
    cfg = UmiGripperConfig(side="left", camera_count=2, camera_width=320, camera_height=240)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        obs = robot.get_observation()
        assert obs["gripper.pos"] == 0.0
        for i in range(2):
            assert obs[f"cam_{i}"].shape == (240, 320, 3)
            assert obs[f"cam_{i}"].dtype == np.uint8
            assert not obs[f"cam_{i}"].any()  # all-zero placeholder
    finally:
        robot.disconnect()


def test_observation_uses_latest_frame_callback(patches):
    cfg = UmiGripperConfig(side="left", camera_count=1, camera_width=320, camera_height=240)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        fake_frame = np.full((240, 320, 3), 7, dtype=np.uint8)
        robot._frame_callback(0, fake_frame, time.time_ns())

        obs = robot.get_observation()
        np.testing.assert_array_equal(obs["cam_0"], fake_frame)
        # get_observation should return a copy, not the same buffer
        assert obs["cam_0"] is not fake_frame
    finally:
        robot.disconnect()


def test_encoder_callback_updates_observation(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        # Encoder callback receives 4 bytes big-endian float
        robot._encoder_callback(struct.pack(">f", 0.042))
        obs = robot.get_observation()
        assert obs["gripper.pos"] == pytest.approx(0.042, rel=1e-5)
    finally:
        robot.disconnect()


def test_tactile_observation(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0, enable_tactile=True)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        # Before any callback fires, observation should still be the zero placeholder
        obs = robot.get_observation()
        assert obs["tactile_left"].shape == (500,)
        assert obs["tactile_right"].shape == (500,)
        assert obs["tactile_left"].dtype == np.int16
        assert not obs["tactile_left"].any()

        # Inject tactile data manually (bypass real conversion function)
        left = list(range(500))
        right = list(range(500, 1000))
        with patch(
            "lerobot.robots.gen_gripper.umi_gripper.convert_tactile_448_to_1000",
            return_value=(left, right),
        ):
            robot._tactile_callback(b"\x00" * 448)

        obs = robot.get_observation()
        np.testing.assert_array_equal(obs["tactile_left"], np.array(left, dtype=np.int16))
        np.testing.assert_array_equal(obs["tactile_right"], np.array(right, dtype=np.int16))
    finally:
        robot.disconnect()


def test_encoder_callback_swallows_errors(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0)
    robot = UmiGripper(cfg)
    # Bad payload (not 4 bytes) — must not raise to avoid killing the SDK thread
    robot._encoder_callback(b"\x00")
    # Latest encoder unchanged (still 0.0 from __init__)
    assert robot._latest_encoder == 0.0


# ----------------------------------------------------------------------
# Action handling
# ----------------------------------------------------------------------


def test_send_action_forwards_to_databus(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        result = robot.send_action({"gripper.pos": 0.05})
        assert result == {"gripper.pos": 0.05}
        robot._databus.set_target_distance.assert_called_once_with(0.05)
    finally:
        robot.disconnect()


def test_send_action_clamps_to_max(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        result = robot.send_action({"gripper.pos": 99.0})
        assert result == {"gripper.pos": cfg.gripper_max_distance}
        robot._databus.set_target_distance.assert_called_once_with(cfg.gripper_max_distance)
    finally:
        robot.disconnect()


def test_send_action_clamps_to_min(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        result = robot.send_action({"gripper.pos": -1.0})
        assert result == {"gripper.pos": cfg.gripper_min_distance}
        robot._databus.set_target_distance.assert_called_once_with(cfg.gripper_min_distance)
    finally:
        robot.disconnect()


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def test_calibrate_forwards_to_databus(patches):
    cfg = UmiGripperConfig(side="left", camera_count=0)
    robot = UmiGripper(cfg)
    robot.connect()
    try:
        robot.calibrate()
        robot._databus.calib_encoder.assert_called_once()
    finally:
        robot.disconnect()


def test_calibrate_no_op_when_not_connected():
    robot = UmiGripper(UmiGripperConfig(side="left"))
    # Should not raise even though databus is None
    robot.calibrate()
