from unittest.mock import MagicMock

from lerobot.robots.franka.config_franka import FrankaConfig
from lerobot.robots.franka.franka import EE_POSE_KEYS, Franka


def _make_connected_franka() -> tuple[Franka, MagicMock]:
    robot = Franka(FrankaConfig(use_gripper=True, gripper_speed=0.1, gripper_force=0.2))
    client = MagicMock()
    robot._robot = client
    robot._is_connected = True
    return robot, client


def _action_with_gripper(width: float) -> dict[str, float]:
    action = {key: 0.0 for key in EE_POSE_KEYS}
    action["gripper.width"] = width
    return action


def test_repeated_close_command_only_starts_one_grasp() -> None:
    robot, client = _make_connected_franka()
    robot._last_gripper_state = {"width": 0.08, "is_grasped": False, "is_moving": False}

    action = _action_with_gripper(0.0)
    robot.send_action(action)
    robot.send_action(action)

    assert client.gripper_grasp.call_count == 1
    client.gripper_grasp.assert_called_once_with(
        speed=0.1,
        force=0.2,
        grasp_width=0.0,
        blocking=False,
    )
    client.gripper_goto.assert_not_called()


def test_open_after_grasp_sends_goto() -> None:
    robot, client = _make_connected_franka()
    robot._last_gripper_state = {"width": 0.08, "is_grasped": False, "is_moving": False}

    robot.send_action(_action_with_gripper(0.0))
    robot._last_gripper_state = {"width": 0.0, "is_grasped": True, "is_moving": False}
    robot.send_action(_action_with_gripper(0.08))

    assert client.gripper_grasp.call_count == 1
    client.gripper_goto.assert_called_once_with(
        width=0.08,
        speed=0.1,
        force=0.2,
        blocking=False,
    )
