"""SpaceMouse 接口冒烟测试。

交互式测试：插上 SpaceMouse 后运行，实时打印动作向量与事件。按 Ctrl+C 退出。
"""

import time

import numpy as np

from lerobot.teleoperators.spacemouse.spacemouse_expert import SpaceMouseExpert


def test_spacemouse_expert() -> None:
    """直接测试 SpaceMouseExpert：10 Hz 打印原始输出。"""
    expert = SpaceMouseExpert()
    print("SpaceMouseExpert 已连接，移动 SpaceMouse 可见输出，按 Ctrl+C 退出。")
    with np.printoptions(precision=3, suppress=True):
        try:
            while True:
                action, buttons = expert.get_action()
                print(f"action={action}  buttons={buttons}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            expert.close()


def test_spacemouse_teleop() -> None:
    """通过公共 API 测试 SpaceMouseTeleop（含 action_dict 与 events 输出）。"""
    from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    cfg = SpaceMouseTeleopConfig(id="dev0", use_gripper=True, use_rotation=True)
    teleop = make_teleoperator_from_config(cfg)
    assert isinstance(teleop, SpaceMouseTeleop), f"期望 SpaceMouseTeleop，得到 {type(teleop)}"

    teleop.connect()
    print(f"action_features: {teleop.action_features}")
    print("SpaceMouseTeleop 已连接，按 Ctrl+C 退出。")
    try:
        while True:
            action = teleop.get_action()
            events = teleop.get_teleop_events()
            print(f"action={action}")
            # print(f"events={events}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    test_spacemouse_teleop()
