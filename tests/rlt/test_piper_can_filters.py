"""CAN 接收过滤器的 ID 集合与容错单测。

滤错 ID 的症状是「某个读数永远不更新」而不是报错 —— 尤其 0x251-0x256 一旦漏掉，
主臂重力补偿的速度项会静默恒为 0。所以把两个角色的 ID 集合钉死。
"""

import pytest

from lerobot.utils.piper_sdk import (
    PIPER_CAN_IDS_HIGH_SPEED_FEEDBACK,
    PIPER_FOLLOWER_CAN_IDS,
    PIPER_LEADER_CAN_IDS,
    apply_piper_can_filters,
)


class FakeBus:
    def __init__(self):
        self.filters = None

    def set_filters(self, filters):
        self.filters = filters


class FakeChannel:
    def __init__(self, bus):
        self.bus = bus


class FakeArm:
    """模拟 C_PiperInterface_V2：bus 藏在名字重整后的私有属性里。"""

    def __init__(self, bus):
        self._C_PiperInterface_V2__arm_can = FakeChannel(bus)


def test_follower_keeps_only_joint_gripper_and_status():
    assert set(PIPER_FOLLOWER_CAN_IDS) == {0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8}


def test_leader_additionally_keeps_high_speed_feedback():
    """主臂重力补偿的 RNEA 科氏项要读 motor_speed，来源就是 0x251-0x256。"""
    assert set(PIPER_CAN_IDS_HIGH_SPEED_FEEDBACK) == {0x251, 0x252, 0x253, 0x254, 0x255, 0x256}
    assert set(PIPER_LEADER_CAN_IDS) == set(PIPER_FOLLOWER_CAN_IDS) | set(
        PIPER_CAN_IDS_HIGH_SPEED_FEEDBACK
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [("follower", PIPER_FOLLOWER_CAN_IDS), ("leader", PIPER_LEADER_CAN_IDS)],
)
def test_filters_are_pushed_to_the_bus(role, expected):
    bus = FakeBus()
    n = apply_piper_can_filters(FakeArm(bus), role)

    assert n == len(expected)
    assert [f["can_id"] for f in bus.filters] == list(expected)
    # 标准帧 11 位掩码，全匹配
    assert all(f["can_mask"] == 0x7FF and f["extended"] is False for f in bus.filters)


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="未知的 role"):
        apply_piper_can_filters(FakeArm(FakeBus()), "master")


def test_missing_bus_degrades_instead_of_raising():
    """过滤纯属优化：SDK 换了内部属性名不该让整场采集起不来。"""

    class NoBus:
        pass

    assert apply_piper_can_filters(NoBus(), "follower") == 0


def test_set_filters_failure_degrades_instead_of_raising():
    class ExplodingBus(FakeBus):
        def set_filters(self, filters):
            raise OSError("socket 已关闭")

    assert apply_piper_can_filters(FakeArm(ExplodingBus()), "leader") == 0
