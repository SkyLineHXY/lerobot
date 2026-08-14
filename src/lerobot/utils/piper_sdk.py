#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

PIPER_JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
)
PIPER_JOINT_ACTION_KEYS = tuple(f"{joint}.pos" for joint in PIPER_JOINT_NAMES)
PIPER_ACTION_KEYS = PIPER_JOINT_ACTION_KEYS + ("gripper.pos",)
PIPER_CTRL_MODE_STANDBY = 0x00
PIPER_CTRL_MODE_CAN = 0x01
PIPER_CTRL_MODE_TEACH = 0x02
PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT = 0x06
# 这两个模式下机械臂扮演"主臂/示教器"角色，不接受 CAN 指令，必须先切回从动
PIPER_TEACH_CTRL_MODES = frozenset({PIPER_CTRL_MODE_TEACH, PIPER_CTRL_MODE_LINKAGE_TEACH_INPUT})

# MasterSlaveConfig 的角色码：0xFC = 设为从动臂（接受 CAN 指令）
PIPER_ROLE_SLAVE = 0xFC

# 每条臂大约广播 16 种消息、各 ~188Hz，合计 ~3000 帧/秒，而 piper_sdk 的 ReadCan 线程
# 会在 Python 里解析每一帧（全程持 GIL）。双臂 4 个接口就是 ~12300 帧/秒，是遥操作
# 回路最大的 GIL 竞争源。下面两组是各角色真正会被读到的 ID，其余在内核里就丢掉。
PIPER_CAN_ID_ARM_STATUS = 0x2A1  # ctrl_mode 守卫要读
PIPER_CAN_IDS_JOINT_FEEDBACK = (0x2A5, 0x2A6, 0x2A7)
PIPER_CAN_ID_GRIPPER_FEEDBACK = 0x2A8
# 0x251-0x256 是高速驱动器反馈，GetArmHighSpdInfoMsgs 的 motor_speed 从这里来。
# 主臂重力补偿的 RNEA 科氏项要用它，滤掉不会报错，只会让速度项恒为 0。
PIPER_CAN_IDS_HIGH_SPEED_FEEDBACK = tuple(range(0x251, 0x257))

PIPER_FOLLOWER_CAN_IDS = (
    PIPER_CAN_ID_ARM_STATUS,
    *PIPER_CAN_IDS_JOINT_FEEDBACK,
    PIPER_CAN_ID_GRIPPER_FEEDBACK,
)
PIPER_LEADER_CAN_IDS = PIPER_FOLLOWER_CAN_IDS + PIPER_CAN_IDS_HIGH_SPEED_FEEDBACK


def apply_piper_can_filters(arm: Any, role: str) -> int:
    """给一条臂的 socketcan 套接字装上接收过滤器，返回保留的 ID 数量。

    过滤下沉到内核，被滤掉的帧根本不会进入 Python，因此这是唯一能真正降低
    ReadCan 线程 GIL 占用的办法（python-can 的 set_filters 对 socketcan 走
    SO_CAN_RAW_FILTER）。

    失败只记日志不抛：过滤纯属优化，任何一个 SDK 版本换了内部属性名都不该让
    整场采集起不来。
    """
    if role == "leader":
        can_ids = PIPER_LEADER_CAN_IDS
    elif role == "follower":
        can_ids = PIPER_FOLLOWER_CAN_IDS
    else:
        raise ValueError(f"未知的 role {role!r}；可选 'leader' / 'follower'")

    bus = _find_can_bus(arm)
    if bus is None:
        logger.warning("没能拿到 piper_sdk 的 can bus 对象，跳过 CAN 过滤（不影响功能）。")
        return 0

    try:
        bus.set_filters([{"can_id": cid, "can_mask": 0x7FF, "extended": False} for cid in can_ids])
    except Exception:
        logger.exception("设置 CAN 过滤器失败，继续按不过滤运行。")
        return 0

    logger.info("已对 %s 启用 CAN 过滤，保留 %d 个 ID。", role, len(can_ids))
    return len(can_ids)


def _find_can_bus(arm: Any) -> Any:
    """从 C_PiperInterface_V2 里挖出 python-can 的 Bus。

    SDK 把它藏在私有属性里，且不同版本层级不同，所以按候选路径找而不是写死一条。
    """
    candidates = (
        "_C_PiperInterface_V2__arm_can",
        "_C_PiperInterface__arm_can",
        "arm_can",
    )
    for name in candidates:
        channel = getattr(arm, name, None)
        bus = getattr(channel, "bus", None)
        if bus is not None and hasattr(bus, "set_filters"):
            return bus
    # 兜底：直接在实例字典里找带 set_filters 的 bus
    for value in vars(arm).values():
        bus = getattr(value, "bus", None)
        if bus is not None and hasattr(bus, "set_filters"):
            return bus
    return None


def milli_to_unit(value: float | int) -> float:
    return float(value) / 57324.840764


def unit_to_milli(value: float | int) -> int:
    return int(round(float(value) * 57324.840764))


@lru_cache(maxsize=1)
def get_piper_sdk() -> tuple[type[Any], Any]:
    try:
        from piper_sdk import C_PiperInterface_V2, LogLevel

        return C_PiperInterface_V2, LogLevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `piper_sdk`. Install Evo-RL dependencies first (for example: `pip install -e .`)."
        ) from exc


def parse_piper_log_level(level_name: str) -> Any:
    _, log_level_enum = get_piper_sdk()
    normalized = level_name.upper()
    try:
        return getattr(log_level_enum, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Invalid Piper log level '{level_name}'. "
            "Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL, SILENT."
        ) from exc


def wait_enable_piper(arm: Any, timeout_s: float, retry_interval_s: float = 0.2) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    interval_s = max(0.01, retry_interval_s)
    while time.monotonic() < deadline:
        if bool(arm.EnablePiper()):
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            break
        time.sleep(min(interval_s, remaining_s))
    return False


def _piper_mode_to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "value") and isinstance(value.value, int):
        return int(value.value)
    if hasattr(value, "__int__"):
        return int(value)
    return None


def read_piper_ctrl_mode(
    arm: Any,
    timeout_s: float = 1.0,
    poll_s: float = 0.02,
    min_timestamp: float = 0.0,
) -> int | None:
    """读取机械臂当前的 ctrl_mode。读不到（CAN 无数据）返回 None。

    `min_timestamp`：只接受时间戳严格晚于它的状态帧。SDK 缓存的是最后一帧，
    切完模式马上再读很可能拿到切换之前的旧值，配合 :func:`read_piper_status_timestamp`
    使用可以确保读到的是一帧新数据。
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        status_msg = arm.GetArmStatus()
        stamp = float(getattr(status_msg, "time_stamp", 0.0) or 0.0)
        if stamp > 0.0 and stamp > min_timestamp:
            arm_status = getattr(status_msg, "arm_status", None)
            if arm_status is not None:
                mode = _piper_mode_to_int(getattr(arm_status, "ctrl_mode", None))
                if mode is not None:
                    return mode
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.005, poll_s))


def read_piper_status_timestamp(arm: Any) -> float:
    """当前缓存状态帧的时间戳，用作 :func:`read_piper_ctrl_mode` 的新鲜度基准。"""
    return float(getattr(arm.GetArmStatus(), "time_stamp", 0.0) or 0.0)


def seed_piper_position_target(arm: Any) -> bool:
    """把 CAN 控制模式的位置目标预置为当前实测位姿。

    从示教模式切到 CAN 控制模式时，位置环会去追固件里**上一次**的目标值——那可能是
    很久以前的一个位姿，手臂会猛地弹过去。先用当前实测关节角下发一次 JointCtrl，
    目标即当前位置，切模式时手臂就停在原地。

    JointCtrl 与 `GetArmJointMsgs()` 的单位相同（0.001 度），所以这里直接透传整数。
    """
    joint_state = getattr(arm.GetArmJointMsgs(), "joint_state", None)
    if joint_state is None:
        return False
    joints = [getattr(joint_state, f"joint_{i}", None) for i in range(1, 7)]
    if any(j is None for j in joints):
        return False
    arm.JointCtrl(*(int(j) for j in joints))
    return True


def recover_piper_from_teach_mode(
    arm: Any,
    *,
    interface_name: str,
    attempts: int = 3,
    settle_s: float = 0.5,
    timeout_s: float = 1.0,
    poll_s: float = 0.02,
    speed_ratio: int = 30,
) -> int | None:
    """把处于示教/主臂角色的机械臂切回从动模式，返回切换后的 ctrl_mode。

    分两步，每轮都先看第一步是否已经够了：

    1. `MasterSlaveConfig(0xFC)` —— 把**角色**改回从动臂。
    2. `MotionCtrl_2(0x01, 0x01, ...)` —— 角色回来了但控制模式仍停在示教时，
       显式请求 CAN 控制模式（先预置位置目标，避免手臂弹跳）。

    返回 None 表示始终读不到状态；返回值仍在示教模式表示恢复失败（通常意味着
    这台臂的固件确实需要断电重启）。
    """
    mode: int | None = None
    for attempt in range(1, max(1, attempts) + 1):
        stamp = read_piper_status_timestamp(arm)
        arm.MasterSlaveConfig(PIPER_ROLE_SLAVE, 0x00, 0x00, 0x00)
        if settle_s > 0:
            time.sleep(settle_s)
        mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s, min_timestamp=stamp)
        if mode is not None and mode not in PIPER_TEACH_CTRL_MODES:
            return mode

        stamp = read_piper_status_timestamp(arm)
        seed_piper_position_target(arm)
        arm.MotionCtrl_2(0x01, 0x01, int(speed_ratio), 0x00)
        if settle_s > 0:
            time.sleep(settle_s)
        mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s, min_timestamp=stamp)
        if mode is not None and mode not in PIPER_TEACH_CTRL_MODES:
            return mode

        logger.warning(
            "[%s] 第 %d/%d 次切回从动模式未生效 (ctrl_mode=%s)，重试中……",
            interface_name,
            attempt,
            attempts,
            "读取失败" if mode is None else f"0x{mode:02X}",
        )
    return mode


def guard_piper_ctrl_mode_on_connect(
    arm: Any,
    *,
    interface_name: str,
    timeout_s: float = 0.5,
    poll_s: float = 0.02,
    settle_s: float = 0.05,
    auto_recover: bool = True,
    recover_attempts: int = 3,
    recover_settle_s: float = 0.5,
) -> None:
    """连接后确认机械臂处于可接受 CAN 指令的模式。

    默认在检测到示教模式时**自动切回从动模式**再继续；只有自动恢复失败才报错。
    `auto_recover=False` 恢复旧行为（发一次 0xFC 后直接抛错，要求断电重启）。
    """
    mode = read_piper_ctrl_mode(arm, timeout_s=timeout_s, poll_s=poll_s)
    if mode is None:
        raise RuntimeError(
            f"[{interface_name}] could not read arm ctrl_mode within {timeout_s:.2f}s. "
            "Check CAN wiring/power and rerun."
        )
    if mode not in PIPER_TEACH_CTRL_MODES:
        return

    if not auto_recover:
        arm.MasterSlaveConfig(PIPER_ROLE_SLAVE, 0x00, 0x00, 0x00)
        if settle_s > 0:
            time.sleep(settle_s)
        raise RuntimeError(
            f"[{interface_name}] arm is in master/teaching role (ctrl_mode=0x{mode:02X}). "
            "Follower role command (0xFC) has been sent. Power-cycle this arm, then retry."
        )

    logger.warning(
        "[%s] 机械臂处于示教/主臂模式 (ctrl_mode=0x%02X)，正在自动切回从动模式；"
        "切换瞬间位置环会接管，请勿把手放在运动范围内。",
        interface_name,
        mode,
    )
    recovered = recover_piper_from_teach_mode(
        arm,
        interface_name=interface_name,
        attempts=recover_attempts,
        settle_s=recover_settle_s,
        timeout_s=max(timeout_s, 1.0),
        poll_s=poll_s,
    )
    if recovered is None or recovered in PIPER_TEACH_CTRL_MODES:
        state = "读取失败" if recovered is None else f"0x{recovered:02X}"
        raise RuntimeError(
            f"[{interface_name}] 自动切回从动模式失败（{recover_attempts} 次尝试后 ctrl_mode={state}）。"
            "从动指令 (0xFC) 已下发，请将这台机械臂断电重启后重试。"
        )
    logger.info("[%s] 已切回从动模式 (ctrl_mode=0x%02X)，继续连接。", interface_name, recovered)
