"""JointSlewLimiter 单测。

限速器改了两件事，各自都能悄悄退化成难查的手感问题，所以都钉死：
锚点从「实测位置」换成「上一次指令」（锚在实测上会让滞后自我强化），
单位从「每拍弧度」换成 rad/s（指令频率一变，同一个数字的物理含义就变了）。
"""

import numpy as np
import pytest

from lerobot.rlt.envs.piper import JointSlewLimiter, rate_limit_joints


def _v(*vals) -> np.ndarray:
    """7 维关节向量（6 关节 + 夹爪）。"""
    arr = np.zeros(7, dtype=np.float32)
    arr[: len(vals)] = vals
    return arr


def test_unsaturated_is_exact_passthrough():
    lim = JointSlewLimiter(max_vel_rad_s=10.0)
    lim.reset(_v(0.0))
    out, saturated = lim(_v(0.01), _v(0.0), dt=1 / 30)
    assert not saturated
    assert out[0] == pytest.approx(0.01)


def test_limit_is_velocity_so_faster_ticks_get_smaller_steps():
    """同一个 rad/s 上限，dt 减半则单步上限减半 —— 这正是换单位的目的。"""
    target, measured = _v(1.0), _v(0.0)

    slow = JointSlewLimiter(max_vel_rad_s=3.0)
    slow.reset(measured)
    out_slow, _ = slow(target, measured, dt=1 / 30)

    fast = JointSlewLimiter(max_vel_rad_s=3.0)
    fast.reset(measured)
    out_fast, _ = fast(target, measured, dt=1 / 60)

    assert out_slow[0] == pytest.approx(3.0 / 30)
    assert out_fast[0] == pytest.approx(3.0 / 60)


def test_anchor_is_previous_command_not_measurement():
    """从臂完全不动时，指令仍应每拍稳步推进。

    旧实现锚在实测上：从臂不动 -> 每拍都从同一个原点起步 -> 指令永远只前进一步，
    这就是「一顿一顿」的正反馈。锚在上次指令上则单调推进。
    """
    lim = JointSlewLimiter(max_vel_rad_s=3.0, max_lead_rad=0.0)  # 关掉 lead 夹紧
    stuck = _v(0.0)
    lim.reset(stuck)

    step = 3.0 / 30
    outs = [lim(_v(1.0), stuck, dt=1 / 30)[0][0] for _ in range(3)]
    assert outs == pytest.approx([step, 2 * step, 3 * step])

    # 对照：锚在实测上则原地踏步
    stalled = [rate_limit_joints(_v(1.0), stuck, step)[0][0] for _ in range(3)]
    assert stalled == pytest.approx([step, step, step])


def test_direction_is_preserved_by_uniform_scaling():
    """整条向量等比缩放，不是逐关节裁剪 —— 否则轨迹会被悄悄掰弯。"""
    lim = JointSlewLimiter(max_vel_rad_s=1.0, max_lead_rad=0.0)
    lim.reset(_v(0.0, 0.0))
    out, saturated = lim(_v(1.0, 0.5), _v(0.0, 0.0), dt=1 / 30)
    assert saturated
    assert out[1] / out[0] == pytest.approx(0.5)


def test_gripper_is_never_rate_limited():
    """夹爪是开度不是角度，限速对它没有物理意义。"""
    lim = JointSlewLimiter(max_vel_rad_s=0.001)
    lim.reset(_v(0.0))
    target = _v(1.0, 0, 0, 0, 0, 0)
    target[6] = 0.07
    out, _ = lim(target, _v(0.0), dt=1 / 30)
    assert out[6] == pytest.approx(0.07)


def test_max_lead_clamps_command_runaway_on_a_blocked_arm():
    """从臂被卡住时指令不能一路跑飞，否则障碍解除瞬间会猛冲。"""
    lim = JointSlewLimiter(max_vel_rad_s=10.0, max_lead_rad=0.2)
    blocked = _v(0.0)
    lim.reset(blocked)
    for _ in range(50):
        out, _ = lim(_v(5.0), blocked, dt=1 / 30)
    assert out[0] <= 0.2 + 1e-6


def test_reset_reseeds_from_measurement():
    """接管时必须用实测位姿重新播种，否则第一拍从陈旧指令起跳。"""
    lim = JointSlewLimiter(max_vel_rad_s=3.0)
    assert not lim.seeded
    lim.reset(_v(0.0))
    for _ in range(5):
        lim(_v(1.0), _v(0.0), dt=1 / 30)

    lim.reset(_v(0.9))
    out, _ = lim(_v(1.0), _v(0.9), dt=1 / 30)
    assert out[0] == pytest.approx(1.0)  # 只差 0.1 rad，未饱和，直接到位


def test_zero_limit_disables_limiting():
    """两个旋钮相互独立：限速关掉不等于把 lead 安全网也关掉。"""
    lim = JointSlewLimiter(max_vel_rad_s=0.0, max_lead_rad=0.0)
    lim.reset(_v(0.0))
    out, saturated = lim(_v(5.0), _v(0.0), dt=1 / 30)
    assert not saturated
    assert out[0] == pytest.approx(5.0)

    # 限速关掉、lead 安全网仍开着时，指令依旧被拦在实测位姿附近
    guarded = JointSlewLimiter(max_vel_rad_s=0.0, max_lead_rad=0.2)
    guarded.reset(_v(0.0))
    out, _ = guarded(_v(5.0), _v(0.0), dt=1 / 30)
    assert out[0] == pytest.approx(0.2)
