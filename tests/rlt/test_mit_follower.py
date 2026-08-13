"""跟随臂 MIT 阻抗控制的单测（不碰硬件）。

这层直接把跟随臂从位置控制换成力矩控制，出错的后果是手臂垮下来或自激振荡，
所以把参数范围、模式切换时序、以及"退出必须切回位置模式"都钉死。
"""

import numpy as np
import pytest
from lerobot.rlt.mit_follower import (
    KD_RANGE,
    KP_RANGE,
    MIT_ON,
    MOVE_J,
    MOVE_M,
    PiperMitFollower,
    read_joint_velocity,
)
from lerobot.rlt.piper_env import jitter_rms


class FakeArm:
    """记录所有下发的 SDK 调用。"""

    def __init__(self):
        self.modes: list[tuple] = []
        self.mit_cmds: list[tuple] = []
        self.speeds = np.zeros(6)

    def MotionCtrl_2(self, ctrl, move, spd, mit):  # noqa: N802
        self.modes.append((ctrl, move, spd, mit))

    def JointMitCtrl(self, motor, pos, vel, kp, kd, tau):  # noqa: N802
        self.mit_cmds.append((motor, pos, vel, kp, kd, tau))

    def GetArmHighSpdInfoMsgs(self):  # noqa: N802
        return type(
            "HS", (),
            {f"motor_{i + 1}": type("M", (), {"motor_speed": self.speeds[i] * 1000})()
             for i in range(6)},
        )()


class FakeGravity:
    """假重力模型：返回固定力矩。"""

    def __init__(self, tau=None):
        self.tau = np.array([0.0, 3.0, 2.5, 0.2, 0.2, 0.0]) if tau is None else np.asarray(tau)

    def _read_q_v(self):
        return np.zeros(6), np.zeros(6)

    def _compute_gravity_torque(self, q, v):
        return self.tau


def test_start_switches_to_mit_mode_and_stop_switches_back():
    """退出必须切回位置模式 —— 否则手臂会一直停在柔顺状态。"""
    arm = FakeArm()
    mit = PiperMitFollower(arm=arm)
    mit.start()
    assert arm.modes[-1][1] == MOVE_M and arm.modes[-1][3] == MIT_ON

    mit.stop()
    assert arm.modes[-1][1] == MOVE_J, "stop() 必须切回 MOVE J"


def test_start_and_stop_are_idempotent():
    arm = FakeArm()
    mit = PiperMitFollower(arm=arm)
    mit.start(); mit.start()
    assert sum(m[1] == MOVE_M for m in arm.modes) == 1
    mit.stop(); mit.stop()
    assert sum(m[1] == MOVE_J for m in arm.modes) == 1


def test_send_before_start_is_refused():
    """没切模式就发 MIT 帧会被静默忽略，必须显式报错而不是假装成功。"""
    mit = PiperMitFollower(arm=FakeArm())
    with pytest.raises(RuntimeError, match="未 start"):
        mit.send(np.zeros(6))


def test_send_emits_one_command_per_arm_joint_with_gains():
    arm = FakeArm()
    mit = PiperMitFollower(arm=arm, kp=42.0, kd=1.5, gravity_model=FakeGravity())
    mit.start()
    pos = np.arange(6, dtype=float) * 0.1
    vel = np.full(6, 0.3)
    mit.send(pos, vel)

    assert len(arm.mit_cmds) == 6, "6 个臂关节各一条；夹爪不走 MIT"
    assert [c[0] for c in arm.mit_cmds] == [1, 2, 3, 4, 5, 6]
    for i, cmd in enumerate(arm.mit_cmds):
        _motor, p, v, kp, kd, tau = cmd
        assert p == pytest.approx(pos[i])
        assert v == pytest.approx(vel[i]), "速度前馈必须真的传下去"
        assert kp == 42.0 and kd == 1.5
    # t_ref 用的是重力模型的输出
    assert arm.mit_cmds[1][5] == pytest.approx(3.0)


def test_gravity_torque_is_clipped_to_the_limit():
    arm = FakeArm()
    mit = PiperMitFollower(
        arm=arm, torque_limit=1.0, gravity_model=FakeGravity([0, 99, -99, 0, 0, 0])
    )
    mit.start()
    mit.send(np.zeros(6))
    taus = [c[5] for c in arm.mit_cmds]
    assert max(taus) == pytest.approx(1.0)
    assert min(taus) == pytest.approx(-1.0)


def test_no_gravity_model_means_zero_feedforward():
    arm = FakeArm()
    mit = PiperMitFollower(arm=arm, gravity_model=None)
    mit.start()
    mit.send(np.zeros(6))
    assert all(c[5] == 0.0 for c in arm.mit_cmds)


def test_gains_outside_sdk_range_are_rejected():
    """kp/kd 超出 SDK 范围会被固件当成未定义行为，必须在构造时就拦下。"""
    with pytest.raises(ValueError, match="kp"):
        PiperMitFollower(arm=FakeArm(), kp=KP_RANGE[1] + 1)
    with pytest.raises(ValueError, match="kd"):
        PiperMitFollower(arm=FakeArm(), kd=KD_RANGE[1] + 1)


def test_velocity_feedforward_reads_leader_speed_in_rad_per_s():
    """SDK 的 motor_speed 单位是 0.001 rad/s。"""
    arm = FakeArm()
    arm.speeds = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0])
    assert np.allclose(read_joint_velocity(arm), arm.speeds)


# ------------------------------------------------------------------ 抖动度量
def test_jitter_rms_ignores_smooth_motion():
    """匀速运动不算抖 —— 一阶差分恒定，二阶差分为 0。"""
    t = np.arange(200)
    ramp = np.stack([t * 0.01] * 6, axis=1)
    assert jitter_rms(ramp, dt=1 / 30) == pytest.approx(0.0, abs=1e-9)


def test_jitter_rms_catches_alternating_noise():
    """方向反复翻转的高频成分才是抖动。"""
    t = np.arange(200)
    smooth = np.stack([t * 0.01] * 6, axis=1)
    shaky = smooth + np.stack([((-1.0) ** t) * 0.002] * 6, axis=1)
    assert jitter_rms(shaky, dt=1 / 30) > 10 * max(jitter_rms(smooth, dt=1 / 30), 1e-9)


def test_jitter_rms_needs_three_samples():
    assert np.isnan(jitter_rms(np.zeros((2, 6)), dt=1 / 30))


# ------------------------------------------------- 报告里的抖动判定
def _run_checker(follower_cls, **kw):
    import sys
    sys.path.insert(0, "tests")
    from rlt.test_teleop_check import FakeLeader
    from lerobot.rlt.teleop_check import TeleopCheckConfig, TeleopChecker

    cfg = TeleopCheckConfig(
        control_hz=30.0, duration_s=1.5, use_cameras=False, align_settle_s=0.0,
        keyboard_backend="none", engage_on_start=True, **kw,
    )
    checker = TeleopChecker(cfg)
    checker.leader = FakeLeader()
    checker.robot = follower_cls()
    checker.run()
    rep = checker.report()
    return next(v for v in rep["verdicts"] if v["check"] == "跟随臂抖动"), rep


def _shaky_follower_cls():
    import sys
    sys.path.insert(0, "tests")
    from rlt.test_teleop_check import FakeFollower
    from lerobot.rlt.piper_env import JOINT_ORDER

    class Shaky(FakeFollower):
        """跟随臂自己在振：实测位姿叠加一个交替扰动。"""

        def get_observation(self):
            self.t = getattr(self, "t", 0) + 1
            j = self.joints + ((-1.0) ** self.t) * 0.004
            return dict(zip(JOINT_ORDER, j.tolist(), strict=True))

    return Shaky


def test_smooth_follower_passes_the_jitter_check():
    """匀速拖动时指令抖动≈0，比值会爆到几万倍 —— 必须先过绝对阈值，否则全是假阳性。"""
    import sys
    sys.path.insert(0, "tests")
    from rlt.test_teleop_check import FakeFollower

    verdict, _rep = _run_checker(FakeFollower)
    assert verdict["status"] == "PASS"
    assert "低于可察觉量级" in verdict["detail"]


def test_oscillating_follower_is_blamed_and_pointed_at_mit():
    verdict, rep = _run_checker(_shaky_follower_cls())
    assert verdict["status"] == "FAIL"
    assert "跟随臂自身在振荡" in verdict["detail"]
    # position 模式下要明确指路到 MIT，而不是只报数字
    assert "follower_control=mit" in verdict["detail"]
    assert rep["jitter_rms_rad_s2"]["measured"] > 5.0


def test_report_records_the_follower_control_mode():
    import sys
    sys.path.insert(0, "tests")
    from rlt.test_teleop_check import FakeFollower

    _verdict, rep = _run_checker(FakeFollower)
    assert rep["follower_control"] == "position"
