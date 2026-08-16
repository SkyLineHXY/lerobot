"""teleop_check 的无硬件冒烟测试。

真机验证脚本本身也会出错，而它出错的时候人是站在机械臂旁边的。这里用假的
主臂 / 跟随臂把整个循环、打点、报告、结论跑一遍，保证上机时挂掉的只可能是
硬件而不是脚本。跟随臂被建模成"落后 2 拍"，用来确认端到端延迟估计是对的。
"""

import numpy as np
import pytest

from lerobot.rlt.envs.piper import (
    JOINT_ORDER,
    follower_action_to_leader,
    leader_action_to_follower,
    rate_limit_joints,
)
from lerobot.rlt.teleop_check import TeleopCheckConfig, TeleopChecker

LEADER_KEYS = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
LAG_STEPS = 2


class FakeArm:
    def __init__(self):
        self.t = 0

    def GetArmJointMsgs(self):  # noqa: N802 - 模仿 Piper SDK 的驼峰命名
        self.t += 1
        return type("Msg", (), {"time_stamp": float(self.t)})()


class FakeLeader:
    """人在匀速拖动主臂：从跟随臂零位出发，每拍各关节前进一点点。

    起点必须和跟随臂对齐，否则接管安全门会（正确地）拒绝释放主臂。
    """

    def __init__(self):
        self.is_connected = True
        self.arm = FakeArm()
        self.k = 0
        self.manual = False
        self.feedback: list[dict] = []

    def get_raw_action(self):
        self.k += 1
        act = {
            key: 0.002 * self.k * (1.0 + 0.2 * i)
            for i, key in enumerate(LEADER_KEYS[:6])
        }
        act["gripper.pos"] = 0.03
        return act

    def get_action(self):
        return self.get_raw_action()

    def send_feedback(self, feedback):
        assert set(feedback) == set(LEADER_KEYS), "交接回位必须带上 gripper.pos"
        self.feedback.append(feedback)

    def set_manual_control(self, enabled):
        self.manual = enabled

    def disconnect(self):
        self.is_connected = False


class FakeFollower:
    """跟随臂：实际位姿是 LAG_STEPS 拍之前收到的目标。"""

    def __init__(self):
        self.is_connected = True
        self.cameras = {}
        self.joints = np.zeros(7, dtype=np.float32)
        self.queue: list[np.ndarray] = []
        self.n_sent = 0

    def get_observation(self):
        return dict(zip(JOINT_ORDER, self.joints.tolist(), strict=True))

    def send_action(self, action):
        self.n_sent += 1
        self.queue.append(np.array([action[k] for k in JOINT_ORDER], dtype=np.float32))
        if len(self.queue) > LAG_STEPS:
            self.joints = self.queue.pop(0)
        return action

    def disconnect(self):
        self.is_connected = False


def _checker(**kw) -> TeleopChecker:
    cfg = TeleopCheckConfig(
        control_hz=200.0,
        duration_s=0.5,
        max_joint_step_rad=0.05,
        use_cameras=False,
        **kw,
    )
    checker = TeleopChecker(cfg)
    checker.leader = FakeLeader()
    checker.robot = FakeFollower()
    return checker


def test_key_mapping_round_trips_between_leader_and_follower():
    """主臂 gripper.pos <-> 跟随臂 joint_7.pos —— 这个映射错了干预会直接 KeyError。"""
    leader_action = {k: float(i) for i, k in enumerate(LEADER_KEYS)}
    follower = leader_action_to_follower(leader_action)
    assert set(follower) == set(JOINT_ORDER)
    assert follower["joint_7.pos"] == leader_action["gripper.pos"]
    assert follower_action_to_leader(follower) == leader_action


def test_key_mapping_reports_missing_keys():
    with pytest.raises(KeyError, match="gripper.pos"):
        leader_action_to_follower(dict.fromkeys(LEADER_KEYS[:6], 0.0))


def test_rate_limit_scales_whole_step_and_flags_saturation():
    current = np.zeros(7, dtype=np.float32)
    target = np.array([0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.03], dtype=np.float32)
    limited, saturated = rate_limit_joints(target, current, 0.05)
    assert saturated
    # 方向必须保持：整条按比例缩放，而不是逐关节裁剪
    assert np.isclose(limited[0] / limited[1], target[0] / target[1])
    assert np.isclose(np.abs(limited[:6]).max(), 0.05)
    assert limited[6] == target[6], "夹爪不参与关节限速"

    small = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    limited, saturated = rate_limit_joints(small, current, 0.05)
    assert not saturated
    assert np.allclose(limited, small)


def test_loop_runs_and_measures_end_to_end_lag():
    checker = _checker()
    checker.keys._intervene = True  # 模拟操作员按下空格握住主臂
    checker.run()
    rep = checker.report()

    assert rep["steps"] > 20
    assert rep["engaged_steps"] > 20
    assert checker.robot.n_sent == rep["engaged_steps"], "接管期间每拍都应下发"
    # 假跟随臂内部滞后 LAG_STEPS 拍；再加 1 拍是因为一拍之内先读观测后下发，
    # 这一拍的读数反映不了这一拍的指令。真机同理，所以这就是诚实的端到端延迟。
    assert rep["tracking"]["lag_steps"] == LAG_STEPS + 1
    assert rep["latency_ms"]["busy_total"]["p50"] >= 0.0
    assert 0.0 <= rep["deadline_miss_rate"] <= 1.0
    assert {v["check"] for v in rep["verdicts"]}  # 结论不为空
    assert all(v["status"] in {"PASS", "WARN", "FAIL"} for v in rep["verdicts"])


def test_dry_run_never_commands_the_follower():
    checker = _checker(dry_run=True)
    checker.keys._intervene = True
    checker.run()
    rep = checker.report()

    assert checker.robot.n_sent == 0, "dry-run 绝不能下发动作"
    assert rep["engaged_steps"] > 0
    lag = next(v for v in rep["verdicts"] if v["check"] == "端到端跟随延迟")
    assert lag["status"] == "WARN" and "dry-run" in lag["detail"]


def test_takeover_is_refused_when_leader_is_far_from_follower():
    """首次接管前主臂没对齐时必须拒绝，否则跟随臂会被拽向主臂当前位姿。"""
    checker = _checker()
    checker.leader.k = 500  # 主臂被拖到离跟随臂很远的地方
    checker.keys._intervene = True
    checker.run()
    rep = checker.report()

    assert rep["operator"]["takeovers_refused"] >= 1
    assert checker.robot.n_sent == 0, "拒绝接管后不应下发任何动作"
    assert checker.leader.feedback, "拒绝后必须重新对齐主臂"


def test_alignment_puts_leader_back_in_command_mode():
    checker = _checker()
    checker.robot.joints = np.arange(7, dtype=np.float32) * 0.1
    checker.align()

    assert checker.leader.manual is False
    sent = checker.leader.feedback[-1]
    assert sent["joint_3.pos"] == pytest.approx(0.2)
    assert sent["gripper.pos"] == pytest.approx(0.6)


class SpyLeader(FakeLeader):
    """记录 set_manual_control 的调用序列 —— 重力补偿的开关就是它。"""

    def __init__(self):
        super().__init__()
        self.history: list[bool] = []

    def set_manual_control(self, enabled):
        self.history.append(enabled)
        super().set_manual_control(enabled)


def _spy_checker(**kw):
    cfg = TeleopCheckConfig(
        control_hz=200.0, duration_s=0.4, use_cameras=False, align_settle_s=0.0,
        keyboard_backend="none", **kw,
    )
    checker = TeleopChecker(cfg)
    checker.leader = SpyLeader()
    checker.robot = FakeFollower()
    return checker


def test_engage_on_start_actually_engages():
    """engage_on_start 必须真的进入接管，而不是开完重力补偿就被状态机撤销。

    曾经的写法是绕过按键 toggle 直接 set_manual_control(True) 并把 prev_engaged
    设成 True；主循环第一拍读到 toggle 仍是 False，就判成"松手"立刻交还 ——
    重力补偿刚起来就被关掉，整场 0 拍接管。
    """
    checker = _spy_checker(engage_on_start=True)
    checker.run()
    rep = checker.report()

    assert rep["engaged_steps"] > 5, "engage_on_start 应当全程处于接管态"
    assert checker.robot.n_sent == rep["engaged_steps"]
    # 重力补偿必须被真正打开过
    assert True in checker.leader.history, "从未调用 set_manual_control(True)"
    # 且不能开完立刻关：True 之后不该紧跟着一次撤销性的 False
    first_true = checker.leader.history.index(True)
    assert checker.leader.history[first_true:].count(False) <= 1, (
        f"重力补偿被反复开关: {checker.leader.history}"
    )


def test_engage_on_start_false_stays_disengaged():
    checker = _spy_checker(engage_on_start=False)
    checker.run()

    assert checker.report()["engaged_steps"] == 0
    assert checker.robot.n_sent == 0
    assert True not in checker.leader.history, "没按空格就不该释放主臂"


def test_gravity_comp_params_are_configurable():
    """重力补偿参数必须能从配置里调 —— tx_ratio 是唯一按手感标定的系数。"""
    import dataclasses

    from lerobot.rlt.teleop_check import LeaderCheckConfig

    names = {f.name for f in dataclasses.fields(LeaderCheckConfig)}
    for required in (
        "gravity_comp_tx_ratio",
        "gravity_comp_base_rpy_deg",
        "gravity_comp_torque_limit",
        "gravity_comp_mit_kp",
        "gravity_comp_mit_kd",
        "gravity_comp_control_hz",
    ):
        assert required in names, f"{required} 无法从 yaml 配置"

    # kp 默认必须是 0：pos_ref 恒为 0，kp>0 会把主臂往零位拽
    assert LeaderCheckConfig().gravity_comp_mit_kp == 0.0


# ------------------------------------------------------------------ 抖动度量
# 这三条原本在 test_mit_follower.py 里。jitter_rms 是 piper_env 的通用度量，与
# 从臂 MIT 无关，所以删掉那个文件时把它们迁到这里。
def test_jitter_rms_ignores_smooth_motion():
    """匀速运动不算抖 —— 一阶差分恒定，二阶差分为 0。"""
    from lerobot.rlt.envs.piper import jitter_rms

    t = np.arange(200)
    ramp = np.stack([t * 0.01] * 6, axis=1)
    assert jitter_rms(ramp, dt=1 / 30) == pytest.approx(0.0, abs=1e-9)


def test_jitter_rms_catches_alternating_noise():
    """方向反复翻转的高频成分才是抖动。"""
    from lerobot.rlt.envs.piper import jitter_rms

    t = np.arange(200)
    smooth = np.stack([t * 0.01] * 6, axis=1)
    shaky = smooth + np.stack([((-1.0) ** t) * 0.002] * 6, axis=1)
    assert jitter_rms(shaky, dt=1 / 30) > 10 * max(jitter_rms(smooth, dt=1 / 30), 1e-9)


def test_jitter_rms_needs_three_samples():
    from lerobot.rlt.envs.piper import jitter_rms

    assert np.isnan(jitter_rms(np.zeros((2, 6)), dt=1 / 30))
