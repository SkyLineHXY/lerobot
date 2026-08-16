"""重力补偿探针的判定逻辑单测（不碰硬件）。

这里最要紧的是把两件事分清楚：某个关节"下发 0 力矩"到底是**正确结果**
（重力对该轴没有力臂，例如绕基座竖直轴的 J1 和绕工具轴的 J6），还是**补偿不足**。
判反了会让人去调一个永远不会有输出的关节。
"""

import numpy as np
import pytest

from lerobot.rlt.gravity_probe import N_JOINTS, joint_verdict, render


def test_no_moment_arm_joints_report_zero_as_correct():
    """J1/J6 这类轴：重力算不出力矩，下发 0 是对的，不能报成"补偿不足"。"""
    verdict = joint_verdict(g_i=0.0, commanded_i=0.0, mit_ok=True)
    assert "无力臂" in verdict
    assert "不足" not in verdict


def test_torque_below_stiction_is_flagged_separately():
    """腕部需求本来就小，缩放后低于静摩擦时手上完全感觉不到 —— 要单独提示。"""
    # J5 典型值：需要 0.218 N·m，tx_ratio=0.2 只给 0.044
    verdict = joint_verdict(g_i=0.218, commanded_i=0.0436, mit_ok=True)
    assert "静摩擦" in verdict


def test_insufficient_compensation_on_a_loaded_joint():
    """J2/J3 这类重载关节：给得太少要明确说会往下掉。"""
    verdict = joint_verdict(g_i=3.077, commanded_i=0.615, mit_ok=True)
    assert "不足" in verdict


def test_adequate_compensation_passes():
    assert "✓" in joint_verdict(g_i=3.077, commanded_i=3.0, mit_ok=True)


def test_mit_not_latched_overrides_every_other_verdict():
    """MIT 没挂上时力矩指令根本没生效，其它判定都没有意义。"""
    for g_i, cmd in ((0.0, 0.0), (3.077, 3.0), (0.218, 0.04)):
        assert "MIT" in joint_verdict(g_i, cmd, mit_ok=False)


def test_render_covers_every_joint_and_marks_selection():
    g = np.array([0.0, 3.077, 2.718, 0.011, 0.218, 0.0])
    ratios = np.full(N_JOINTS, 0.2)
    lines = render(ratios, g, ratios * g, np.zeros(N_JOINTS), "MOVE_M(0x4)", True, selected=4)

    joint_lines = lines[2:]
    assert len(joint_lines) == N_JOINTS, "每个关节都必须单独一行，不能只显示峰值"
    # 只有被选中的关节带 * 标记
    assert joint_lines[4].startswith("*J5")
    assert sum(line.startswith("*") for line in joint_lines) == 1


def test_render_selecting_all_marks_every_joint():
    g = np.zeros(N_JOINTS)
    lines = render(g, g, g, g, "MOVE_M(0x4)", True, selected=None)[2:]
    assert all(line.startswith("*") for line in lines)


def test_per_joint_ratios_are_independent():
    """腕部必须能单独调高 —— 各关节减速比/力矩常数不同，统一标量调不出来。"""
    g = np.array([0.0, 3.077, 2.718, 0.011, 0.218, 0.0])
    ratios = np.array([0.2, 0.2, 0.2, 1.0, 1.0, 0.2])
    commanded = ratios * g

    assert commanded[4] == np.float64(0.218), "J5 提到 1.0 后应拿到全部自重补偿"
    assert commanded[1] < 0.7, "J2 不该被一起改动"
    assert "✓" in joint_verdict(g[4], commanded[4], mit_ok=True)


# --------------------------------------------------- 末端负载 / URDF 选择
def test_payload_raises_wrist_gravity_torque():
    """末端装了夹爪示教器而模型不含它，腕部重力矩会被系统性低估。

    实测：J4/J5 的真实需求约为无负载模型的 3.9 倍。这种偏差调 tx_ratio 补不回来
    —— 缩放的是一个本身就错的 g(q)，只能在某一个位姿上凑对。
    """
    pin = pytest.importorskip("pinocchio")  # noqa: F841
    from importlib import resources

    from lerobot.teleoperators.piper_leader.gravity_compensation import (
        PiperGravityCompensationLoop,
    )

    base = resources.files("lerobot").joinpath("assets/piper_description/urdf")
    common = dict(
        arm=None, control_hz=200.0, tx_ratio=(1.0,) * 6, torque_limit=8.0,
        mit_kp=0.0, mit_kd=0.0, base_rpy_deg=(0.0, 0.0, 0.0),
        mode_refresh_interval_s=1.0, move_speed_ratio=100,
        urdf_path=str(base.joinpath("piper_no_gripper_description.urdf")),
    )
    q = np.deg2rad([0, 60, -60, 90, 90, 90])
    zeros = np.zeros(6)

    bare = PiperGravityCompensationLoop(**common)._compute_gravity_torque(q, zeros)
    loaded = PiperGravityCompensationLoop(
        **common, payload_mass=0.5, payload_com=(0.0, 0.0, 0.05)
    )._compute_gravity_torque(q, zeros)

    assert abs(loaded[3]) > 3 * abs(bare[3]), "J4 的重力矩必须随末端负载显著增大"
    assert abs(loaded[1]) > abs(bare[1]), "J2 也要跟着增大"
    # 负载不该凭空给无力臂的轴造出力矩
    assert abs(loaded[0]) < 0.01 and abs(loaded[5]) < 0.01


def test_zero_payload_leaves_the_model_untouched():
    pytest.importorskip("pinocchio")
    from importlib import resources

    from lerobot.teleoperators.piper_leader.gravity_compensation import (
        PiperGravityCompensationLoop,
    )

    base = resources.files("lerobot").joinpath("assets/piper_description/urdf")
    common = dict(
        arm=None, control_hz=200.0, tx_ratio=(1.0,) * 6, torque_limit=8.0,
        mit_kp=0.0, mit_kd=0.0, base_rpy_deg=(0.0, 0.0, 0.0),
        mode_refresh_interval_s=1.0, move_speed_ratio=100,
        urdf_path=str(base.joinpath("piper_no_gripper_description.urdf")),
    )
    q = np.deg2rad([0, 60, -60, 90, 90, 90])
    a = PiperGravityCompensationLoop(**common)._compute_gravity_torque(q, np.zeros(6))
    b = PiperGravityCompensationLoop(**common, payload_mass=0.0)._compute_gravity_torque(
        q, np.zeros(6)
    )
    assert np.allclose(a, b)


def test_urdf_override_resolution():
    """gravity_comp_urdf 可以是包内相对路径，也可以是绝对路径；写错要报错。"""
    from types import SimpleNamespace

    from lerobot.teleoperators.piper_leader.piper_leader import (
        DEFAULT_PIPER_GRAVITY_URDF,
        PiperLeader,
    )

    def resolve(urdf):
        stub = SimpleNamespace(
            config=SimpleNamespace(gravity_comp_urdf=urdf),
            gravity_comp_urdf_relpath=DEFAULT_PIPER_GRAVITY_URDF,
        )
        return PiperLeader._resolve_gravity_urdf(stub)

    assert resolve(None).endswith("piper_no_gripper_description.urdf")
    assert resolve("assets/piper_description/urdf/piper_description.urdf").endswith(
        "piper_description.urdf"
    )
    with pytest.raises(FileNotFoundError, match="gravity_comp_urdf"):
        resolve("/definitely/not/here.urdf")
