"""重力补偿探针的判定逻辑单测（不碰硬件）。

这里最要紧的是把两件事分清楚：某个关节"下发 0 力矩"到底是**正确结果**
（重力对该轴没有力臂，例如绕基座竖直轴的 J1 和绕工具轴的 J6），还是**补偿不足**。
判反了会让人去调一个永远不会有输出的关节。
"""

import numpy as np
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
