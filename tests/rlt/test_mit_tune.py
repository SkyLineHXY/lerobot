"""从臂 PD 标定里那段"把响应翻译成该调哪个旋钮"的逻辑（不碰硬件）。

阶跃响应的判读是调 kd 的唯一客观依据，判错了会让人往反方向调。
"""

import numpy as np
import pytest
from lerobot.rlt.mit_tune import analyze_step_response, step_verdict

DT = 1 / 50


def _first_order(tau_s, n=150, step=0.05):
    """一阶收敛：无超调、无振荡 —— 阻尼充足的典型形态。"""
    t = np.arange(n) * DT
    return step * (1 - np.exp(-t / tau_s))


def _underdamped(n=150, step=0.05, zeta=0.15, wn=25.0):
    """欠阻尼二阶：明显超调 + 振荡 —— kd 不够的典型形态。"""
    t = np.arange(n) * DT
    wd = wn * np.sqrt(1 - zeta**2)
    return step * (1 - np.exp(-zeta * wn * t) * np.cos(wd * t))


def test_no_overshoot_on_a_well_damped_response():
    res = analyze_step_response(_first_order(0.08), 0.0, 0.05, DT)
    assert res["overshoot"] == pytest.approx(0.0, abs=1e-6)
    assert res["oscillations"] == 0
    assert 0.0 < res["settling_s"] < 1.0


def test_underdamped_response_is_flagged_for_more_damping():
    res = analyze_step_response(_underdamped(), 0.0, 0.05, DT)
    assert res["overshoot"] > 0.3, "欠阻尼必须测出明显超调"
    assert res["oscillations"] >= 2
    assert "加 kd" in step_verdict(res)


def test_sluggish_response_is_flagged_for_more_stiffness():
    """收敛太慢是刚度不足，要往上加 kp —— 不能误判成阻尼问题。"""
    res = analyze_step_response(_first_order(1.5, n=400), 0.0, 0.05, DT)
    assert res["oscillations"] == 0
    assert "加 kp" in step_verdict(res)


def test_healthy_response_is_accepted():
    res = analyze_step_response(_first_order(0.05), 0.0, 0.05, DT)
    assert "可用" in step_verdict(res) or "kp 偏高" in step_verdict(res)


def test_degenerate_inputs_do_not_crash():
    zero = analyze_step_response(np.zeros(50), 0.0, 0.0, DT)
    assert np.isnan(zero["overshoot"])
    assert "测不出来" in step_verdict(zero)
    assert np.isnan(analyze_step_response(np.zeros(2), 0.0, 0.05, DT)["overshoot"])


def test_settling_time_grows_with_a_slower_response():
    fast = analyze_step_response(_first_order(0.05, n=400), 0.0, 0.05, DT)
    slow = analyze_step_response(_first_order(0.40, n=400), 0.0, 0.05, DT)
    assert slow["settling_s"] > fast["settling_s"]


def test_negative_step_is_handled():
    """向下的阶跃：超调判据不能因为符号而反过来。"""
    traj = -_first_order(0.08)
    res = analyze_step_response(traj, 0.0, -0.05, DT)
    assert res["overshoot"] == pytest.approx(0.0, abs=1e-6)
    assert res["oscillations"] == 0
