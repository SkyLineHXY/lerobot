"""从臂 MIT 阻抗参数标定：重力前馈 + kp/kd，边看数值边调。

只碰**一条**机械臂（`--port` 指定的那条），不涉及主从联动。

调参顺序不能反
--------------
    τ = kp·(pos_ref − q) + kd·(vel_ref − q̇) + t_ref,   t_ref = gravity_ratio · g(q)

1. **先定 gravity_ratio**（kp=kd=0）。此时只剩重力前馈，手臂应该既不下坠也不上飘
   —— 这是唯一能把"模型准不准"单独测出来的状态。
2. **再加 kp**。手臂开始往 pos_ref 收敛，看跟踪误差。
3. **最后加 kd** 压振荡。

反过来调必然白费功夫：kp 会把重力模型的误差硬扛住，你就永远看不出 gravity_ratio
是不是对的，换个位姿又不对了。

重力系数这一步其实和主臂完全同构，也可以直接用::

    python -m lerobot.rlt.gravity_probe --port can_left_f --payload-mass 0.5

`PiperLeader` 并没有"主臂专属"逻辑，它就是"某个 CAN 口上的 Piper 臂"。区别只在
从臂末端挂的是真夹爪，所以 URDF / 负载要按从臂的实际情况给。

两种测试
--------
* **保持模式**（默认）：pos_ref 锁在进入时的位姿，实时显示逐关节跟踪误差与抖动。
  适合调 gravity_ratio 和粗调 kp。
* **阶跃测试**（按 `t`）：给选中关节一个小阶跃，记录响应并算出超调量、稳定时间、
  振荡次数 —— 这是调 kd 的正经依据，比"感觉不抖了"可靠得多。

安全设计
--------
* 三个参数都从 **0 起步**。⚠ kp=kd=gravity=0 时手臂是**完全柔顺**的，会直接垮下来。
  **开始前请把手臂摆到低位、收拢的安全姿态，并用手扶住。**
* **自动兜底**：任一关节偏离 pos_ref 超过 `--max-sag` 就立刻切回位置模式，由位置环
  接住手臂。这是从臂相对主臂最重要的安全差别 —— 位置环随时可以接管。
* `0` / `q` / `Esc` 都会立刻切回位置模式。

用法::

    python -m lerobot.rlt.mit_tune --port can_left_f \\
        --urdf assets/piper_description/urdf/piper_description.urdf

    g / p / d   选择要调的参数（gravity_ratio / kp / kd）
    + / -       调整选中参数
    1-6         选择关节（阶跃测试用）
    t           对选中关节做一次阶跃测试
    h           重新捕获保持位姿（手动把臂摆到别处后用）
    0 / q / Esc 立刻切回位置模式并退出
"""
import argparse
import logging
import sys
import time

import numpy as np

from lerobot.rlt.intervention import KeyboardEventListener
from lerobot.rlt.mit_follower import KD_RANGE, KP_RANGE, N_ARM_JOINTS, PiperMitFollower
from lerobot.rlt.piper_env import jitter_rms

logger = logging.getLogger(__name__)


def analyze_step_response(
    traj: np.ndarray, start: float, target: float, dt: float
) -> dict:
    """从阶跃响应里算出调 PD 需要的三个量。

    * 超调量：越过目标多少（占阶跃幅度的比例）。kp 偏高 / kd 偏低时变大。
    * 稳定时间：误差最后一次超出 5% 阶跃带的时刻。
    * 振荡次数：误差符号翻转的次数，直接反映阻尼够不够。
    """
    step = target - start
    if abs(step) < 1e-9 or traj.size < 3:
        return {"overshoot": float("nan"), "settling_s": float("nan"), "oscillations": -1}

    err = target - traj
    # 超调：响应越过目标的最大幅度
    beyond = np.maximum((traj - target) * np.sign(step), 0.0)
    overshoot = float(beyond.max() / abs(step))

    band = 0.05 * abs(step)
    outside = np.nonzero(np.abs(err) > band)[0]
    settling = float((outside[-1] + 1) * dt) if outside.size else 0.0

    sign = np.sign(err[np.abs(err) > band * 0.2])
    oscillations = int(np.count_nonzero(np.diff(sign) != 0)) if sign.size > 1 else 0
    return {"overshoot": overshoot, "settling_s": settling, "oscillations": oscillations}


def step_verdict(res: dict) -> str:
    """把阶跃响应翻译成"下一步该动哪个旋钮"。"""
    osc, over = res["oscillations"], res["overshoot"]
    if osc < 0 or not np.isfinite(over):
        return "阶跃太小，测不出来"
    if osc >= 4 or over > 0.3:
        return f"振荡 {osc} 次 / 超调 {over:.0%} —— 阻尼不足，加 kd（或降 kp）"
    if res["settling_s"] > 1.0:
        return f"稳定用了 {res['settling_s']:.2f}s —— 太肉，加 kp"
    if over < 0.02 and res["settling_s"] < 0.05:
        return "响应很快且几乎无超调 —— 可能 kp 偏高，注意接触时会硬"
    return f"超调 {over:.0%}、{res['settling_s']:.2f}s 稳定、振荡 {osc} 次 —— 可用"


class MitTuner:
    def __init__(self, port, urdf, payload_mass, payload_com, base_rpy, max_sag, hz, step):
        from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig
        from lerobot.teleoperators.piper_leader.gravity_compensation import (
            PiperGravityCompensationLoop,
        )

        self.hz, self.dt, self.step = hz, 1.0 / hz, step
        self.max_sag = max_sag
        # 借 PiperLeader 做连接与 URDF 解析；它不含任何"主臂专属"逻辑。
        # manual_control=False：连上先留在位置模式，何时进 MIT 由本脚本决定。
        self.arm_wrapper = PiperLeader(
            PiperLeaderConfig(
                port=port, id="mit_tune", require_calibration=False,
                manual_control=False, gravity_comp_urdf=urdf,
            )
        )
        self.arm_wrapper.connect()
        arm = self.arm_wrapper.arm
        print(f"[tune] 已连接 {port}，当前为位置模式")
        print(f"[tune] 重力模型: {urdf or '内置 piper_no_gripper_description.urdf'}"
              + (f"  负载 {payload_mass}kg @ {payload_com}" if payload_mass else "  无额外负载"))

        self.gravity = PiperGravityCompensationLoop(
            arm=arm, urdf_path=self.arm_wrapper._resolve_gravity_urdf(),
            control_hz=200.0, tx_ratio=(1.0,) * 6, torque_limit=8.0,
            mit_kp=0.0, mit_kd=0.0, base_rpy_deg=tuple(base_rpy),
            mode_refresh_interval_s=1.0, move_speed_ratio=100,
            payload_mass=payload_mass, payload_com=tuple(payload_com),
        )
        self.mit = PiperMitFollower(
            arm=arm, kp=0.0, kd=0.0, gravity_model=self.gravity, gravity_ratio=0.0
        )
        self.pos_ref = np.zeros(N_ARM_JOINTS)
        self.selected_param = "gravity_ratio"
        self.selected_joint = 1  # J2，重力负载最大的那个

    # ------------------------------------------------------------------ 读取
    def measured(self) -> np.ndarray:
        q, _v = self.gravity._read_q_v()
        return np.asarray(q[:N_ARM_JOINTS], dtype=np.float64)

    def capture_hold_pose(self) -> None:
        self.pos_ref = self.measured().copy()

    # ------------------------------------------------------------------ 调参
    def adjust(self, delta: float) -> None:
        if self.selected_param == "gravity_ratio":
            self.mit.gravity_ratio = float(np.clip(self.mit.gravity_ratio + delta, 0.0, 1.5))
        elif self.selected_param == "kp":
            self.mit.kp = float(np.clip(self.mit.kp + delta * 200.0, *KP_RANGE))
        else:
            self.mit.kd = float(np.clip(self.mit.kd + delta * 10.0, 0.0, KD_RANGE[1]))

    def params_line(self) -> str:
        def mark(name):
            return f"[{name}]" if self.selected_param == name else f" {name} "

        return (
            f"{mark('gravity_ratio')}={self.mit.gravity_ratio:.2f}   "
            f"{mark('kp')}={self.mit.kp:.1f}   {mark('kd')}={self.mit.kd:.2f}   "
            f"选中关节 J{self.selected_joint + 1}"
        )

    # -------------------------------------------------------------- 阶跃测试
    def run_step_test(self, amplitude: float, settle_s: float = 1.5) -> dict:
        j = self.selected_joint
        start = float(self.measured()[j])
        target = start + amplitude
        base = self.pos_ref.copy()
        traj = []
        self.pos_ref[j] = target
        deadline = time.perf_counter() + settle_s
        while time.perf_counter() < deadline:
            self.mit.send(self.pos_ref)
            traj.append(float(self.measured()[j]))
            time.sleep(self.dt)
        self.pos_ref = base  # 回到原参考位姿
        res = analyze_step_response(np.array(traj), start, target, self.dt)
        res["joint"] = j + 1
        res["amplitude_rad"] = amplitude
        return res

    # ------------------------------------------------------------------ 主循环
    def run(self, keys: KeyboardEventListener) -> None:
        self.mit.start()
        self.capture_hold_pose()
        meas_hist: list[np.ndarray] = []
        n_lines = 0
        last_step: dict | None = None

        while True:
            if keys.should_quit():
                return
            for key in keys.poll_extra():
                if key == "g":
                    self.selected_param = "gravity_ratio"
                elif key == "p":
                    self.selected_param = "kp"
                elif key == "d":
                    self.selected_param = "kd"
                elif key in ("+", "="):
                    self.adjust(self.step)
                elif key in ("-", "_"):
                    self.adjust(-self.step)
                elif key in tuple("123456"):
                    self.selected_joint = int(key) - 1
                elif key == "h":
                    self.capture_hold_pose()
                elif key == "t":
                    last_step = self.run_step_test(amplitude=0.05)
                elif key in ("0", "q"):
                    return

            q = self.measured()
            err = self.pos_ref - q
            # 安全兜底：偏得太远说明托不住（或在发散），交回位置环接住手臂。
            if np.abs(err).max() > self.max_sag:
                print(
                    f"\n[tune] ⚠ J{int(np.argmax(np.abs(err))) + 1} 偏离参考 "
                    f"{np.abs(err).max():.3f} rad，超过 --max-sag {self.max_sag}。"
                    "已切回位置模式。请提高 gravity_ratio 或 kp 后重试。"
                )
                return

            tau = np.clip(
                self.mit.gravity_ratio * self.gravity._compute_gravity_torque(q, np.zeros(6)),
                -self.mit.torque_limit, self.mit.torque_limit,
            )
            self.mit.send(self.pos_ref)
            meas_hist.append(q.copy())
            if len(meas_hist) > 90:
                meas_hist.pop(0)

            lines = [
                self.params_line(),
                "  (g/p/d 选参数 | +/- 调节 | 1-6 选关节 | t 阶跃测试 | h 重捕位姿 | q 退出)",
                f"{'关节':<6}{'pos_ref':>9}{'实测q':>9}{'误差':>9}{'g(q)':>9}{'t_ref':>9}",
            ]
            for i in range(N_ARM_JOINTS):
                g_i = self.gravity._compute_gravity_torque(q, np.zeros(6))[i]
                mark = "*" if i == self.selected_joint else " "
                lines.append(
                    f"{mark}J{i + 1:<4}{self.pos_ref[i]:>9.3f}{q[i]:>9.3f}"
                    f"{err[i]:>9.4f}{g_i:>9.3f}{tau[i]:>9.3f}"
                )
            jit = jitter_rms(np.stack(meas_hist), self.dt) if len(meas_hist) > 2 else float("nan")
            lines.append(
                f"跟踪误差 RMS={np.sqrt((err**2).mean()):.4f} rad   抖动={jit:6.2f} rad/s²"
            )
            if last_step:
                lines.append(
                    f"最近阶跃 J{last_step['joint']}: 超调={last_step['overshoot']:.0%} "
                    f"稳定={last_step['settling_s']:.2f}s 振荡={last_step['oscillations']} 次"
                    f"  -> {step_verdict(last_step)}"
                )
            else:
                lines.append("按 t 对选中关节做阶跃测试（调 kd 的正经依据）")

            if n_lines:
                print(f"\033[{n_lines}A", end="")
            print("\n".join(f"{ln:<96}" for ln in lines), flush=True)
            n_lines = len(lines)
            time.sleep(self.dt)

    def close(self) -> None:
        try:
            self.mit.stop()  # 切回位置模式，手臂由位置环接住
        finally:
            self.arm_wrapper.disconnect()


def main() -> int:
    p = argparse.ArgumentParser(description="从臂 MIT 阻抗参数标定（重力前馈 + kp/kd）")
    p.add_argument("--port", default="can_left_f", help="机械臂 CAN 接口名")
    p.add_argument("--urdf", default=None, help="重力模型 URDF（从臂装夹爪就要指明）")
    p.add_argument("--payload-mass", type=float, default=0.0, help="末端额外负载 kg")
    p.add_argument("--payload-com", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"), help="负载质心相对 joint6，单位 m")
    p.add_argument("--base-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("R", "P", "Y"), help="安装姿态，单位度")
    p.add_argument("--max-sag", type=float, default=0.25,
                   help="偏离参考超过该值(rad)就自动切回位置模式")
    p.add_argument("--hz", type=float, default=50.0, help="控制/刷新频率")
    p.add_argument("--step", type=float, default=0.05, help="每次 +/- 的相对步长")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print(__doc__.split("用法::")[1].strip())
    print(
        "\n⚠ 三个参数都从 0 起步，此时手臂完全柔顺、会直接垮下来。"
        "\n⚠ 请先把手臂摆到低位收拢的安全姿态，并用手扶住，再按 Enter。"
    )
    input("按 Enter 开始（Ctrl-C 取消）...")

    tuner = MitTuner(
        args.port, args.urdf, args.payload_mass, args.payload_com,
        args.base_rpy, args.max_sag, args.hz, args.step,
    )
    keys = KeyboardEventListener()
    keys.start()
    if not keys.available:
        print("[tune] 警告：按键不可用，无法调参。请在真终端里运行。")
    try:
        tuner.run(keys)
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        tuner.close()
        m = tuner.mit
        print(f"\n[tune] 已切回位置模式。最终参数："
              f"gravity_ratio={m.gravity_ratio:.2f} kp={m.kp:.1f} kd={m.kd:.2f}")
        print("[tune] 写进 examples/rlt/teleop_check.yaml 的 follower_mit 段：")
        print(f"          kp: {m.kp:.1f}\n          kd: {m.kd:.2f}"
              f"\n          gravity_ratio: {m.gravity_ratio:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
