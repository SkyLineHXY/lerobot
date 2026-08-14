"""Per-joint live probe for the leader arm's gravity compensation.

Only ever touches the *leader*; never commands the follower.

Gravity load differs by more than an order of magnitude across Piper's joints
(model, max over poses): J1 0.000, J2 3.185, J3 2.718, J4 0.218, J5 0.218,
J6 0.000 N*m. J1/J6 reading 0 is correct, not a fault — gravity has no moment
arm about the base vertical axis or the tool axis, so those two can only hold
themselves by friction. J4/J5 need roughly 1/15 of what J2/J3 do, so a single
tx_ratio tuned until J2 feels right leaves the wrist below stiction and the arm
feels like "the wrist has no compensation". Hence `gravity_comp_tx_ratio` is a
6-tuple and this probe tunes each joint separately.

It also separates the two causes of "no torque after engaging":
  A. MIT never engaged — `JointMitCtrl` silently ignored. mode_feed never
     reaches MOVE_M(0x4) and the arm stays stiff (position loop still holding).
  B. MIT engaged but compensation too weak — arm goes limp and sags, with the
     commanded torque far below the model's g(q).

Safety: every tx_ratio starts at 0 (no compensation at all) and is raised by
hand one step at a time, hard-capped at `--max-ratio`; `0` zeroes everything
instantly, and exit always hands the leader back to position mode.
"""
import argparse
import logging
import sys
import time

import numpy as np

from lerobot.rlt.intervention import KeyboardEventListener
from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

logger = logging.getLogger(__name__)

MOVE_M = 0x04  # mode_feed value for MIT / torque pass-through mode
N_JOINTS = 6
# Below this, gravity has no moment arm about the axis: commanding 0 is correct.
NO_LOAD_NM = 0.01
# Stiction of the wrist harmonic drives: below this the operator feels nothing.
STICTION_NM = 0.05

# Printed to the operator before the loop starts; kept in Chinese like the rest of
# this script's console output.
KEY_HELP = """按键：
    1-6   选中要调的关节        a   选中全部关节
    +/=   选中项 tx_ratio +0.05    -/_  -0.05
    0     全部归零（手臂重新变沉，等同于急停补偿）
    q/Esc 退出并交还位置模式"""


def _mode_feed(arm) -> tuple[int | None, str]:
    """Read back mode_feed as (int value, raw repr)."""
    status = getattr(arm.GetArmStatus(), "arm_status", None)
    raw = getattr(status, "mode_feed", None)
    if raw is None:
        return None, "<无>"
    value = getattr(raw, "value", raw)
    try:
        return int(value), str(raw)
    except (TypeError, ValueError):
        return None, str(raw)


def _efforts(arm) -> np.ndarray:
    hs = arm.GetArmHighSpdInfoMsgs()
    return np.array(
        [float(getattr(getattr(hs, f"motor_{i}", None), "effort", 0.0) or 0.0) / 1000.0
         for i in range(1, N_JOINTS + 1)],
        dtype=np.float64,
    )


def joint_verdict(g_i: float, commanded_i: float, mit_ok: bool) -> str:
    """Per-joint verdict, separating "should be zero" from "under-compensated"."""
    if not mit_ok:
        return "❌ MIT 未挂上，指令被忽略"
    if abs(g_i) < NO_LOAD_NM:
        return "— 重力对该轴无力臂，0 正确"
    if abs(commanded_i) < STICTION_NM:
        return "⚠ 低于静摩擦，手上感觉不到"
    if abs(commanded_i) < 0.5 * abs(g_i):
        return "⚠ 补偿不足，该轴会往下掉"
    return "✓ 接近自重"


def render(
    ratios: np.ndarray,
    g: np.ndarray,
    commanded: np.ndarray,
    eff: np.ndarray,
    mode_txt: str,
    mit_ok: bool,
    selected: int | None,
) -> list[str]:
    """Build the redrawn text block, one line per joint."""
    sel_txt = "全部" if selected is None else f"J{selected + 1}"
    lines = [
        f"MIT模式: {mode_txt:<14}选中: {sel_txt:<6}"
        f"(1-6 选关节 | a 全选 | +/- 调节 | 0 全部归零 | q 退出)",
        f"{'关节':<6}{'tx_ratio':>9}{'g(q)':>9}{'下发':>9}{'effort':>9}   判定",
    ]
    for i in range(N_JOINTS):
        mark = "*" if (selected is None or selected == i) else " "
        lines.append(
            f"{mark}J{i + 1:<4}{ratios[i]:>9.2f}{g[i]:>9.3f}"
            f"{commanded[i]:>9.3f}{eff[i]:>9.3f}   {joint_verdict(g[i], commanded[i], mit_ok)}"
        )
    return lines


def probe(
    port: str,
    step: float,
    max_ratio: float,
    hz: float,
    urdf: str | None = None,
    payload_mass: float = 0.0,
    payload_com: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> int:
    leader = PiperLeader(
        PiperLeaderConfig(
            port=port,
            id="gravity_probe",
            require_calibration=False,
            # Stay stiff on connect; this script decides when to go limp.
            manual_control=False,
            gravity_comp_tx_ratio=(0.0,) * N_JOINTS,
            gravity_comp_urdf=urdf,
            gravity_comp_payload_mass=payload_mass,
            gravity_comp_payload_com=payload_com,
        )
    )
    leader.connect()
    print(f"[probe] 主臂已连接 ({port})，当前为位置模式")
    print(f"[probe] 重力模型: {urdf or '内置 piper_no_gripper_description.urdf'}")
    if payload_mass > 0:
        print(f"[probe] 末端负载: {payload_mass} kg @ {payload_com}")
    else:
        print("[probe] 末端负载: 无。若主臂装了夹爪/示教手柄，腕部重力矩会被系统性"
              "低估，务必用 --payload-mass 或 --urdf 补上。")
    print()
    print(KEY_HELP)
    print("\n⚠ 先用手扶住主臂再开始。手臂自己往上飘 = 过补偿，立刻按 0。")
    input("按 Enter 开始（Ctrl-C 取消）...")

    # Enter MIT with every tx_ratio at 0: position loop released, no compensating
    # torque at all, i.e. free sag.
    leader.set_manual_control(True)
    loop = leader._gravity_comp_loop
    if loop is None:
        print("[probe] 错误：重力补偿回路没有创建。", file=sys.stderr)
        return 1

    ratios = np.zeros(N_JOINTS, dtype=np.float64)
    selected: int | None = None  # None = all joints
    keys = KeyboardEventListener()
    keys.start()
    if not keys.available:
        print("[probe] 警告：按键不可用，无法调 tx_ratio。请在真终端里运行。")

    dt = 1.0 / hz
    n_lines = 0
    try:
        while True:
            if keys.should_quit():
                break
            # Must go through poll_extra(): should_quit() drains the backend queue
            # into the six built-in bindings, and `+` matches none of them.
            for key in keys.poll_extra():
                idx = slice(None) if selected is None else selected
                if key in ("+", "="):
                    ratios[idx] = np.minimum(ratios[idx] + step, max_ratio)
                elif key in ("-", "_"):
                    ratios[idx] = np.maximum(ratios[idx] - step, 0.0)
                elif key == "0":
                    ratios[:] = 0.0
                elif key == "a":
                    selected = None
                elif key in tuple("123456"):
                    selected = int(key) - 1
                elif key == "q":
                    raise KeyboardInterrupt

            # The loop re-reads this attribute every tick, so it is live-tunable.
            loop._tx_ratio = ratios.copy()

            q_rad, v_rad = loop._read_q_v()
            g = loop._compute_gravity_torque(q_rad, v_rad)
            commanded = np.clip(ratios * g, -loop._torque_limit, loop._torque_limit)
            eff = _efforts(leader.arm)
            mode_val, mode_txt = _mode_feed(leader.arm)

            lines = render(ratios, g, commanded, eff, mode_txt, mode_val == MOVE_M, selected)
            if n_lines:
                print(f"\033[{n_lines}A", end="")  # cursor back to block start
            print("\n".join(f"{line:<78}" for line in lines), flush=True)
            n_lines = len(lines)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        print("\n[probe] 交还位置模式 …")
        try:
            leader.set_manual_control(False)
        finally:
            leader.disconnect()
        print(f"[probe] 结束。最终 tx_ratio = {np.array2string(ratios, precision=2)}")
        if ratios.any():
            print("[probe] 把它写进 examples/rlt/teleop_check.yaml 的 leader 段：")
            print(f"          gravity_comp_tx_ratio: {[round(v, 2) for v in ratios]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="主臂重力补偿实时探针（逐关节）")
    parser.add_argument("--port", default="can_left_f", help="主臂 CAN 接口名")
    parser.add_argument("--step", type=float, default=0.05, help="每次 +/- 的步长")
    parser.add_argument("--max-ratio", type=float, default=1.2, help="tx_ratio 硬上限")
    parser.add_argument("--hz", type=float, default=20.0, help="刷新频率")
    parser.add_argument(
        "--urdf", default=None,
        help="重力模型 URDF；末端装夹爪时可用 "
             "assets/piper_description/urdf/piper_description.urdf",
    )
    parser.add_argument(
        "--payload-mass", type=float, default=0.6,
        help="末端额外负载质量 kg（夹爪示教器 / 手柄 / 相机）",
    )
    parser.add_argument(
        "--payload-com", type=float, nargs=3, default=[0.0, 0.0, 0.6],
        metavar=("X", "Y", "Z"), help="负载质心相对 joint6 坐标系的位置，单位 m",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    return probe(args.port, args.step, args.max_ratio, args.hz,
                 args.urdf, args.payload_mass, tuple(args.payload_com))


if __name__ == "__main__":
    sys.exit(main())
