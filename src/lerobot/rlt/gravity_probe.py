"""主臂重力补偿实时探针：逐关节看数值、逐关节标定 tx_ratio。

只碰**主臂**，全程不给跟随臂发任何指令。

为什么必须逐关节
----------------
Piper 各关节的重力负载相差一个数量级以上（模型实算，取多个位姿的最大值）：

    J1      J2      J3      J4      J5      J6
  0.000   3.185   2.718   0.218   0.218   0.000   N·m

* **J1 / J6 恒为 0 是正确的**，不是故障：J1 绕基座竖直轴旋转、J6 绕工具轴旋转，
  重力对这两根轴没有力臂，压根产生不了力矩。这两个关节永远只能靠摩擦自持。
* **J4 / J5 的需求只有 J2/J3 的约 1/15**。用一个统一的 tx_ratio 时，把 J2 调到
  手感合适，腕部拿到的力矩会低到被静摩擦完全吃掉 —— 手上就是"腕部没有补偿"。

所以 `gravity_comp_tx_ratio` 本来就是 6 元组（各关节减速比与电机力矩常数不同），
本探针也按关节分别调。

另外：默认 URDF 是 `piper_no_gripper_description.urdf`，**腕部不含夹爪/手柄**
（link6 质量仅 0.007 kg）。如果主臂末端装了夹爪或遥操作手柄，真实腕部负载比模型
大得多，模型会系统性地少给力矩 —— 这种情况要换一个含负载的 URDF，光调 tx_ratio
补不回来。

判定「按下接管后没有力矩输出」属于哪一种：

  A. MIT 模式没挂上 —— `JointMitCtrl` 被静默忽略，力矩指令根本没生效。
     现象：mode_feed 一直不是 MOVE_M(0x4)，手臂发硬（位置环还在托着）。
  B. MIT 挂上了但补偿太弱 —— 位置环被释放、手臂变软，而 tx_ratio 缩放后的
     力矩托不住自重，手臂往下掉。
     现象：mode_feed = MOVE_M(0x4)，下发力矩 << 模型 g(q)。

安全设计：
  * 所有关节的 tx_ratio 从 **0 起步**（等于完全不补偿），必须手动一档档往上加。
  * 上限硬性钳在 `--max-ratio`（默认 1.2）。
  * 按 `0` 全部立即归零；退出时一定把主臂交还位置模式。

用法::

    python -m lerobot.rlt.gravity_probe --port can_left_l

    1-6   选中要调的关节        a   选中全部关节
    +/=   选中项 tx_ratio +0.05    -/_  -0.05
    0     全部归零（手臂重新变沉，等同于急停补偿）
    q/Esc 退出并交还位置模式

⚠ 开始前先用手扶住主臂。tx_ratio 往上加的过程中手臂会逐渐变软。
⚠ **一旦发现手臂自己往上飘 = 过补偿（正反馈），立刻按 0。**
"""
import argparse
import logging
import sys
import time

import numpy as np

from lerobot.rlt.intervention import KeyboardEventListener
from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

logger = logging.getLogger(__name__)

MOVE_M = 0x04  # MIT / 力矩透传模式的 mode_feed 取值
N_JOINTS = 6
# 低于这个值就认为「重力对该轴没有力臂」，此时下发 0 是正确结果而不是故障。
NO_LOAD_NM = 0.01
# 腕部谐波减速器的静摩擦量级：下发力矩低于它，手上根本感觉不到补偿。
STICTION_NM = 0.05


def _mode_feed(arm) -> tuple[int | None, str]:
    """读回 mode_feed，返回 (整数值, 原始表示)。"""
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
    """单个关节的判定。区分「本来就不该有力矩」和「补偿不足」。"""
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
    """组装要刷新的整块文本（每行一个关节）。"""
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


def probe(port: str, step: float, max_ratio: float, hz: float) -> int:
    leader = PiperLeader(
        PiperLeaderConfig(
            port=port,
            id="gravity_probe",
            require_calibration=False,
            # 连上先别放软：由本脚本显式控制何时进重力补偿。
            manual_control=False,
            gravity_comp_tx_ratio=(0.0,) * N_JOINTS,
        )
    )
    leader.connect()
    print(f"[probe] 主臂已连接 ({port})，当前为位置模式\n")
    print(__doc__.split("用法::")[1].split("⚠")[0].strip())
    print("\n⚠ 先用手扶住主臂再开始。手臂自己往上飘 = 过补偿，立刻按 0。")
    input("按 Enter 开始（Ctrl-C 取消）...")

    # tx_ratio 全 0 进入 MIT：位置环被释放但不给任何补偿力矩，等于「自由下垂」。
    leader.set_manual_control(True)
    loop = leader._gravity_comp_loop
    if loop is None:
        print("[probe] 错误：重力补偿回路没有创建。", file=sys.stderr)
        return 1

    ratios = np.zeros(N_JOINTS, dtype=np.float64)
    selected: int | None = None  # None = 全部关节
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
            # 必须走 poll_extra()：should_quit() 内部的 _read_keys() 会把后端队列
            # 一次抽干并分派到六个内置绑定上，`+` 不匹配任何一个就被丢掉。
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

            # 回路每一拍都重读这个属性，所以可以在线改。
            loop._tx_ratio = ratios.copy()

            q_rad, v_rad = loop._read_q_v()
            g = loop._compute_gravity_torque(q_rad, v_rad)
            commanded = np.clip(ratios * g, -loop._torque_limit, loop._torque_limit)
            eff = _efforts(leader.arm)
            mode_val, mode_txt = _mode_feed(leader.arm)

            lines = render(ratios, g, commanded, eff, mode_txt, mode_val == MOVE_M, selected)
            if n_lines:
                print(f"\033[{n_lines}A", end="")  # 光标回到块首，原地刷新
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
    parser.add_argument("--port", default="can_left_l", help="主臂 CAN 接口名")
    parser.add_argument("--step", type=float, default=0.05, help="每次 +/- 的步长")
    parser.add_argument("--max-ratio", type=float, default=1.2, help="tx_ratio 硬上限")
    parser.add_argument("--hz", type=float, default=20.0, help="刷新频率")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    return probe(args.port, args.step, args.max_ratio, args.hz)


if __name__ == "__main__":
    sys.exit(main())
