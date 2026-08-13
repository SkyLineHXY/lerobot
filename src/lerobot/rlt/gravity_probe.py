"""主臂重力补偿实时探针：边看数值边标定 tx_ratio。

只碰**主臂**，全程不给跟随臂发任何指令。

要回答的是「按下接管后主臂一点力矩都没有」到底属于哪一种：

  A. MIT 模式没挂上 —— `JointMitCtrl` 被静默忽略，力矩指令根本没生效。
     现象：mode_feed 一直不是 MOVE_M(0x4)，手臂发硬（位置环还在托着）。
  B. MIT 挂上了但补偿太弱 —— 位置环被释放、手臂变软，而 tx_ratio 缩放后的
     力矩托不住自重，手臂直接往下掉。
     现象：mode_feed = MOVE_M(0x4)，下发力矩 << 模型 g(q)。

这两种在操作员手上的感觉都是"没有力矩输出"，但修法完全相反，所以必须看数值。

每一拍打印：MIT 模式是否挂上、模型算出的 g(q) 峰值、经 tx_ratio 缩放并限幅后
实际下发的力矩峰值、以及电机反馈的 effort 峰值。

安全设计：
  * tx_ratio 从 **0 起步**（等于完全不补偿，和不运行本脚本时一样），必须由操作员
    手动一档档往上加，绝不会一上来就把手臂放软。
  * 上限硬性钳在 `--max-ratio`（默认 1.2），防止手滑打到过补偿。
  * 退出时一定把主臂交还位置模式。

用法::

    python -m lerobot.rlt.gravity_probe --port can_left_l

    +/=  tx_ratio 加 0.05      -/_  减 0.05
    0    立刻归零（手臂重新变沉，等同于急停补偿）
    q / Esc  退出并交还位置模式

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
         for i in range(1, 7)],
        dtype=np.float64,
    )


def probe(port: str, step: float, max_ratio: float, hz: float) -> int:
    leader = PiperLeader(
        PiperLeaderConfig(
            port=port,
            id="gravity_probe",
            require_calibration=False,
            # 连上先别放软：由本脚本显式控制何时进重力补偿。
            manual_control=False,
            gravity_comp_tx_ratio=(0.0,) * 6,
        )
    )
    leader.connect()
    print(f"[probe] 主臂已连接 ({port})，当前为位置模式\n")

    print(__doc__.split("用法::")[1].split("⚠")[0].strip())
    print("\n⚠ 先用手扶住主臂，再开始往上加 tx_ratio。手臂自己往上飘 = 过补偿，立刻按 0。")
    input("按 Enter 开始（Ctrl-C 取消）...")

    # tx_ratio=0 进入 MIT：位置环被释放但不给任何补偿力矩，等于「自由下垂」。
    # 所以这一步之前务必扶住手臂。
    leader.set_manual_control(True)
    loop = leader._gravity_comp_loop
    if loop is None:
        print("[probe] 错误：重力补偿回路没有创建。", file=sys.stderr)
        return 1

    ratio = 0.0
    keys = KeyboardEventListener()
    keys.start()
    if not keys.available:
        print("[probe] 警告：按键不可用，无法调 tx_ratio。请在真终端里运行。")

    dt = 1.0 / hz
    print(f"\n{'tx_ratio':>9}{'MIT模式':>10}{'|g(q)|峰值':>12}"
          f"{'下发力矩峰值':>14}{'实测effort峰值':>15}   判定")
    try:
        while True:
            if keys.should_quit():
                break
            for key in (keys._impl.poll() if keys._impl else []):
                if key in ("+", "="):
                    ratio = min(ratio + step, max_ratio)
                elif key in ("-", "_"):
                    ratio = max(ratio - step, 0.0)
                elif key == "0":
                    ratio = 0.0
                    print("\n[probe] tx_ratio 归零，主臂重新变沉。")
                elif key == "q":
                    raise KeyboardInterrupt

            # 回路每一拍都重读这个属性，所以可以在线改。
            loop._tx_ratio = np.full(6, ratio, dtype=np.float64)

            q_rad, v_rad = loop._read_q_v()
            g = loop._compute_gravity_torque(q_rad, v_rad)
            commanded = np.clip(ratio * g, -loop._torque_limit, loop._torque_limit)
            eff = _efforts(leader.arm)
            mode_val, mode_txt = _mode_feed(leader.arm)

            mit_ok = mode_val == MOVE_M
            if not mit_ok:
                verdict = "❌ 情形A：MIT 未挂上，力矩指令被忽略"
            elif ratio == 0.0:
                verdict = "补偿已关（按 + 开始往上加）"
            elif np.abs(commanded).max() < 0.5 * np.abs(g).max():
                verdict = "⚠ 情形B：下发力矩远小于自重，手臂会往下掉"
            else:
                verdict = "✓ 补偿量已接近自重"

            print(
                f"\r{ratio:>9.2f}{mode_txt:>10}{np.abs(g).max():>12.3f}"
                f"{np.abs(commanded).max():>14.3f}{np.abs(eff).max():>15.3f}   {verdict}"
                + " " * 8,
                end="",
                flush=True,
            )
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        keys.stop()
        print("\n\n[probe] 交还位置模式 …")
        try:
            leader.set_manual_control(False)
        finally:
            leader.disconnect()
        print(f"[probe] 结束。本次停在 tx_ratio = {ratio:.2f}")
        if ratio > 0:
            print(f"[probe] 把它写进 examples/rlt/teleop_check.yaml:")
            print(f"          gravity_comp_tx_ratio: {[round(ratio, 2)] * 6}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="主臂重力补偿实时探针")
    parser.add_argument("--port", default="can_left_l", help="主臂 CAN 接口名")
    parser.add_argument("--step", type=float, default=0.05, help="每次 +/- 的步长")
    parser.add_argument("--max-ratio", type=float, default=1.2, help="tx_ratio 硬上限")
    parser.add_argument("--hz", type=float, default=20.0, help="刷新频率")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    return probe(args.port, args.step, args.max_ratio, args.hz)


if __name__ == "__main__":
    sys.exit(main())
