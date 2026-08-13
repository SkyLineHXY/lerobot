"""只做遥操作的 HIL 链路验证脚本：不训练、不加载 VLA、不写数据集。
用途是在跑 `train_online.py` 之前，单独回答两个问题：
1. **实时性够不够？** 重力补偿主臂 -> 跟随臂这条链路每一拍花在哪：主臂 CAN 读数、
   跟随臂观测（含相机 `async_read`）、限速换算、`send_action` 下发。逐项统计
   p50/p95/p99/max，并统计错拍率（单拍耗时超过 1/control_hz 的比例）。
   再用互相关估计端到端跟随延迟（主臂指令 -> 跟随臂实际到位）。

2. **流程对不对？** 复用 `train_online.py` 真机干预路径上的**同一批**函数
   （`leader_action_to_follower` / `rate_limit_joints` / `KeyboardEventListener`），
   所以这里跑通就等于那条路径的键名映射、限速、按键语义、主从对齐都是对的。
   给了 `--rl_token` 还会顺带验证归一化往返：人的关节读数经阶段 1 的
   normalizer 正/反变换后能否原样回来 —— 这是干预动作能否落在策略动作空间里
   的前提。
典型用法:
    # 1) 空跑：只读不下发，先看链路耗时和主臂读数是否正常
    python -m lerobot.rlt.teleop_check --config_path examples/rlt/teleop_check.yaml \
        --dry_run=true

    # 2) 真实跟随：按空格握住主臂开始拖动，再按空格交还
    python -m lerobot.rlt.teleop_check --config_path examples/rlt/teleop_check.yaml

    # 3) 连带验证阶段 1 归一化边界
    python -m lerobot.rlt.teleop_check --config_path examples/rlt/teleop_check.yaml \
        --rl_token=outputs/rl_token

按键与 `train_online.py` 完全一致：空格切换接管、s/f 打成功/失败标签、
← 打一个分段标记、Esc 结束。
"""
# 注意：这里**不能**加 `from __future__ import annotations`。
# `lerobot.configs.parser.wrap()` 是直接读 `inspect.getfullargspec(fn).annotations`
# 取配置类的，PEP 563 会把注解变成字符串，draccus 就会收到 "TeleopCheckConfig"
# 而不是那个 dataclass，报 "must be called with a dataclass type or instance"。
# 仓库里所有 @parser.wrap() 的入口脚本都遵守这一点。
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.utils.robot_utils import precise_sleep

from lerobot.rlt.intervention import JOINT_KEYS_6, KeyboardEventListener
from lerobot.rlt.piper_env import (
    JOINT_ORDER,
    build_piper_cameras,
    follower_action_to_leader,
    leader_action_to_follower,
    rate_limit_joints,
)

logger = logging.getLogger(__name__)


@dataclass
class LeaderCheckConfig:
    port: str = "can1"
    id: str = "piper_leader"
    # 与 PiperLeaderTeleopConfig 同义：默认读绝对关节角，而不是标定偏移
    use_calibrated_offsets: bool = False
    # 校验脚本默认不强制标定，避免连上就掉进交互式标定流程
    require_calibration: bool = False
    # 主从关节差超过这个值就拒绝接管（0 表示不检查）
    max_takeover_delta_rad: float = 0.15


@dataclass
class CameraCheckSpec:
    name: str = "cam"
    index_or_path: str | None = None
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class TeleopCheckConfig:
    can_port: str = "can0"
    leader: LeaderCheckConfig = field(default_factory=LeaderCheckConfig)

    control_hz: float = 30.0
    duration_s: float = 30.0
    # 与 PiperEnvConfig 保持一致：反归一化后的每步关节限速
    max_joint_step_rad: float = 0.05

    # 相机是控制回路里最贵的一环，必须计入实时性预算。留空则沿用 PIPERConfig
    # 自带的相机；`use_cameras=false` 可以关掉相机单独看机械臂链路。
    cameras: list[CameraCheckSpec] = field(default_factory=list)
    use_cameras: bool = True

    # 只读不下发。第一次上机永远先用它。
    dry_run: bool = False
    # 是否一启动就处于接管态；默认 false，要求操作员按空格，这样也顺带验证
    # 「握住主臂 -> 交还」的模式切换流程。
    engage_on_start: bool = False

    # 给了阶段 1 输出目录就额外验证归一化往返
    rl_token: str | None = None
    device: str = "cpu"

    out: str = "outputs/teleop_check"


# --------------------------------------------------------------------- 统计
def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _stats_ms(values: list[float]) -> dict[str, float]:
    """把一列秒转成毫秒统计量。"""
    if not values:
        return {}
    arr = np.asarray(values) * 1e3
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def estimate_tracking_lag(
    cmd: np.ndarray, meas: np.ndarray, max_lag: int
) -> tuple[int, float]:
    """用最小 RMSE 的时移估计端到端跟随延迟。

    `cmd[k]` 是第 k 拍下发的关节目标，`meas[k]` 是同一拍读到的实际关节角。真实
    机械臂总是滞后若干拍，所以把 meas 相对 cmd 前移 lag 拍后误差最小的那个 lag
    就是延迟估计。返回 (lag 拍数, 该 lag 下的 RMS 误差 rad)。

    只有操作员真的拖动过主臂时才有意义：指令几乎不动的话所有 lag 的误差都一样。

    搜索范围会被压到样本数的 1/4 以内：接管时间短的时候宁可给出一个保守的估计，
    也好过因为窗口不够直接放弃测量。
    """
    max_lag = min(max_lag, len(cmd) // 4)
    if len(cmd) < 20 or max_lag < 1:
        return -1, float("nan")
    best_lag, best_rmse = -1, float("inf")
    for lag in range(max_lag + 1):
        a = cmd[: len(cmd) - lag]
        b = meas[lag:]
        rmse = float(np.sqrt(((a - b) ** 2).mean()))
        if rmse < best_rmse:
            best_lag, best_rmse = lag, rmse
    return best_lag, best_rmse


# ------------------------------------------------------------------ 主流程
class TeleopChecker:
    """连接主臂 + 跟随臂，按 control_hz 跑遥操作并全程打点。"""

    def __init__(self, cfg: TeleopCheckConfig):
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.keys = KeyboardEventListener()
        self.robot = None
        self.leader = None
        self.normalizer = None

        # 打点缓冲
        self.t_leader: list[float] = []
        self.t_obs: list[float] = []
        self.t_map: list[float] = []
        self.t_write: list[float] = []
        self.t_norm: list[float] = []
        self.t_busy: list[float] = []  # 一拍里除 sleep 外的总耗时
        self.t_loop: list[float] = []  # 实际周期
        self.saturated: list[bool] = []
        self.stale_leader: list[bool] = []
        self.cmd_hist: list[np.ndarray] = []
        self.meas_hist: list[np.ndarray] = []
        self.norm_err: list[float] = []
        self.takeover_deltas: list[float] = []
        self.n_success = 0
        self.n_failure = 0
        self.n_marks = 0
        self.n_engaged_steps = 0
        self.n_refused = 0

    # ------------------------------------------------------------- 连接
    def connect(self) -> None:
        from lerobot.robots.piper.config_piper import PIPERConfig
        from lerobot.robots.piper.piper import Piper
        from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig

        cfg = self.cfg
        robot_cfg = PIPERConfig(can_port=cfg.can_port)
        if not cfg.use_cameras:
            robot_cfg.cameras = {}
        elif cfg.cameras:
            robot_cfg.cameras = build_piper_cameras(cfg.cameras, cfg.control_hz)
        robot_cfg.max_joint_step_rad = cfg.max_joint_step_rad
        self.robot = Piper(robot_cfg)
        if not self.robot.is_connected:
            self.robot.connect()
        print(f"[check] 跟随臂已连接 ({cfg.can_port})，相机: {list(self.robot.cameras) or '无'}")

        self.leader = PiperLeader(
            PiperLeaderConfig(
                port=cfg.leader.port,
                id=cfg.leader.id,
                require_calibration=cfg.leader.require_calibration,
            )
        )
        self.leader.connect()
        print(f"[check] 主臂已连接 ({cfg.leader.port})，重力补偿线程已启动")

        if cfg.rl_token:
            self._load_normalizer()

    def _load_normalizer(self) -> None:
        """加载阶段 1 的 preprocessor，取出动作 normalizer 用于往返验证。"""
        from lerobot.policies.rlt import load_stage1_processors

        from lerobot.rlt.piper_env import _find_action_normalizer

        preprocessor, _post = load_stage1_processors(self.cfg.rl_token, device=self.cfg.device)
        self.normalizer = _find_action_normalizer(preprocessor)
        if self.normalizer is None:
            print("[check] 警告：阶段 1 preprocessor 里没找到动作 normalizer，跳过归一化验证")
        else:
            print(f"[check] 已加载阶段 1 归一化 ({self.cfg.rl_token})")

    # --------------------------------------------------------- 读写原语
    def read_leader(self) -> dict[str, float]:
        raw = (
            self.leader.get_action()
            if self.cfg.leader.use_calibrated_offsets
            else self.leader.get_raw_action()
        )
        return leader_action_to_follower(raw)

    def read_follower(self) -> tuple[dict, np.ndarray]:
        obs = self.robot.get_observation()
        joints = np.array([obs[k] for k in JOINT_ORDER], dtype=np.float32)
        return obs, joints

    def align(self) -> None:
        """把主臂拉回命令模式并对到跟随臂当前位姿 —— 与干预路径的 `_exit` 同构。"""
        _obs, joints = self.read_follower()
        action = dict(zip(JOINT_ORDER, joints.tolist(), strict=True))
        self.leader.send_feedback(follower_action_to_leader(action))
        self.leader.set_manual_control(False)

    def takeover_delta(self) -> float:
        leader = self.read_leader()
        _obs, joints = self.read_follower()
        follower = dict(zip(JOINT_ORDER, joints.tolist(), strict=True))
        return max(abs(leader[k] - follower[k]) for k in JOINT_KEYS_6)

    def leader_timestamp(self) -> float:
        return float(getattr(self.leader.arm.GetArmJointMsgs(), "time_stamp", 0.0) or 0.0)

    # ------------------------------------------------------------- 主循环
    def run(self) -> None:
        cfg = self.cfg
        print(
            "\n[check] 按 空格 握住/交还主臂 | s 成功 | f 失败 | ← 打标记 | Esc 结束\n"
            f"[check] 目标 {cfg.control_hz:.0f} Hz，时长 {cfg.duration_s:g}s，"
            f"{'DRY-RUN（不下发）' if cfg.dry_run else '真实下发'}\n"
        )
        # 先对齐再进裸终端：对齐可能要等主臂使能，不该被按键监听打断
        self.align()

        engaged = cfg.engage_on_start
        if engaged:
            self.leader.set_manual_control(True)
        prev_engaged = engaged
        last_ts = self.leader_timestamp()

        self.keys.start()
        t_start = time.perf_counter()
        try:
            while time.perf_counter() - t_start < cfg.duration_s:
                if self.keys.should_quit():
                    print("[check] 操作员结束")
                    break
                loop_t0 = time.perf_counter()

                engaged = self.keys.intervening
                if engaged and not prev_engaged:
                    engaged = self._on_engage()
                elif prev_engaged and not engaged:
                    self._on_disengage()
                prev_engaged = engaged

                # 1) 主臂读数
                t0 = time.perf_counter()
                leader_action = self.read_leader()
                self.t_leader.append(time.perf_counter() - t0)

                ts = self.leader_timestamp()
                self.stale_leader.append(ts <= last_ts)
                last_ts = ts

                # 2) 跟随臂观测（含相机），与训练时的观测开销一致
                t0 = time.perf_counter()
                _obs, measured = self.read_follower()
                self.t_obs.append(time.perf_counter() - t0)

                # 3) 换算 + 限速
                t0 = time.perf_counter()
                target = np.array([leader_action[k] for k in JOINT_ORDER], dtype=np.float32)
                limited, saturated = rate_limit_joints(target, measured, cfg.max_joint_step_rad)
                self.t_map.append(time.perf_counter() - t0)
                # 只在接管期间统计限速饱和：没接管时不下发动作，跟随臂不会向主臂
                # 靠拢，主从差会一直挂着，饱和率恒等于 100% —— 那是记账假象，
                # 不是"跟随臂追不上人手"。
                if engaged:
                    self.saturated.append(bool(saturated))

                # 4) 下发
                t0 = time.perf_counter()
                if engaged and not cfg.dry_run:
                    self.robot.send_action(dict(zip(JOINT_ORDER, limited.tolist(), strict=True)))
                self.t_write.append(time.perf_counter() - t0)

                # 5) 归一化往返（可选）
                if self.normalizer is not None:
                    self._check_normalization(target)

                if engaged:
                    self.n_engaged_steps += 1
                    self.cmd_hist.append(limited[:6].copy())
                    self.meas_hist.append(measured[:6].copy())

                self._poll_labels()

                busy = time.perf_counter() - loop_t0
                self.t_busy.append(busy)
                precise_sleep(max(self.dt - busy, 0.0))
                self.t_loop.append(time.perf_counter() - loop_t0)
        finally:
            self.keys.stop()
            if prev_engaged:
                self.align()

    def _on_engage(self) -> bool:
        """空格按下：检查主从对齐后再释放主臂。返回是否真的进入接管。"""
        delta = self.takeover_delta()
        self.takeover_deltas.append(delta)
        limit = self.cfg.leader.max_takeover_delta_rad
        if delta > limit > 0:
            self.n_refused += 1
            print(
                f"\n[check] 拒绝接管：主臂与跟随臂相差 {delta:.3f} rad（上限 {limit:.3f}）。"
                "已重新对齐，把主臂放回机器人位姿后再按空格。"
            )
            self.keys.clear_intervention()
            self.align()
            return False
        print(f"\n[check] 接管开始（主从差 {delta:.4f} rad）")
        self.leader.set_manual_control(True)
        return True

    def _on_disengage(self) -> None:
        t0 = time.perf_counter()
        self.align()
        print(f"[check] 交还主臂，对齐耗时 {1e3 * (time.perf_counter() - t0):.0f} ms")

    def _check_normalization(self, joints: np.ndarray) -> None:
        """人的关节读数经归一化正反变换后能否原样回来。"""
        import torch

        t0 = time.perf_counter()
        x = torch.from_numpy(joints).float()
        norm = self.normalizer._normalize_action(x, inverse=False)
        back = self.normalizer._normalize_action(norm, inverse=True)
        self.t_norm.append(time.perf_counter() - t0)
        self.norm_err.append(float((back - x).abs().max()))

    def _poll_labels(self) -> None:
        success, failure = self.keys.poll_outcome()
        self.n_success += int(success)
        self.n_failure += int(failure)
        if success:
            print("[check] 标签: 成功 (s)")
        if failure:
            print("[check] 标签: 失败 (f)")
        if self.keys.poll_discard():
            self.n_marks += 1
            print("[check] 标记 (←)")

    # ---------------------------------------------------------------- 报告
    def report(self) -> dict:
        cfg = self.cfg
        n = len(self.t_loop)
        if n == 0:
            return {"error": "没有采到任何一拍"}

        budget = self.dt
        misses = sum(1 for b in self.t_busy if b > budget)
        achieved_hz = n / sum(self.t_loop) if sum(self.t_loop) > 0 else 0.0

        lag_steps, lag_rmse = -1, float("nan")
        moved = float("nan")
        if len(self.cmd_hist) > 20 and not cfg.dry_run:
            cmd = np.stack(self.cmd_hist)
            meas = np.stack(self.meas_hist)
            moved = float(cmd.std(axis=0).max())
            lag_steps, lag_rmse = estimate_tracking_lag(
                cmd, meas, max_lag=int(0.5 * cfg.control_hz)
            )

        rep = {
            "config": asdict(cfg),
            "steps": n,
            "engaged_steps": self.n_engaged_steps,
            "target_hz": cfg.control_hz,
            "achieved_hz": achieved_hz,
            "deadline_misses": misses,
            "deadline_miss_rate": misses / n,
            "latency_ms": {
                "leader_read": _stats_ms(self.t_leader),
                "follower_obs": _stats_ms(self.t_obs),
                "map_ratelimit": _stats_ms(self.t_map),
                "send_action": _stats_ms(self.t_write),
                "normalize_roundtrip": _stats_ms(self.t_norm),
                "busy_total": _stats_ms(self.t_busy),
                "loop_period": _stats_ms(self.t_loop),
            },
            "ratelimit_saturation_rate": float(np.mean(self.saturated)) if self.saturated else 0.0,
            "leader_stale_feedback_rate": (
                float(np.mean(self.stale_leader)) if self.stale_leader else 0.0
            ),
            "tracking": {
                "lag_steps": lag_steps,
                "lag_ms": lag_steps * self.dt * 1e3 if lag_steps >= 0 else None,
                "rms_error_rad": lag_rmse,
                "cmd_motion_std_rad": moved,
            },
            "normalization_roundtrip_max_err": max(self.norm_err) if self.norm_err else None,
            "operator": {
                "success_keys": self.n_success,
                "failure_keys": self.n_failure,
                "marks": self.n_marks,
                "takeovers": len(self.takeover_deltas),
                "takeovers_refused": self.n_refused,
                "takeover_delta_rad": self.takeover_deltas,
            },
        }
        rep["verdicts"] = self._verdicts(rep)
        return rep

    def _verdicts(self, rep: dict) -> list[dict]:
        """把原始数字翻译成"这条链路能不能用"的判断。"""
        v: list[dict] = []

        def add(name: str, ok: bool | None, detail: str) -> None:
            # ok=None 表示"这项没被验证到"，和"验证不通过"必须区分开
            status = "WARN" if ok is None else ("PASS" if ok else "FAIL")
            v.append({"check": name, "status": status, "detail": detail})

        budget_ms = self.dt * 1e3
        busy_p95 = rep["latency_ms"]["busy_total"].get("p95", float("nan"))
        add(
            "控制回路实时性",
            busy_p95 < budget_ms,
            f"单拍有效耗时 p95={busy_p95:.1f}ms，预算={budget_ms:.1f}ms "
            f"(错拍率 {rep['deadline_miss_rate']:.1%})",
        )

        stale = rep["leader_stale_feedback_rate"]
        add(
            "主臂 CAN 反馈新鲜度",
            stale < 0.2,
            f"{stale:.1%} 的拍没读到新的关节反馈；偏高说明主臂反馈率跟不上 "
            f"{self.cfg.control_hz:.0f}Hz 控制频率",
        )

        sat = rep["ratelimit_saturation_rate"]
        if not self.saturated:
            add("限速余量", None, "整场没有接管，限速余量只在下发动作时才有意义")
        else:
            add(
                "限速余量",
                sat < 0.3,
                f"接管期间 {sat:.1%} 的拍触发了限速 (max_joint_step_rad="
                f"{self.cfg.max_joint_step_rad}). 持续饱和说明跟随臂追不上人手，"
                "干预时会有明显拖滞",
            )

        lag_ms = rep["tracking"]["lag_ms"]
        moved = rep["tracking"]["cmd_motion_std_rad"]
        if self.cfg.dry_run:
            add("端到端跟随延迟", None, "dry-run 未下发动作，无法测量")
        elif lag_ms is None or not np.isfinite(moved) or moved < 0.02:
            add("端到端跟随延迟", None, "主臂几乎没动，样本不足；请拖动主臂后重测")
        else:
            add(
                "端到端跟随延迟",
                lag_ms <= 150.0,
                f"约 {lag_ms:.0f}ms ({rep['tracking']['lag_steps']} 拍)，"
                f"该时移下跟踪 RMS={rep['tracking']['rms_error_rad']:.4f} rad",
            )

        err = rep["normalization_roundtrip_max_err"]
        if err is None:
            add("归一化往返", None, "未提供 --rl_token，跳过")
        else:
            add(
                "归一化往返",
                err < 1e-3,
                f"最大往返误差 {err:.2e} rad；超过 1e-3 说明人的动作进不了策略动作空间"
                "（多半是 min/max 归一化把关节角截断了）",
            )

        deltas = rep["operator"]["takeover_delta_rad"]
        if not deltas:
            add("主从对齐 / 接管流程", None, "整场没有按空格接管，流程未被验证")
        else:
            worst = max(deltas)
            add(
                "主从对齐 / 接管流程",
                worst <= self.cfg.leader.max_takeover_delta_rad,
                f"{len(deltas)} 次接管，接管瞬间主从最大差 {worst:.4f} rad，"
                f"被拒 {rep['operator']['takeovers_refused']} 次",
            )
        return v

    def close(self) -> None:
        """关机路径不允许抛异常。

        `check()` 在 `finally` 里调用它，所以这里冒出来的任何异常都会顶掉
        `connect()` / `run()` 真正的那个错误 —— 上机排障时看到的就会是一个
        无关的收尾异常。逐段兜住并记录，让原始错误活着传出去。
        """
        if self.leader is not None and getattr(self.leader, "is_connected", False):
            try:
                self.align()
            except Exception:
                logger.exception("[check] 交还主臂失败")
            try:
                self.leader.disconnect()
            except Exception:
                logger.exception("[check] 主臂断开失败")
        if self.robot is not None:
            try:
                if self.robot.is_connected:
                    self.robot.disconnect()
            except Exception:
                logger.exception("[check] 跟随臂断开失败")


def _pad(text: str, width: int) -> str:
    """左对齐补空格，按显示宽度算（中文占两列）。"""
    import unicodedata

    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - shown, 0)


def print_report(rep: dict) -> None:
    if "error" in rep:
        print(f"[check] {rep['error']}")
        return
    print("\n" + "=" * 74)
    print("遥操作链路验证报告")
    print("=" * 74)
    print(
        f"采样 {rep['steps']} 拍（接管 {rep['engaged_steps']} 拍）| "
        f"目标 {rep['target_hz']:.1f} Hz -> 实测 {rep['achieved_hz']:.1f} Hz"
    )
    print("\n--- 各环节耗时 (ms) " + "-" * 50)
    print(_pad("环节", 24) + f"{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}")
    labels = {
        "leader_read": "主臂读数",
        "follower_obs": "跟随臂观测(含相机)",
        "map_ratelimit": "换算+限速",
        "send_action": "下发动作",
        "normalize_roundtrip": "归一化往返",
        "busy_total": "单拍有效耗时",
        "loop_period": "实际周期",
    }
    for key, label in labels.items():
        s = rep["latency_ms"].get(key) or {}
        if not s:
            continue
        print(
            _pad(label, 24)
            + f"{s['p50']:>9.2f}{s['p95']:>9.2f}{s['p99']:>9.2f}{s['max']:>9.2f}"
        )

    print("\n--- 结论 " + "-" * 64)
    for v in rep["verdicts"]:
        icon = {"PASS": "✓", "WARN": "?", "FAIL": "✗"}[v["status"]]
        print(f" {icon} [{v['status']:<4}] {v['check']}\n         {v['detail']}")
    print("=" * 74 + "\n")


def check(cfg: TeleopCheckConfig) -> dict:
    checker = TeleopChecker(cfg)
    try:
        checker.connect()
        checker.run()
    finally:
        rep = checker.report()
        checker.close()

    print_report(rep)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"teleop_check_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    print(f"[check] 报告已保存: {path}")
    return rep


@parser.wrap()
def main(cfg: TeleopCheckConfig):
    check(cfg)


if __name__ == "__main__":
    main()
