"""Teleoperation-only HIL link check: no training, no VLA, no dataset writes.

Run this before `train_online.py` to answer two questions separately from RL:

1. Is the loop fast enough? Per-tick timing of leader CAN read, follower
   observation (camera `async_read` included), rate limiting and `send_action`,
   reported as p50/p95/p99/max plus the overrun rate, and an end-to-end
   follow lag estimated by time-shifting the command against the measurement.
2. Is the plumbing right? It calls the *same* functions the real-robot
   intervention path uses (`leader_action_to_follower`, `rate_limit_joints`,
   `KeyboardEventListener`), so a clean run here means key mapping, rate
   limiting, key semantics and leader/follower alignment are all correct there
   too. With `--rl_token` it also round-trips joint readings through the
   stage-1 normalizer, which is what decides whether human corrections land
   inside the policy's action space at all.

    # dry run: read only, never command the follower
    python -m lerobot.rlt.teleop_check --config_path examples/rlt/piper/teleop_check.yaml \
        --dry_run=true
    python -m lerobot.rlt.teleop_check --config_path examples/rlt/piper/teleop_check.yaml
    python -m lerobot.rlt.teleop_check --config_path examples/rlt/piper/teleop_check.yaml \
        --rl_token=outputs/rl_token

Keys match `train_online.py`: space toggles takeover, s/f label success/failure,
left arrow marks a segment, Esc ends the run.
"""
# No `from __future__ import annotations` here: `lerobot.configs.parser.wrap()`
# reads `inspect.getfullargspec(fn).annotations`, and PEP 563 would hand draccus
# the string "TeleopCheckConfig" instead of the dataclass.
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lerobot.configs import parser
from lerobot.rlt.envs.piper import (
    JOINT_KEYS_6,
    JOINT_ORDER,
    build_piper_cameras,
    follower_action_to_leader,
    jitter_rms,
    leader_action_to_follower,
    rate_limit_joints,
)
from lerobot.rlt.teleop.keys import KeyboardEventListener
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

# Jitter you can feel by hand (rad/s^2): at 30 Hz a +-0.002 rad alternating
# oscillation is ~7, and that already hums audibly. Below this, stop blaming sides.
JITTER_NOTICEABLE = 5.0


@dataclass
class LeaderCheckConfig:
    port: str = "can1"
    id: str = "piper_leader"
    # Same meaning as PiperLeaderTeleopConfig: read absolute joint angles rather
    # than calibrated offsets.
    use_calibrated_offsets: bool = False
    # Off by default so connecting does not drop into interactive calibration.
    require_calibration: bool = False
    # Refuse takeover past this leader/follower gap (0 disables the check).
    max_takeover_delta_rad: float = 0.15

    # Gravity compensation, passed straight through to PiperLeaderConfig. These
    # must be reachable from yaml: tx_ratio is the only coefficient calibrated by
    # feel, and base_rpy_deg decides whether g(q) is computed in the right frame
    # at all.
    #
    # tx_ratio scales RNEA joint torques into SDK MIT torque units and doubles as
    # a derating factor. Piper's 0.2 default compensates about a fifth of the
    # arm's weight, which feels like none at all. Raise it until the arm holds
    # itself up; if it starts drifting *upwards* that is over-compensation —
    # lower it now. Written out in words on purpose: a percent sign in a draccus
    # field comment reaches argparse as a format string and kills --help for
    # every entry point sharing this config tree.
    gravity_comp_tx_ratio: list[float] = field(
        default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    )
    # Mounting attitude. Without it a side/tilted mount never rotates gravity into
    # the base frame and every g(q) is wrong.
    gravity_comp_base_rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Per-joint torque cap (SDK t_ref hardware range +-18 N*m). Measured peak
    # self-weight torque is ~3.3 N*m, so 8.0 never trips in normal use — it only
    # guards against a broken URDF or a NaN.
    gravity_comp_torque_limit: float = 8.0
    # kp must stay 0: pos_ref is always 0, so kp > 0 drags the leader toward zero.
    # Tune jitter with kd (pure damping) only.
    gravity_comp_mit_kp: float = 0.0
    gravity_comp_mit_kd: float = 0.0
    gravity_comp_control_hz: float = 200.0
    # URDF for the gravity model. null uses the built-in
    # piper_no_gripper_description.urdf (nothing on the wrist); with a gripper
    # fitted, switch to assets/piper_description/urdf/piper_description.urdf.
    gravity_comp_urdf: str | None = None
    # Extra end-effector payload (teach handle, camera) in kg and m, relative to
    # the joint6 frame. The built-in URDF has an empty wrist, so a real payload
    # the model does not know about systematically underestimates wrist gravity
    # torque — it presents as "the wrist has no compensation" and no amount of
    # tx_ratio tuning fixes it.
    gravity_comp_payload_mass: float = 0.0
    gravity_comp_payload_com: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


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
    # Matches PiperEnvConfig: per-step joint rate limit after un-normalisation.
    max_joint_step_rad: float = 0.5

    # Cameras are the most expensive part of the control loop and must count
    # against the real-time budget. Empty keeps PIPERConfig's own cameras;
    # `use_cameras=false` isolates the arm link.
    cameras: list[CameraCheckSpec] = field(default_factory=list)
    use_cameras: bool = True

    # Read only, never command. Always the first run on real hardware.
    dry_run: bool = False
    engage_on_start: bool = False
    # Time budget for the leader to actually reach the follower's pose after
    # alignment. JointCtrl is asynchronous; pressing space before it arrives is
    # rejected by the takeover safety gate and looks like "space did nothing".
    align_settle_s: float = 2.0

    # auto (termios when stdin is a real terminal, else pynput's global X11 hook)
    # / termios / pynput / none. In an IDE console stdin is a pipe, so auto falls
    # back to pynput.
    keyboard_backend: str = "auto"

    # Pointing at a stage-1 output directory also checks the normalisation
    # round-trip.
    rl_token: str | None = None
    device: str = "cpu"

    out: str = "outputs/teleop_check"


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def _stats_ms(values: list[float]) -> dict[str, float]:
    """Seconds -> millisecond summary statistics."""
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
    """End-to-end follow lag as the time shift minimising RMSE.

    `cmd[k]` is the joint target sent on tick k, `meas[k]` the angle read on the
    same tick. A real arm always trails by some ticks, so the shift that best
    aligns them is the lag estimate. Returns (lag in ticks, RMS error in rad).

    Only meaningful once the operator has actually moved the leader: with a
    near-constant command every lag scores the same.

    The search is capped at a quarter of the sample count — with a short takeover
    a conservative estimate beats refusing to measure at all.
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


class TeleopChecker:
    """Connect leader + follower, run teleoperation at control_hz and time every step."""

    def __init__(self, cfg: TeleopCheckConfig):
        self.cfg = cfg
        self.dt = 1.0 / cfg.control_hz
        self.keys = KeyboardEventListener(backend=cfg.keyboard_backend)
        self.robot = None
        self.leader = None
        self.normalizer = None

        self.t_leader: list[float] = []
        self.t_obs: list[float] = []
        self.t_map: list[float] = []
        self.t_write: list[float] = []
        self.t_norm: list[float] = []
        self.t_busy: list[float] = []  # per-tick time excluding sleep
        self.t_loop: list[float] = []  # measured period
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
            self.robot.connect(calibrate=False)
        print(f"[check] 跟随臂已连接 ({cfg.can_port})，相机: {list(self.robot.cameras) or '无'}")

        lead = cfg.leader
        self.leader = PiperLeader(
            PiperLeaderConfig(
                port=lead.port,
                id=lead.id,
                require_calibration=lead.require_calibration,
                gravity_comp_tx_ratio=tuple(lead.gravity_comp_tx_ratio),
                gravity_comp_base_rpy_deg=tuple(lead.gravity_comp_base_rpy_deg),
                gravity_comp_torque_limit=lead.gravity_comp_torque_limit,
                gravity_comp_mit_kp=lead.gravity_comp_mit_kp,
                gravity_comp_mit_kd=lead.gravity_comp_mit_kd,
                gravity_comp_control_hz=lead.gravity_comp_control_hz,
                gravity_comp_urdf=lead.gravity_comp_urdf,
                gravity_comp_payload_mass=lead.gravity_comp_payload_mass,
                gravity_comp_payload_com=tuple(lead.gravity_comp_payload_com),
            )
        )
        self.leader.connect()
        print(
            f"[check] 主臂已连接 ({lead.port})  重力补偿: "
            f"tx_ratio={lead.gravity_comp_tx_ratio} "
            f"base_rpy={lead.gravity_comp_base_rpy_deg} "
            f"limit={lead.gravity_comp_torque_limit}N·m "
            f"kp={lead.gravity_comp_mit_kp} kd={lead.gravity_comp_mit_kd}"
        )
        # Whether the gravity model carries a payload has to be printed: an entire
        # class of "the wrist has no compensation" faults comes down to the model
        # silently missing the gripper or teach handle, invisible in the logs.
        print(f"[check]   重力模型: {lead.gravity_comp_urdf or '内置 piper_no_gripper_description.urdf'}")
        if lead.gravity_comp_payload_mass > 0:
            print(
                f"[check]   末端负载: {lead.gravity_comp_payload_mass} kg "
                f"@ {lead.gravity_comp_payload_com} m"
            )
        elif not lead.gravity_comp_urdf:
            print(
                "[check]   末端负载: 无。主臂若装了夹爪 / 示教手柄，腕部重力矩会被系统性"
                "低估（实测约 3.9 倍），表现为「腕部没有重力补偿」，且调 tx_ratio 补不回来。"
                "请设 gravity_comp_payload_mass 或换含夹爪的 gravity_comp_urdf。"
            )
        if max(lead.gravity_comp_tx_ratio) <= 0.25:
            print(
                "[check] 提示：tx_ratio 很低，只补偿约两成自重，主臂拖起来会明显发沉。"
                "若感觉「重力补偿没起作用」，先往上调这个值。"
            )

        if cfg.rl_token:
            self._load_normalizer()

    def _load_normalizer(self) -> None:
        """Load the stage-1 preprocessor and pull out its action normalizer."""
        from lerobot.policies.rlt import load_stage1_processors
        from lerobot.rlt.envs.base import find_action_normalizer

        preprocessor, _post = load_stage1_processors(self.cfg.rl_token, device=self.cfg.device)
        self.normalizer = find_action_normalizer(preprocessor)
        if self.normalizer is None:
            print("[check] 警告：阶段 1 preprocessor 里没找到动作 normalizer，跳过归一化验证")
        else:
            print(f"[check] 已加载阶段 1 归一化 ({self.cfg.rl_token})")

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

    def align(self, settle_s: float = 0.0) -> None:
        """Put the leader back in command mode and onto the follower's pose.

        Mirrors `_exit` on the intervention path. The JointCtrl that
        `send_feedback` issues is *asynchronous* — the leader only starts moving
        after the command lands. Pressing space before it arrives makes the
        takeover gate compare an unconverged leader/follower gap and refuse, which
        looks like "space did not release the leader". `settle_s > 0` polls until
        it has arrived.
        """
        _obs, joints = self.read_follower()
        action = dict(zip(JOINT_ORDER, joints.tolist(), strict=True))
        self.leader.send_feedback(follower_action_to_leader(action))
        self.leader.set_manual_control(False)
        if settle_s <= 0:
            return
        limit = max(self.cfg.leader.max_takeover_delta_rad, 1e-3)
        deadline = time.perf_counter() + settle_s
        while time.perf_counter() < deadline:
            if self.takeover_delta() <= limit * 0.5:
                return
            time.sleep(0.02)
        logger.warning(
            "[check] 主臂在 %.1fs 内没走到跟随臂位姿（当前差 %.3f rad）。"
            "接管可能被安全门拒绝；检查主臂是否使能、或调大 align_settle_s。",
            settle_s,
            self.takeover_delta(),
        )

    def _send_to_follower(self, limited: np.ndarray) -> None:
        """把一拍的关节目标下发给跟随臂（JointCtrl 位置阶跃）。"""
        self.robot.send_action(dict(zip(JOINT_ORDER, limited.tolist(), strict=True)))

    def takeover_delta(self) -> float:
        leader = self.read_leader()
        _obs, joints = self.read_follower()
        follower = dict(zip(JOINT_ORDER, joints.tolist(), strict=True))
        return max(abs(leader[k] - follower[k]) for k in JOINT_KEYS_6)

    def leader_timestamp(self) -> float:
        return float(getattr(self.leader.arm.GetArmJointMsgs(), "time_stamp", 0.0) or 0.0)

    def run(self) -> None:
        cfg = self.cfg
        print(
            "\n[check] 按 空格 握住/交还主臂 | s 成功 | f 失败 | ← 打标记 | Esc 结束\n"
            f"[check] 目标 {cfg.control_hz:.0f} Hz，时长 {cfg.duration_s:g}s，"
            f"{'DRY-RUN（不下发）' if cfg.dry_run else '真实下发'}\n"
        )
        # Align before going raw-terminal: alignment may wait on the leader being
        # enabled, and the key listener must not interrupt that.
        self.align(settle_s=cfg.align_settle_s)

        # engage_on_start has to seed the listener's toggle rather than call
        # set_manual_control(True): the toggle is what the loop reads each tick, so a
        # takeover arranged behind its back reads as `prev_engaged and not engaged`
        # on the next tick and is cancelled immediately — gravity compensation comes
        # up and is switched straight back off.
        if cfg.engage_on_start:
            self.keys.set_intervening(True)
        # Start disengaged so the first takeover goes through _on_engage() and its
        # alignment safety gate like any other.
        prev_engaged = False
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

                t0 = time.perf_counter()
                leader_action = self.read_leader()
                self.t_leader.append(time.perf_counter() - t0)

                ts = self.leader_timestamp()
                self.stale_leader.append(ts <= last_ts)
                last_ts = ts

                # Cameras included, so this costs what it costs during training.
                t0 = time.perf_counter()
                _obs, measured = self.read_follower()
                self.t_obs.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                target = np.array([leader_action[k] for k in JOINT_ORDER], dtype=np.float32)
                limited, saturated = rate_limit_joints(target, measured, cfg.max_joint_step_rad)
                self.t_map.append(time.perf_counter() - t0)
                # Only count rate-limit saturation while engaged. Disengaged, no
                # action is sent, so the follower never closes on the leader and the
                # gap keeps saturating at 100% — an accounting artefact, not the
                # follower failing to keep up.
                if engaged:
                    self.saturated.append(bool(saturated))

                t0 = time.perf_counter()
                if engaged and not cfg.dry_run:
                    self._send_to_follower(limited)
                self.t_write.append(time.perf_counter() - t0)

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
        """Space pressed: check alignment before releasing the leader.

        Returns whether the takeover actually started.
        """
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
            # Wait for the leader to arrive before returning: otherwise the next
            # space press compares the same unconverged gap and is refused again.
            # (The disengage path must *not* wait — it would stall the control loop.)
            self.align(settle_s=self.cfg.align_settle_s)
            return False
        print(f"\n[check] 接管开始（主从差 {delta:.4f} rad）")
        self.leader.set_manual_control(True)
        return True

    def _on_disengage(self) -> None:
        t0 = time.perf_counter()
        self.align()
        print(f"[check] 交还主臂，对齐耗时 {1e3 * (time.perf_counter() - t0):.0f} ms")

    def _check_normalization(self, joints: np.ndarray) -> None:
        """Does a human joint reading survive a normalize/un-normalize round trip?"""
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
            "jitter_rms_rad_s2": {
                "commanded": (
                    jitter_rms(np.stack(self.cmd_hist), self.dt) if len(self.cmd_hist) > 2
                    else float("nan")
                ),
                "measured": (
                    jitter_rms(np.stack(self.meas_hist), self.dt) if len(self.meas_hist) > 2
                    else float("nan")
                ),
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
        """Turn the raw numbers into a per-item "is this link usable" verdict."""
        v: list[dict] = []

        def add(name: str, ok: bool | None, detail: str) -> None:
            # ok=None means "not exercised", which must stay distinct from "failed"
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

        jit = rep["jitter_rms_rad_s2"]
        cmd_j, meas_j = jit["commanded"], jit["measured"]
        if not np.isfinite(meas_j):
            add("跟随臂抖动", None, "接管样本不足，无法评估")
        elif meas_j < JITTER_NOTICEABLE:
            # Absolute threshold first: with a smooth drag the command barely
            # jitters, so the ratio explodes into the thousands and only produces
            # false positives. Small jitter is small; the ratio adds nothing.
            add(
                "跟随臂抖动",
                True,
                f"实测 {meas_j:.2f} rad/s²，低于可察觉量级 {JITTER_NOTICEABLE:.0f}",
            )
        else:
            # Only past the threshold is blame worth assigning: measured >> command
            # means the follower is ringing on its own; comparable values mean the
            # input itself is jittery (the leader side). That decides which end to fix.
            ratio = meas_j / cmd_j if np.isfinite(cmd_j) and cmd_j > 1e-6 else float("inf")
            blames_follower = ratio > 2.0
            if blames_follower:
                ratio_txt = "≫" if not np.isfinite(ratio) else f"{ratio:.1f} 倍"
                detail = (
                    f"实测 {meas_j:.2f} rad/s²，是下发指令 {cmd_j:.2f} 的 {ratio_txt}"
                    " —— 跟随臂自身在振荡。位置模式没有可调阻尼，每拍都是全速冲向位置"
                    "阶跃；先减小 max_joint_step_rad，再检查主臂读数是否本身在跳。"
                )
            else:
                detail = (
                    f"实测 {meas_j:.2f} rad/s²，下发指令 {cmd_j:.2f}（比值 {ratio:.1f}）"
                    " —— 跟随臂忠实跟随，抖动来自输入侧（主臂重力补偿/人手）"
                )
            add("跟随臂抖动", not blames_follower, detail)

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
        """Teardown must never raise.

        `check()` calls this from a `finally`, so anything escaping here masks the
        real `connect()` / `run()` failure and hardware debugging ends up chasing an
        unrelated shutdown error. Guard and log each step so the original survives.
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
