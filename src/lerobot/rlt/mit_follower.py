"""用 MIT 阻抗控制驱动跟随臂，替代 30 Hz 的位置阶跃指令。

为什么位置模式必然抖
--------------------
`PiperMotorsBus.write()` 每一拍都发::

    MotionCtrl_2(0x01, 0x01, 100, 0xAD)   # MOVE J，速度比例拉满 100
    JointCtrl(j1..j6)                      # 纯位置阶跃，没有速度前馈

每 33 ms 落一个新位置点，内部轨迹发生器以最大速度冲过去、到点就停，下一拍再来
一次 —— 加减速反复，一秒抖 30 次。而且位置模式**没有任何可调阻尼**，操作员手上
没有旋钮能压住这个振荡。

MIT 模式给的是阻抗控制
----------------------
电机侧执行的是::

    τ = kp · (pos_ref − q) + kd · (vel_ref − q̇) + t_ref

三处改善直指抖动的成因：

* **kd 是真正的阻尼项**。位置模式里根本没有这一项，而阻尼正是消振最直接的手段。
* **vel_ref 是速度前馈**。跟随臂跟着"指令速度"走，而不是每拍去追一个阶跃；
  加减速的反复因此消失。速度取自**主臂自己的 `motor_speed` 反馈**（驱动器已滤过），
  而不是对 30 Hz 采样做差分 —— 后者噪声很大，喂进 vel_ref 反而会制造抖动。
* **t_ref 放跟随臂自身的重力补偿**。有了它，kp 不必靠刚度硬扛自重，可以调低，
  跟随臂随之变柔顺 —— 这对会发生接触的 HIL 采集也更安全。

与低通滤波的区别：滤波是把抖动换成延迟（而端到端延迟恰恰是这条链路要盯的指标），
MIT 阻尼消除的是振荡的来源。

⚠ 安全须知
----------
这是把跟随臂从位置控制换成力矩控制，风险等级完全不同：

* **kp 太低 + 重力补偿不准 = 手臂直接垮下来。** 所以 `gravity_ratio` 默认 1.0，
  且必须先用 `gravity_probe` 把跟随臂（含其末端负载）的重力模型确认好。
* **kp 太高会自激振荡**，比位置模式抖得更厉害。从小往大加。
* 夹爪（joint_7）不走 MIT，仍由 `GripperCtrl` 单独控制。
* 退出时必须切回位置模式，否则手臂会一直处于柔顺状态。
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

N_ARM_JOINTS = 6

# SDK JointMitCtrl 的参数范围（见 piper_sdk 的文档字符串）
KP_RANGE = (0.0, 500.0)
KD_RANGE = (-5.0, 5.0)
TORQUE_RANGE = (-18.0, 18.0)

MOVE_J, MOVE_M = 0x01, 0x04
MIT_ON = 0xAD


class PiperMitFollower:
    """把跟随臂切到 MIT 阻抗模式，按 (位置, 速度, 重力前馈) 驱动。

    只负责下发，不自己起线程：调用方在原本的控制回路里按拍调 :meth:`send`，
    这样时序、打点、限幅都还归调用方管，和位置模式那条路径可以逐拍对照。
    """

    def __init__(
        self,
        arm,
        kp: float = 30.0,
        kd: float = 1.0,
        torque_limit: float = 8.0,
        gravity_model=None,
        gravity_ratio: float = 1.0,
        speed_ratio: int = 100,
        mode_refresh_interval_s: float = 1.0,
    ):
        if not KP_RANGE[0] <= kp <= KP_RANGE[1]:
            raise ValueError(f"kp 必须在 {KP_RANGE} 内，收到 {kp}")
        if not KD_RANGE[0] <= kd <= KD_RANGE[1]:
            raise ValueError(f"kd 必须在 {KD_RANGE} 内，收到 {kd}")
        self._arm = arm
        self.kp = float(kp)
        self.kd = float(kd)
        self.torque_limit = float(torque_limit)
        # 用于算 t_ref 的重力模型（PiperGravityCompensationLoop，只当模型用，不 start）
        self._gravity = gravity_model
        self.gravity_ratio = float(gravity_ratio)
        self._speed_ratio = int(speed_ratio)
        self._mode_refresh_interval_s = max(0.0, float(mode_refresh_interval_s))
        self._last_mode_refresh_t = 0.0
        self._active = False

    # ------------------------------------------------------------------ 模式
    def _send_mit_mode(self) -> None:
        import time

        self._arm.MotionCtrl_2(0x01, MOVE_M, self._speed_ratio, MIT_ON)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_mode_if_needed(self) -> None:
        import time

        if self._mode_refresh_interval_s <= 0:
            return
        if time.monotonic() - self._last_mode_refresh_t >= self._mode_refresh_interval_s:
            self._send_mit_mode()

    def start(self) -> None:
        """切入 MIT 模式。此后跟随臂由力矩驱动，位置环不再托着它。"""
        if self._active:
            return
        self._send_mit_mode()
        self._active = True
        logger.info(
            "跟随臂已切到 MIT 阻抗模式: kp=%.1f kd=%.2f 重力前馈=%.2f",
            self.kp,
            self.kd,
            self.gravity_ratio,
        )

    def stop(self) -> None:
        """切回位置模式。退出路径必须调，否则手臂一直是柔顺的。"""
        if not self._active:
            return
        self._arm.MotionCtrl_2(0x01, MOVE_J, self._speed_ratio, MIT_ON)
        self._active = False
        logger.info("跟随臂已切回位置模式。")

    # ------------------------------------------------------------------ 下发
    def gravity_torque(self) -> np.ndarray:
        """跟随臂当前位姿下的重力矩，作为 t_ref 前馈。没有模型就返回 0。"""
        if self._gravity is None:
            return np.zeros(N_ARM_JOINTS)
        q_rad, v_rad = self._gravity._read_q_v()
        tau = self._gravity._compute_gravity_torque(q_rad, v_rad)
        return np.asarray(tau[:N_ARM_JOINTS], dtype=np.float64)

    def send(self, pos_ref: np.ndarray, vel_ref: np.ndarray | None = None) -> np.ndarray:
        """下发一拍。pos_ref/vel_ref 是 6 维关节量（rad, rad/s）。返回本拍的 t_ref。"""
        if not self._active:
            raise RuntimeError("PiperMitFollower 未 start()，不能下发 MIT 指令。")
        self._refresh_mode_if_needed()

        pos = np.asarray(pos_ref, dtype=np.float64)[:N_ARM_JOINTS]
        vel = (
            np.zeros(N_ARM_JOINTS)
            if vel_ref is None
            else np.asarray(vel_ref, dtype=np.float64)[:N_ARM_JOINTS]
        )
        tau = np.clip(
            self.gravity_ratio * self.gravity_torque(), -self.torque_limit, self.torque_limit
        )
        for i in range(N_ARM_JOINTS):
            self._arm.JointMitCtrl(
                i + 1,
                float(pos[i]),
                float(vel[i]),
                self.kp,
                self.kd,
                float(np.clip(tau[i], *TORQUE_RANGE)),
            )
        return tau


def read_joint_velocity(arm) -> np.ndarray:
    """从 HighSpd 反馈读关节速度（rad/s），用作 MIT 的 vel_ref。

    SDK 里 `motor_speed` 的单位是 0.001 rad/s。用驱动器给的这个值，而不是对
    30 Hz 采样做差分：后者噪声大得多，喂进 vel_ref 会自己制造抖动。
    """
    hs = arm.GetArmHighSpdInfoMsgs()
    return np.array(
        [
            float(getattr(getattr(hs, f"motor_{i}", None), "motor_speed", 0.0) or 0.0) * 1e-3
            for i in range(1, N_ARM_JOINTS + 1)
        ],
        dtype=np.float64,
    )
