---
type: analysis
tags: [UMI, Pose Transformation, SE3, Coordinate Frames, Imitation Learning, Inference, Robot Control, Calibration, Franka]
sources: [Chi 等 - 2024 - Universal Manipulation Interface, Zhaxizhuoma 等 - 2025 - FastUMI]
created: 2026-04-25
updated: 2026-05-06 (修正 §1/§2：明确 SLAM 世界系 W 以 t₀ 相机位姿为原点，非任意帧)
---

# UMI ee6d 位姿变换：EE（法兰）-at-$t_0$ 推理完整推导

> **核心问题**：UMI 采集的 ee6d 数据如何以 $t_0$ 时刻**法兰坐标系**为参考表达，推理时机械臂如何将策略输出还原为可执行的基坐标系命令？

**关联来源**：
- [[wiki/sources/2026-04-24 UMI (Chi 2024)]]（原始论文，相对轨迹定义）
- [[wiki/sources/2026-04-24 FastUMI (Zhaxizhuoma 2025)]]（FastUMI 变换公式 Eq.1–6）
- [[wiki/entities/UMI]]（系统总览）

---

## 1. 坐标系定义

推导涉及四个坐标系：

| 符号      | 名称                 | 描述                                              | 生命周期    |
| ------- | ------------------ | ----------------------------------------------- | ------- |
| $\{W\}$ | SLAM 世界系           | 以 $t_0$ 时刻相机位姿为原点（${}^W T_C(t_0) = I$），轨迹经归一化处理 | 仅采集阶段存在 |
| $\{C\}$ | 相机系                | 中心鱼眼相机的光学中心，随手持设备运动                             | 采集 + 推理 |
| $\{F\}$ | 法兰系（`panda_link8`） | Franka 第 7 轴输出端，夹爪安装面；策略控制目标                    | 采集 + 推理 |
| $\{B\}$ | 机器人基坐标系            | 机械臂底座固定系（`panda_link0`），FK 输出的参考系               | 仅推理阶段   |


**已知常量（手眼标定）**：
- ${}^C T_F$：相机→法兰固定外参，采集时用于从 SLAM 轨迹计算法兰轨迹

**符号约定**：${}^A T_B$ 表示"坐标系 $B$ 在坐标系 $A$ 中的位姿"，即将 $B$ 中的点变换到 $A$ 中的齐次变换矩阵 $\in SE(3)$。


---

## 2. 采集阶段：SLAM 输出与法兰轨迹

SLAM 跟踪相机运动，轨迹以 $t_0$ 帧为参考（每段演示数据归一化使 ${}^W T_C(t_0) = I$，即 $\{W\} \equiv \{C(t_0)\}$），输出每帧相机位姿：

$${}^W T_C(t) \in SE(3)$$

法兰位姿通过固定外参推算：

$${}^W T_F(t) = {}^W T_C(t) \cdot {}^C T_F$$

---

## 3. ee6d 参考定义：EE（法兰）-at-$t_0$

### 3.1 动作定义

$$\boxed{\Delta_k = {}^{F(t_0)} T_{F(t_k)} = \bigl({}^W T_F(t_0)\bigr)^{-1} \cdot {}^W T_F(t_k) \in SE(3)}$$

含义：**时刻 $t_k$ 的法兰位姿，以 $t_0$ 时刻法兰坐标系为参考**。

	初始条件：$\Delta_{t_0} = I$（单位矩阵）。

### 3.2 从 SLAM 输出直接计算（采集管道）

展开 ${}^W T_F(t) = {}^W T_C(t) \cdot {}^C T_F$：

$$\Delta_k = \bigl({}^W T_C(t_0) \cdot {}^C T_F\bigr)^{-1} \cdot {}^W T_C(t_k) \cdot {}^C T_F$$

$$= ({}^C T_F)^{-1} \cdot \underbrace{\bigl({}^W T_C(t_0)\bigr)^{-1} \cdot {}^W T_C(t_k)}_{\Delta C_k \text{（相对相机运动）}} \cdot {}^C T_F$$

---

## 4. 严谨数学推导

### 4.1 符号表

| 符号                      | 类型             | 含义                          |
| ----------------------- | -------------- | --------------------------- |
| $t_0$                   | 时刻             | 当前观测时刻（策略推理的起点）             |
| $t_k = t_0 + k\Delta t$ | 时刻             | 第 $k$ 步预测时刻                 |
| ${}^W T_C(t)$           | $SE(3)$        | SLAM 输出的相机位姿                |
| ${}^W T_F(t)$           | $SE(3)$        | 采集时法兰位姿                     |
| ${}^B T_F(t)$           | $SE(3)$        | 推理时法兰位姿（机器人 FK 输出）          |
| ${}^C T_F$              | $SE(3)$        | 相机→法兰固定外参（手眼标定值）            |
| $\Delta_k$              | $SE(3)$        | 策略预测的第 $k$ 步 EE-at-$t_0$ 动作 |
| $H$                     | $\mathbb{Z}^+$ | 预测视野（action horizon）        |

### 4.2 推导目标

给定策略预测序列 $\{\Delta_k\}_{k=1}^H$（EE-at-$t_0$ 参考），求机器人基坐标系下的目标法兰位姿序列 $\{{}^B T_F(t_k)\}_{k=1}^H$。

### 4.3 推导步骤

**Step 1**：法兰在世界系中的轨迹

$${}^W T_F(t) = {}^W T_C(t) \cdot {}^C T_F$$

**Step 2**：EE-at-$t_0$ 相对动作展开

$$\Delta_k = \bigl({}^W T_F(t_0)\bigr)^{-1} \cdot {}^W T_F(t_k)$$
	
$$= \bigl({}^W T_C(t_0) \cdot {}^C T_F\bigr)^{-1} \cdot {}^W T_C(t_k) \cdot {}^C T_F$$

$$= ({}^C T_F)^{-1} \cdot \Delta C_k \cdot {}^C T_F \tag{1}$$

（依据 $SE(3)$ 中 $(AB)^{-1} = B^{-1} A^{-1}$）

**Step 3**：相对表示的帧无关性

$\Delta_k$ 描述**物理相对运动**，与绝对坐标系（SLAM 世界系还是机器人基系）无关：

$$\Delta_k^\text{infer} = \Delta_k^\text{collect} \tag{2}$$

这是相对动作表示实现零样本跨平台迁移的根本原因：**绝对世界系消失了**。

**Step 4**：还原推理时的目标法兰位姿

推理时法兰的初始位姿为 ${}^B T_F(t_0)$（由 FK 读取），目标位姿为：

$$\boxed{{}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k} \tag{3}$$

### 4.4 分量展开（平移 + 旋转）

设 ${}^B T_F(t_0) = (\mathbf{p}_{BF_0},\ \mathbf{R}_{BF_0})$，$\Delta_k = (\mathbf{p}_{\Delta k},\ \mathbf{R}_{\Delta k})$，则：

$$\mathbf{p}_F^{(k)} = \mathbf{p}_{BF_0} + \mathbf{R}_{BF_0} \cdot \mathbf{p}_{\Delta k}$$

$$\mathbf{R}_F^{(k)} = \mathbf{R}_{BF_0} \cdot \mathbf{R}_{\Delta k}$$

---

## 5. 推理时完整变换流程

```
观测时刻 t_0
│
├─ [传感器] 获取腕部相机图像 I(t_0)
│
├─ [机器人 FK] 读取当前法兰位姿
│   ᴮT_F(t₀) ← robot.get_ee_pose()     ← 唯一需要的机器人状态，无需外参
│
├─ [策略推理] policy(I(t_0), ...) → {Δ₁, Δ₂, ..., Δ_H}
│   其中 Δ_k = ᶠ⁽ᵗ⁰⁾T_F(t_k)，初始值 Δ₀ = I
│
├─ [坐标变换] 对每个预测步 k = 1..H：
│   ᴮT_F(t_k) = ᴮT_F(t₀) · Δ_k        ← 不需要外参 ᶜT_F
│
├─ [延迟匹配] 根据 UMI PD1，选取对应执行时刻的 Δ_{k*}
│   k* = ⌈Δt_total / Δt⌉
│
└─ [逆运动学 IK] θ(t_k) = IK(ᴮT_F(t_k))
    └─ 发送关节指令给机器人控制器
```

---

## 6. 关键约束与潜在陷阱

### 6.1 ${}^C T_F$ 精度影响数据质量

EE-at-$t_0$ 方案在**数据预处理**时使用 ${}^C T_F$，而非推理时。若标定不准，误差被固化进训练数据，不会在推理时动态放大。需在数据质量检验阶段而非推理管道中处理标定误差。

### 6.2 重力方向一致性

SLAM 世界系 $\{W\}$ 通常以初始化时的重力方向定义 Z 轴（IMU 辅助）。机器人基坐标系 $\{B\}$ 同样以重力为 Z 轴。

相对表示消除了**平移原点**的差异，但若两个系的**姿态基准**（Roll/Pitch 零点）不一致，$\mathbf{R}_{\Delta k}$ 的旋转方向将出现系统性偏差。

✅ 对策：确保 SLAM 初始化时设备水平放置，与机器人 URDF 的重力定义对齐。

### 6.3 $t_0$ 选取：观测时刻 ≠ 执行时刻

UMI 的延迟匹配（PD1）指出：
- **$t_0$** = 图像被采集的时刻（观测时刻）
- **$t_\text{act}$** = 动作实际被执行的时刻（含推理延迟 + 执行延迟）

${}^B T_F(t_0)$ 必须以**图像采集时刻**的 FK 快照为准，而非发送命令时的实时状态。否则参考帧时间错位，导致轨迹偏移。

$$\Delta t_\text{total} = \underbrace{(t_\text{input} - t_\text{obs})}_\text{观测延迟} + \underbrace{(t_\text{output} - t_\text{input})}_\text{推理延迟} + \underbrace{(t_\text{act} - t_\text{output})}_\text{执行延迟}$$

发送给机器人的是 $\Delta_{k^*}$，其中 $k^* = \lceil \Delta t_\text{total} / \Delta t \rceil$。

### 6.4 SLAM 漂移的影响

相对表示 $\Delta_k$ 仅在 $t_0$ 到 $t_k$ 的**短时窗**内有效，长时 SLAM 漂移会传播进 $\Delta_k$。

✅ 对策：限制 action horizon $H$ 在合理范围（UMI 通常 $H \leq 16$ 帧），并在 FastUMI 中使用回环闭合抑制漂移。

---

## 7. 与 FastUMI 公式的对齐

FastUMI（Eq.5–6）定义的是**逐帧增量**（速度形式），与 UMI 原版的 **$t_0$ 为固定原点**的绝对相对位姿不同：

|      | UMI（Chi 2024, EE-at-$t_0$）                      | FastUMI Eq.5–6                                                                                               |
| ---- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 动作形式 | $\Delta_k = {}^{F(t_0)} T_{F(t_k)}$（$t_0$ 固定原点） | $\mathbf{p}_\text{rel}^{(i)} = \mathbf{p}_F^{(i+1)} - \mathbf{p}_F^{(i)}$（逐帧差分）                              |
| 推理公式 | ${}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k$  | ${}^B \mathbf{p}_F^{(i+1)} = {}^B \mathbf{p}_F^{(i)} + \mathbf{R}_F^{(i)} \cdot \mathbf{p}_\text{rel}^{(i)}$ |
| 漂移积累 | 不积累（每次以新 $t_0$ 为基准）                             | 积累（逐帧累加误差）                                                                                                   |
| 适配策略 | Diffusion Policy（多步预测）                          | ACT / Smooth-ACT（逐帧自回归）                                                                                      |

---

## 8. 小结

| 步骤 | 操作 | 公式 |
| --- | --- | --- |
| ① 读 FK | 获取当前法兰位姿 | ${}^B T_F(t_0)$ |
| ② 策略推理 | 输入图像，输出 ee6d | $\{\Delta_k\}_{k=1}^H$ |
| ③ 坐标变换 | EE-at-$t_0$ 参考→基坐标系 | ${}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k$ |
| ④ 延迟补偿 | 选取对应执行时刻的动作 | $k^* = \lceil \Delta t_\text{total} / \Delta t \rceil$ |
| ⑤ IK 求解 | 发送关节命令 | $\boldsymbol{\theta}(t_k) = \text{IK}({}^B T_F(t_k))$ |

**核心等式**：

$$\boxed{{}^B T_F(t_k) = {}^B T_F(t_0) \cdot \Delta_k}$$

其中 $\Delta_k = {}^{F(t_0)} T_{F(t_k)} \in SE(3)$ 是策略预测的 ee6d，${}^B T_F(t_0)$ 由 Polymetis FK 实时读取。

---

## 9. Franka 具体实现

> **场景**：EE 定义为 Franka 法兰（`panda_link8`），Polymetis 驱动控制。

### 9.1 Franka 内部坐标系链

$${}^O T_{EE} = \underbrace{{}^O T_F}_{\substack{\text{纯运动学 FK} \\ \text{（7 个关节角决定）}}} \cdot \underbrace{{}^F T_{NE}}_{\substack{\text{法兰→名义 EE} \\ \text{（出厂固定，默认 }I\text{）}}} \cdot \underbrace{{}^{NE} T_{EE}}_{\substack{\text{用户自定义偏移} \\ \text{（可通过 Desk 设置）}}}$$

| 帧符号 | ROS TF 名称 | 含义 |
| --- | --- | --- |
| $\{O\}$ = $\{B\}$ | `panda_link0` | 机器人基坐标系 |
| $\{F\}$ | `panda_link8` | **法兰**（第 7 轴输出端，夹爪安装面） |
| $\{NE\}$ | — | 名义末端执行器（出厂写死） |
| $\{EE\}$ | `panda_EE` | 用户定义的实际 EE |

**关键结论**：当 `F_T_NE = I` 且 `NE_T_EE = I`（Franka 出厂默认），`O_T_EE` 直接等于 ${}^B T_F$。

### 9.2 法兰帧的轴方向

`panda_link8` 的右手坐标系：

- **+Z**：垂直法兰面向外（安装工具的接近方向）
- **+X**：水平，指向 `panda_link7` 连接侧
- **+Y**：右手系补全（$= \hat{Z} \times \hat{X}$）

> GoPro 光轴通常与法兰 +Z 轴存在固定旋转偏移，需在手眼标定 ${}^C T_F$ 中体现。

### 9.3 读取 ${}^B T_F$（polymetis）

```python
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R
import numpy as np

robot = RobotInterface(ip_address="172.16.0.1")

def get_T_BF() -> np.ndarray:
    """读取当前法兰位姿 ᴮT_F ∈ SE(3)。"""
    pos, quat = robot.get_ee_pose()  # quat: [x, y, z, w]（scipy 约定）
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat.numpy()).as_matrix()
    T[:3, 3]  = pos.numpy()
    return T
```

**验证**：`T[:3, :3]` 应为正交旋转矩阵（行列式 = 1），`T[:3, 3]` 应为合理的平移量（单位米）。

### 9.4 数据预处理：从 SLAM 输出生成 EE-at-$t_0$ 动作

```python
import numpy as np

T_CF = np.load("camera_to_flange_extrinsic.npy")  # (4,4), ᶜT_F

# 从 SLAM 相机轨迹直接输出 EE-at-t0 格式
W_T_F_t0 = W_T_C_list[0] @ T_CF            # t0 时刻法兰位姿
W_T_F_t0_inv = np.linalg.inv(W_T_F_t0)

deltas = []
for W_T_C_tk in W_T_C_list:
    W_T_F_tk = W_T_C_tk @ T_CF
    Delta_k = W_T_F_t0_inv @ W_T_F_tk      # EE-at-t0 相对动作，初始值为 I
    deltas.append(Delta_k)
```

### 9.5 ⚠️ 标定对齐：${}^C T_F$ 的"手"必须取 `panda_link8`

手眼标定时，"手"参考点必须与 ee6d 的 $\{F\}$ 定义完全一致：

```
采集时手持 UMI                    推理时机械臂（F = panda_link8）
┌──────────────────────────┐     ┌──────────────────────────┐
│ 相机固定在 3D 打印夹爪     │     │ 相机固定在法兰            │
│ 标定：相机 → 夹爪 TCP     │  ≠  │ 标定：相机 → 法兰         │
│ 结果：ᶜT_TCP              │     │ 结果：ᶜT_F               │
└──────────────────────────┘     └──────────────────────────┘
          若两者不一致，Δ_k 的坐标意义出现系统性偏差
```

### 9.6 完整推理骨架

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from polymetis import RobotInterface

robot = RobotInterface(ip_address="172.16.0.1")

def get_T_BF() -> np.ndarray:
    pos, quat = robot.get_ee_pose()
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat.numpy()).as_matrix()
    T[:3, 3]  = pos.numpy()
    return T

def inference_step(policy, camera) -> None:
    # ① 读取 t₀ 时刻的 FK 快照（与图像时间戳对齐）
    T_BF_t0 = get_T_BF()

    # ② 策略推理 → EE-at-t₀ 格式的动作序列（初始值为 I）
    img = camera.capture()
    deltas = policy.predict(img)            # List[(4,4)]，Δ_k

    # ③ 坐标变换（无需外参）
    T_BF_targets = [T_BF_t0 @ delta_k for delta_k in deltas]

    # ④ 延迟补偿：选取对应执行时刻的动作
    k_star = compute_delay_offset()
    T_BF_exec = T_BF_targets[k_star]

    # ⑤ IK 求解 → 发送关节命令
    q_target = ik_solver.solve(T_BF_exec)
    robot.send_joint_command(q_target)
```
