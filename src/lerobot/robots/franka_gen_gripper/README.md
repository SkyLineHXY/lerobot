# UMI Franka 机器人参考手册

本目录实现了 **Franka 7-DoF 机械臂 + Gen Controller UMI 夹爪** 的 LeRobot `Robot` 封装，并内置了 UMI 论文 ee6d 坐标变换的完整工具链，是 UMI Franka 系统的核心入口。

---

## 目录

1. [模块概述](#1-模块概述)
2. [目录结构总览](#2-目录结构总览)
3. [UMI ee6d 坐标变换速查](#3-umi-ee6d-坐标变换速查)
4. [FrankaGenGripper 公开 API](#4-frankagengrip-公开-api)
5. [数据集预处理 API](#5-数据集预处理-api)
6. [配置速查](#6-配置速查)
7. [观测 / 动作键 Schema](#7-观测--动作键-schema)
8. [端到端工作流指引](#8-端到端工作流指引)
9. [训练入口](#9-训练入口)
10. [已知限制 / 注意事项](#10-已知限制--注意事项)

---

## 1. 模块概述

```
franka_gen_gripper (Robot)
├── Franka (Robot)           ← 7-DoF 机械臂，zerorpc 通信
│   └── FrankaInterfaceClient  ← RPC → franka_interface_server（NUC 实时机）
└── UmiGripper (Robot)       ← 1-DoF 夹爪，串口 + V4L2 摄像头
    ├── DataBus              ← 串口 921600 baud，编码器 / 触觉线程
    └── CameraCapture        ← 3× V4L2，MJPEG，~30 FPS
```

**核心能力**：

- 统一 `get_observation()` / `send_action()` LeRobot 接口
- 与训练侧数据预处理完全对齐的图像处理（立体相机中心裁切、BGR→RGB）
- 线程安全的 t₀ 参考帧捕获

---

## 2. 目录结构总览

| 文件                           | 是否为入口 | 关键导出                                                                                                                   | 简述                                                      |
| ------------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `franka_gen_gripper.py`        | ✅         | `FrankaGenGripper`                                                                                                         | 主 Robot 类；send/recv 接口与 UMI ee6d 推理方法           |
| `config_franka_gen_gripper.py` | ✅         | `FrankaGenGripperConfig`                                                                                                   | draccus dataclass 配置；手眼标定路径、相机裁切等          |
| `umi_transforms.py`            | 工具库     | `load_flange_to_camera_extrinsic`, `compute_ee_at_t0_deltas`, `ee_relative_action_to_base`, `umi_camera_t0_action_to_base` | SE(3) 数学工具，两种 ee6d 约定的核心推导                  |
| `umi_dataset_transform.py`     | 工具库     | `apply_umi_sample_relative_transform`                                                                                      | 训练时 DataLoader 中的 sample-relative 变换（批量 torch） |
| `__init__.py`                  | —          | 上述所有导出                                                                                                               | 统一导出入口                                              |

---

## 3. UMI ee6d 坐标变换速查

### 坐标系链

```
VIO 相机世界系 W
    │  ᵂT_C(t)   相机 SLAM 轨迹
    ▼
  相机系 C
    │  inv(ᶠT_C) = ᶜT_F   hand-eye 标定
    ▼
  法兰系 F = E = panda_link8
    │  ᴮT_E   Franka FK
    ▼
  机械臂基坐标系 B
```

### 两种 ee6d 约定对照

| 模式                | `delta_k` 语义                     | 推理公式                            | 适用场景                                         |
| ------------------- | ---------------------------------- | ----------------------------------- | ------------------------------------------------ |
| `ee_at_t0`          | `Δ_k = inv(ᵂT_F(t₀)) · ᵂT_F(t_k)`  | `T_BE_t0 @ Δ_k`                     | **推荐**；`compute_ee_at_t0_deltas` 预处理后格式 |
| `camera_t0`         | `ΔC_k = inv(ᵂT_C(t₀)) · ᵂT_C(t_k)` | `T_BE_t0 @ T_FC @ ΔC_k @ inv(T_FC)` | 原始 SLAM 相机轨迹，未经 ᶜT_F 共轭               |
| `abs_pos_world_rot` | 位置相对 t₀ + 姿态绝对             | 混合插值                            | 特殊"置零位置+绝对四元数"预处理变体              |

> **警告**：两种模式数学等价（`Δ_k = ᶠT_C · ΔC_k · inv(ᶠT_C)`），但输入 `delta_k` 的语义完全不同。混用会导致系统性旋转误差，策略完全无法执行。

---

## 4. FrankaGenGripper 公开 API

### 标准 LeRobot 接口

| 方法                      | 说明                                                       |
| ------------------------- | ---------------------------------------------------------- |
| `connect(calibrate=True)` | 顺序建立 Franka + 夹爪连接；夹爪失败时自动回滚 Franka 连接 |
| `disconnect()`            | 先断夹爪（停电机/摄像头）再断 Franka；收集并记录两侧错误   |
| `get_observation()`       | 返回合并的 ee_pose + gripper.pos + camera0/1/2 字典        |
| `send_action(action)`     | 按 `ee_pose.*` / `gripper.*` 前缀分发到两个子设备          |
| `reset()`                 | Franka 回 home + 夹爪设为 5 cm 中间位置                    |

### UMI ee6d 推理接口

| 方法 / 属性                                                      | 签名                        | 说明                                                                     |
| ---------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------ |
| `capture_t0()`                                                   | `() → None`                 | 读取当前 FK 快照存为 `_T_BE_t0`（**线程安全**，每 chunk 起点调用）       |
| `get_t0()`                                                       | `() → np.ndarray \| None`   | 线程安全读取 `T_BE_t0`（4×4 SE(3)）                                      |
| `umi_camera_extrinsic`                                           | 属性 → `np.ndarray (4,4)`   | 懒加载 `ᶠT_C`（panda_link8 → camera），来自 `camera_extrinsic_yaml_path` |
| `get_ee_se3()`                                                   | `() → np.ndarray (4,4)`     | 从 Franka FK 读取当前 `ᴮT_E`                                             |
| `umi_ee6d_to_base(delta_k, T_BE_t0, reference)`                  | `(4,4), (4,4), str → (4,4)` | 按 `reference` 模式将 ee6d 动作转换为基坐标系目标法兰位姿                |
| `send_umi_action(delta_k, T_BE_t0, *, reference, gripper_width)` | —                           | 转换后直接发送；`gripper_width=None` 时跳过夹爪                          |

### UMI 数据集对齐接口

| 方法 / 属性                | 说明                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `umi_observation_features` | 与 `mcap_to_lerobotv3.py world_flange` 格式严格对齐的 lerobot_features 字典                 |
| `get_umi_observation()`    | 返回原始 key-value（`gripper`, `pos_x_W`…`rotvec_z_W`, `camera{i}`），供服务端按 names 组装 |

### 控制器生命周期

| 方法                                              | 说明                                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `start_cartesian_impedance(Kx, Kxd, settle_time)` | 启动 Franka 笛卡尔阻抗控制器（**必须**在 `send_action` / `send_umi_action` 之前调用） |
| `terminate_policy()`                              | 终止当前 Franka 控制器                                                                |

---

## 5. 数据集预处理 API

### `umi_transforms.py`

```python
from lerobot.robots.franka_gen_gripper import (
    load_flange_to_camera_extrinsic,  # 加载 hand-eye 标定 YAML → 4×4 ᶠT_C
    compute_ee_at_t0_deltas,          # 相机轨迹 → EE-at-t₀ Δ_k 列表（数据集预处理）
    compute_sample_relative_poses,    # 绝对位姿序列 → sample-relative（以 base_idx 为原点）
)
```

| 函数                                                       | 签名                               | 用途                                                                      |
| ---------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| `load_flange_to_camera_extrinsic(yaml_path)`               | `str \| Path → (4,4)`              | 从 `franka_easy_handeye` YAML 加载 ᶠT_C；校验 `parent_frame: panda_link8` |
| `compute_ee_at_t0_deltas(W_T_C_list, T_FC)`                | `list[(4,4)], (4,4) → list[(4,4)]` | 将 SLAM 相机轨迹转为 EE-at-t₀ 法兰 Δ_k；`Δ_0 = I`                         |
| `compute_sample_relative_poses(W_T_F_window, base_idx=-1)` | `(T,4,4), int → (T,4,4)`           | 以窗口最后一帧为 t₀ 归一化，用于在线推理                                  |

详细用法见 [`docs/umi_franka_training.md`](../../../../docs/umi_franka_training.md) Step 1。

### `umi_dataset_transform.py`

```python
from lerobot.robots.franka_gen_gripper.umi_dataset_transform import (
    apply_umi_sample_relative_transform,
)
```

| 函数                                                                  | 输入          | 输出             | 说明                                                                                                               |
| --------------------------------------------------------------------- | ------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------ |
| `apply_umi_sample_relative_transform(batch, *, remove_umi_pose=True)` | `batch: dict` | 修改后的 `batch` | 将 `action (B,chunk,7)` 原地转为相对于 `observation.umi_pose` 末帧的 sample-relative；训练时在 DataLoader 之后调用 |

详细用法见 [`docs/umi_franka_training.md`](../../../../docs/umi_franka_training.md) Step 2–3。

---

## 6. 配置速查

`FrankaGenGripperConfig` 字段表（全部可通过 CLI `--robot.<field>=<value>` 覆盖）：

| 字段                         | 默认值                                  | 说明                                                    |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------- |
| `robot_ip`                   | `192.168.172.134`                       | Franka NUC IP                                           |
| `robot_port`                 | `4242`                                  | zerorpc 端口                                            |
| `control_mode`               | `cartesian_impedance`                   | Franka 控制模式                                         |
| `gripper_side`               | `left`                                  | 夹爪侧（`left` / `right`）                              |
| `gripper_serial_port`        | `None`（自动 udev）                     | 串口路径                                                |
| `gripper_camera_width`       | `640`                                   | 摄像头帧宽度                                            |
| `gripper_camera_height`      | `480`                                   | 摄像头帧高度                                            |
| `gripper_camera_count`       | `3`                                     | 摄像头数量                                              |
| `gripper_enable_tactile`     | `False`                                 | 触觉传感器                                              |
| `gripper_encoder_freq`       | `30.0`                                  | 编码器轮询频率（Hz）                                    |
| `stereo_crop`                | `False`                                 | 是否对 camera1/2 做立体中心裁切                         |
| `stereo_crop_ratio`          | `0.75`                                  | 裁切保留比例（0 < ratio ≤ 1），裁切后 resize 到配置尺寸 |
| `camera_extrinsic_yaml_path` | `~/franka_ws/.../camera_transform.yaml` | hand-eye 标定 YAML 路径                                 |
| `Kx`                         | `None`（用 Franka 类默认值）            | 笛卡尔阻抗平动刚度                                      |
| `Kxd`                        | `None`（用 Franka 类默认值）            | 笛卡尔阻抗平动阻尼                                      |

---

## 7. 观测 / 动作键 Schema

### `get_observation()` — 标准模式

| 键                 | 类型              | 说明                                               |
| ------------------ | ----------------- | -------------------------------------------------- |
| `ee_pose.x/y/z`    | `float`           | 末端执行器位置（米）                               |
| `ee_pose.rx/ry/rz` | `float`           | 末端执行器姿态（轴角旋转矢量，弧度）               |
| `gripper.pos`      | `float`           | 夹爪开口宽度（0.0 ~ 0.103 米）                     |
| `camera0`          | `ndarray (H,W,3)` | 中央摄像头 RGB                                     |
| `camera1`          | `ndarray (H,W,3)` | 左立体摄像头 RGB（若 `stereo_crop=True` 则已裁切） |
| `camera2`          | `ndarray (H,W,3)` | 右立体摄像头 RGB（同上）                           |
| `tactile_left`     | `ndarray (500,)`  | 左触觉（`gripper_enable_tactile=True` 时）         |
| `tactile_right`    | `ndarray (500,)`  | 右触觉（同上）                                     |

### `get_umi_observation()` — UMI 数据集对齐模式

> 返回 key-value 字典，服务端 `build_dataset_frame` 按 `umi_observation_features` 的 `names` 列表组装为 parquet 张量。

| 键                                 | 对应数据集特征                 | 说明                    |
| ---------------------------------- | ------------------------------ | ----------------------- |
| `gripper`                          | `observation.state[0]`         | 夹爪宽度（米）          |
| `pos_x_W/pos_y_W/pos_z_W`          | `observation.umi_pose[0:3]`    | EE 位置（VIO 世界系 W） |
| `rotvec_x_W/rotvec_y_W/rotvec_z_W` | `observation.umi_pose[3:6]`    | EE 姿态旋转矢量（W）    |
| `camera{i}`                        | `observation.images.camera{i}` | 摄像头 RGB 帧           |

### `send_action()` / `send_umi_action()` — 动作键

| 键                 | 类型    | 说明                                     |
| ------------------ | ------- | ---------------------------------------- |
| `ee_pose.x/y/z`    | `float` | 目标 EE 位置（米，基坐标系）             |
| `ee_pose.rx/ry/rz` | `float` | 目标 EE 姿态（轴角，弧度，基坐标系）     |
| `gripper.pos`      | `float` | 目标夹爪宽度（自动 clamp 到 [0, 0.103]） |

---

## 8. 端到端工作流指引

> 本节仅给出步骤标题与指引，详细命令和参数见 [`docs/umi_franka_training.md`](../../../../docs/umi_franka_training.md)。

| 步骤                    | 内容                                                                                        | 详见        |
| ----------------------- | ------------------------------------------------------------------------------------------- | ----------- |
| **Step 1** 数据集转换   | `mcap_to_lerobotv3.py --pose-format world_flange` 将 DAS mcap 录制转为 LeRobot v3 格式      | docs Step 1 |
| **Step 2** 预训练验证   | Parquet 健全性检查 + `apply_umi_sample_relative_transform` 变换验证（base delta ≈ 0）       | docs Step 2 |
| **Step 3** 策略训练     | `lerobot-train --policy.type=act ...`，`lerobot_train_umi.py` 自动注入 sample-relative 变换 | docs Step 3 |
| **Step 4** 真机轨迹回放 | `python -m lerobot.scripts.lerobot_replay_umi_franka` 验证数据集动作                        | docs Step 4 |
| **Step 5** 异步推理部署 | `policy_server.py`（GPU 机）+ `franka_umi_async_client.py`（NUC）                           | docs Step 5 |

**手眼标定**（外部步骤）：使用 `franka_easy_handeye` 完成标定后，将 YAML 路径配置到 `camera_extrinsic_yaml_path`。

---

## 9. 训练入口

训练脚本为 [`src/lerobot/scripts/lerobot_train_umi.py`](../../scripts/lerobot_train_umi.py)，在标准 lerobot 训练循环基础上注入 sample-relative 变换：

```bash
python -m lerobot.scripts.lerobot_train_umi \
    --policy.type=act \
    --dataset.repo_id=<hf_user/dataset_name> \
    --umi_transform=true      # 默认 true；world_flange 数据集必须开启
```

`UMITrainConfig.umi_transform=True` 时，每个 batch 在送入策略前会自动调用 `apply_umi_sample_relative_transform`，将绝对法兰位姿转为相对当前 EE 的增量。

---

## 10. 已知限制 / 注意事项

1. **手眼标定 YAML 的 `parent_frame` 必须为 `panda_link8`**。`load_flange_to_camera_extrinsic` 会显式校验此字段；若使用其他法兰链接，需修改校验逻辑。

2. **`capture_t0()` 必须在每个推理 chunk 起点显式调用**。它读取当前 FK 快照作为 t₀ 参考；若遗漏调用，`send_umi_action` 将使用上一个 chunk 的 t₀，导致累积位置误差。

3. **立体相机裁切必须与训练侧对齐**。`stereo_crop` / `stereo_crop_ratio` 应与 `mcap_to_lerobotv3.py --stereo-crop` 选项完全一致，否则推理端图像分布将偏移训练分布。

4. **`se3_to_xyz_rotvec` 输出的轴角格式**与 Franka `robot_update_desired_ee_pose` / `ee_pose.{rx,ry,rz}` action 键约定一致，但与四元数格式不兼容——不可与 `se3_to_xyz_quat` 混用。

---
