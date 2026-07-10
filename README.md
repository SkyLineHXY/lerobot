[//]: # (<p align="center">)

[//]: # (  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">)

[//]: # (</p>)

<div align="center">

[//]: # ([![License]&#40;https://img.shields.io/badge/License-Apache%202.0-blue.svg&#41;]&#40;./LICENSE&#41;)
[//]: # ([![Python versions]&#40;https://img.shields.io/pypi/pyversions/lerobot&#41;]&#40;https://www.python.org/downloads/&#41;)

</div>

# 人手户外采集演示数据到真机策略部署

本仓库在 [LeRobot](https://github.com/huggingface/lerobot) 基础上，打通了一条完整链路：

> **DAS-Gripper 人手采集数据（mcap）→ LeRobot v3 数据集 → 模仿学习/VLA策略训练 → Franka 真机部署**

UMI（Universal Manipulation Interface）通过手持夹爪 + 鱼眼相机 + VIO 采集人手演示，无需真机遥操即可批量获取机器人演示数据。本仓库解决了其中最关键的工程难题——**如何把 VIO 世界系下的相机轨迹，正确转换为机器人的TCP相对增量动作，并保证训练与推理的坐标约定完全一致**。

---

## 🎬 真机推理演示

以下为训练好的 ACT 策略在 Franka 上的 **pick and place 真机推理** 演示：

<table>
  <tr>
    <td align="center">
      <video src="https://github.com/SkyLineHXY/lerobot/raw/main/videos/demo_1.mp4" controls width="100%"></video>
      <br><b>Demo 1</b>
    </td>
    <td align="center">
      <video src="https://github.com/SkyLineHXY/lerobot/raw/main/videos/demo_2.mp4" controls width="100%"></video>
      <br><b>Demo 2</b>
    </td>
    <td align="center">
      <video src="https://github.com/SkyLineHXY/lerobot/raw/main/videos/demo_3.mp4" controls width="100%"></video>
      <br><b>Demo 3</b>
    </td>
  </tr>
</table>

---

## 核心概念

### UMI t₀ 约定（sample-relative）

UMI 的关键设计：每次推理窗口的 **最后一帧观测** 作为当前 t₀，整个动作 chunk（`chunk_size` 步）共用同一个 t₀，每一步都是相对该参考的增量：

```
t₀  = 当次推理采集 obs 时刻的 EE 法兰世界位姿（每个 chunk 刷新一次，chunk 内固定）
Δ_k = inv(W_T_F(t₀)) @ W_T_F(t₀ + k·dt)        k = 0, 1, ..., chunk_size-1
```

**训练与推理必须保持一致**：

- **训练时**：`apply_umi_sample_relative_transform` 取 `umi_pose[..., -1, :]`（最后一帧 obs）作为整个 chunk 的统一 base，把数据集里的绝对位姿转换为 sample-relative 增量。
- **推理时**：每收到一个新 chunk 调用一次 `capture_t0()`（`t0_mode=per_chunk`），chunk 内所有动作共用同一 t₀；策略输出的 7D `[pos3, rotvec3, gripper]` 直接作为 Δ_k 使用。

### `world_flange` 数据集格式

| 字段 | 维度 | 内容 |
|---|---|---|
| `action` | 7D | `[pos_xyz_W, rotvec_xyz_W, gripper]`（VIO 世界系绝对法兰位姿 + 夹爪宽度） |
| `observation.state` | 1D | `[gripper_width]`（仅夹爪宽度，避免绝对位姿冗余进入 state） |
| `observation.umi_pose` | 6D | `[pos_xyz_W, rotvec_xyz_W]`（同帧绝对位姿，供 sample-relative processor 使用） |
| `observation.images.camera0` | H×W×3 | RGB 图像（鱼眼中心镜头） |

### 坐标系关系

```
VIO 世界系 W  ──(W_T_C)──▶  相机光学系 C
                                │
                           inv(T_FC) = T_CF
                                │
                                ▼
Franka 基座系 B ──(T_BE)──▶  法兰系 F（panda_link8）
```

- `T_FC`（ᶠT_C）：`panda_link8 → camera_color_optical_frame`，由 `franka_easy_handeye` 标定
- 法兰绝对位姿：`W_T_F(t) = W_T_C(t) @ inv(T_FC)`
- **推理目标**：`B_T_E(t_k) = B_T_E(t₀) @ inv(W_T_F(t₀)) @ W_T_F(t_k)`

---

## 系统架构

```
DAS mcap 文件
     │
     ▼ mcap_to_lerobotv3.py (--pose-format=world_flange)
     │
LeRobot v3 数据集（Parquet + MP4）
│  action: 7D abs W_T_F + gripper
│  observation.umi_pose: 6D abs W_T_F
│  observation.state: 1D gripper
│  observation.images.camera0: RGB
     │
     ▼ apply_umi_sample_relative_transform（训练 collate）
     │
sample-relative action chunk（7D）→ ACT / Diffusion 训练
     │
     ▼ 策略检查点
     │
FrankaUMIClient + per_chunk t₀ 刷新 → 真机推理
```

---

## 完整流程（5 步）

> 下面仅列关键命令与要点，完整参数、验证脚本与排错见 **[docs/umi_franka_training.md](./docs/umi_franka_training.md)**。

### Step 1 · 数据集转换

将 DAS `.mcap` 转换为 LeRobot v3 数据集（存绝对法兰位姿）：

```bash
python mcap_to_lerobotv3.py \
    --task-dir /path/to/pick_and_place_20260428 \
    --repo-id  yourname/pick_and_place \
    --target-dir /path/to/output_dataset \
    --pose-format world_flange \
    --camera-extrinsic-yaml /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml \
    --fps 30 --task "pick and place"
```

[//]: # (`--camera-extrinsic-yaml` 必须存在且 `parent_frame: panda_link8`。)

### Step 2 · 训练前验证

对数据集做 sanity check（action 覆盖 0.3~0.7m 量级、首帧非全零），并验证 sample-relative 变换后基准帧增量近似为零：

```python
from lerobot.robots.franka_gen_gripper.umi_dataset_transform import apply_umi_sample_relative_transform
batch = apply_umi_sample_relative_transform(batch, remove_umi_pose=False)
print("base delta (应接近零):", batch["action"][0, 0, :6])
```

### Step 3 · 模仿学习/VLA策略训练

在 DataLoader 的 collate 中注入 sample-relative 变换，再启动训练：

```python
def collate_fn(batch_list):
    batch = default_collate(batch_list)
    return apply_umi_sample_relative_transform(batch)   # 变换为 7D 增量，并删除 umi_pose
```

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=yourname/pick_and_place \
    --dataset.root=/path/to/output_dataset \
    --policy.chunk_size=15 --policy.n_action_steps=15 \
    --training.batch_size=16 \
    --output_dir=/path/to/checkpoints/act_umi
```

关键约定：`observation.state` 为 1D（gripper），`action` 为 7D（pos3 + rotvec3 + gripper1）。

### Step 4 · 真机轨迹回放验证

训练前先用数据集原始轨迹直接驱动 Franka，确认坐标变换链路正确：

```bash
python -m lerobot.scripts.lerobot_replay_umi_franka \
    --dataset-root /path/to/output_dataset \
    --episode 0 --robot-ip 192.168.1.104 \
    --camera-extrinsic-yaml /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml \
    --use-gripper --fps 20
```

`world_flange` 格式回放内部自动以 episode 首帧为固定参考（`ee_at_t0`）。回放前确保工作空间无障碍物。

### Step 5 · 异步推理部署

策略服务端（GPU 机器）+ 机器人客户端（NUC）通过 gRPC 通信：

```bash
# GPU 机器：策略服务端
python policy_server.py --host 0.0.0.0 --port 8081 --fps 15

# NUC：Franka 客户端
python franka_umi_async_client.py \
    --pretrained_name_or_path /path/to/checkpoints/act_umi/checkpoints/000500/pretrained_model \
    --server_address <GPU_IP>:8081 \
    --robot.robot_ip 192.168.1.104 \
    --robot.camera_extrinsic_yaml_path /home/zzq/.../camera_transform.yaml \
    --t0_mode per_chunk --reference ee_at_t0 --use_gripper \
    --fps 15 --task "pick and place"
```

`--t0_mode per_chunk` 与训练 collate 保持一致，是坐标约定正确性的关键。

---

## 关键代码位置

| 文件 | 作用 |
|---|---|
| [`src/lerobot/robots/franka_gen_gripper/umi_transforms.py`](./src/lerobot/robots/franka_gen_gripper/umi_transforms.py) | UMI 三种坐标变换模式核心实现 |
| [`src/lerobot/robots/franka_gen_gripper/umi_dataset_transform.py`](./src/lerobot/robots/franka_gen_gripper/umi_dataset_transform.py) | 训练用 sample-relative collate 变换 |
| [`src/lerobot/robots/franka_gen_gripper/franka_gen_gripper.py`](./src/lerobot/robots/franka_gen_gripper/franka_gen_gripper.py) | Franka gen-gripper 机器人抽象层 |
| [`src/lerobot/robots/franka_gen_gripper/config_franka_gen_gripper.py`](./src/lerobot/robots/franka_gen_gripper/config_franka_gen_gripper.py) | 机器人配置（含手眼标定路径） |
| [`src/lerobot/scripts/lerobot_replay_umi_franka.py`](./src/lerobot/scripts/lerobot_replay_umi_franka.py) | 数据集轨迹真机回放 |
| [`src/lerobot/async_inference/policy_server.py`](./src/lerobot/async_inference/policy_server.py) | 异步推理策略服务端 |
| [`src/lerobot/async_inference/franka_umi_async_client.py`](./src/lerobot/async_inference/franka_umi_async_client.py) | Franka 异步推理客户端（t₀ 刷新） |

**三种坐标变换模式**（混用会导致系统性旋转误差）：

| 模式 | delta_k 语义 | 推理公式 |
|---|---|---|
| `ee_at_t0` | Δ_k = inv(ᵂT_F(t₀))·ᵂT_F(t_k) | `T_BE_t0 @ Δ_k` |
| `camera_t0` | ΔC_k = inv(ᵂT_C(t₀))·ᵂT_C(t_k) | `T_BE_t0 @ T_FC @ ΔC_k @ inv(T_FC)` |
| `abs_pos_world_rot` | 位置相对 t₀ + 姿态绝对 | 混合 |

---

## 相关文档

- 📘 **[docs/umi_franka_training.md](./docs/umi_franka_training.md)** — 从 mcap 到真机部署的完整训练指南
- 🧮 **[examples/umi_gripper/docs/UMI ee6d 位姿变换推理.md](./examples/umi_gripper/docs/UMI%20ee6d%20位姿变换推理.md)** — 坐标变换数学推导
---

## 致谢

本仓库基于 [huggingface/lerobot](https://github.com/huggingface/lerobot) 二次开发，遵循 [Apache-2.0](./LICENSE) 许可证。感谢 LeRobot 与 UMI（Chi et al., 2024）开源社区。
