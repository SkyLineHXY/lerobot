# UMI + Franka 数据集训练指南

本文档描述从 DAS-Gripper mcap 数据到 ACT 策略训练、真机部署的完整流程。

---

## 目录

1. [核心概念](#核心概念)
2. [系统架构](#系统架构)
3. [Step 1：数据集转换](#step-1数据集转换)
4. [Step 2：训练前验证](#step-2训练前验证)
5. [Step 3：ACT 策略训练](#step-3act-策略训练)
6. [Step 4：真机轨迹回放验证](#step-4真机轨迹回放验证)
7. [Step 5：策略推理部署](#step-5策略推理部署)
8. [手眼标定文件格式](#手眼标定文件格式)
9. [常见问题排查](#常见问题排查)

---

## 核心概念

### UMI t₀ 约定（sample-relative）

UMI 论文的关键设计：每次推理窗口的 **最后一帧观测** 作为当前 t₀，整个动作 chunk（chunk_size 步）共用同一个 t₀，每一步是相对此参考的增量。

```
t₀ = 当次推理时（采集 obs 的时刻）EE 法兰世界位姿（每个 chunk 刷新一次，chunk 内固定）
Δ_k = inv(W_T_F(t₀)) @ W_T_F(t₀ + k·dt)        for k = 0, 1, ..., chunk_size-1
```

**训练/推理必须保持一致**：
- 训练时：`apply_umi_sample_relative_transform` 取 `umi_pose[..., -1, :]`（最后一帧 obs）
  作为整个 chunk 的统一 base，把数据集绝对位姿转为 sample-relative。
- 推理时：每收到一个新 chunk 调用一次 `capture_t0()`（`t0_mode=per_chunk`），chunk 内
  所有动作都用同一个 t₀；策略输出 7D `[pos3, rotvec3, gripper]` 直接作为 Δ_k 使用。

### world_flange 数据集格式

| 字段 | 维度 | 内容 |
|---|---|---|
| `action` | 7D | `[pos_x_W, pos_y_W, pos_z_W, rotvec_x_W, rotvec_y_W, rotvec_z_W, gripper]`（VIO 世界系绝对法兰位姿 + 夹爪宽度） |
| `observation.state` | 1D | `[gripper_width]`（仅夹爪宽度，避免绝对位姿作为 state 引入冗余） |
| `observation.umi_pose` | 6D | `[pos_x_W, pos_y_W, pos_z_W, rotvec_x_W, rotvec_y_W, rotvec_z_W]`（同一帧绝对位姿，供 sample-relative processor 使用） |
| `observation.images.camera0` | H×W×3 | RGB 图像（鱼眼中心镜头） |

[//]: # (> **旧格式（ee_at_t0_flange）**：action/state 均为 8D `[pos_xyz, quat_xyzw, gripper]`，以 episode 首帧为固定 t₀。新项目推荐使用 world_flange 格式。)

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
- 推理目标：`B_T_E(t_k) = B_T_E(t₀) @ inv(W_T_F(t₀)) @ W_T_F(t_k)`

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
sample-relative action chunk（7D）→ ACT/Diffusion 训练
     │
     ▼ 策略检查点
     │
FrankaUMIClient + per_chunk t₀ 刷新 → 真机推理
```

---

## Step 1：数据集转换

将 DAS 格式 `.mcap` 文件转换为 LeRobot v3 数据集：

```bash
cd /home/zzq/gendas_dataset/das-datakit

python mcap_to_lerobotv3.py \
    --task-dir /path/to/pick_and_place_20260428 \
    --repo-id   yourname/pick_and_place \
    --target-dir /path/to/output_dataset \
    --pose-format world_flange \
    --camera-extrinsic-yaml /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml \
    --fps 30 \
    --task "pick and place"
```

**重要参数说明**：

| 参数 | 说明 |
|---|---|
| `--pose-format world_flange` | **推荐**，存绝对法兰位姿；旧值 `ee_at_t0_flange` 已弃用 |
| `--camera-extrinsic-yaml` | hand-eye 标定 YAML，必须存在且 `parent_frame: panda_link8` |
| `--fps` | 应与录制帧率一致（通常 30Hz） |
| `--stereo-camera` | 可选，同时写入 camera1/camera2 |
| `--force` | 覆盖已存在的目标目录（无交互提示） |

转换完成后控制台会打印 t₀ 时刻法兰位姿，用于目视核实：

```
[world_flange] t₀ 法兰位姿: pos=[0.4823, -0.0012, 0.3201], rotvec=[...]
```

---

## Step 2：训练前验证

### 2.1 数据集 sanity check

```python
import pyarrow.parquet as pq
import numpy as np

df = pq.read_table("/path/to/output_dataset/data/chunk-000/episode_000000.parquet").to_pandas()
ep = df[df["episode_index"] == 0].sort_values("frame_index")

actions = np.stack(ep["action"].to_numpy())
umi_poses = np.stack(ep["observation.umi_pose"].to_numpy())

print("action shape:", actions.shape)       # (T, 7)
print("umi_pose shape:", umi_poses.shape)   # (T, 6)
print("首帧绝对位置 (m):", actions[0, :3])
print("尾帧绝对位置 (m):", actions[-1, :3])
```

action 应覆盖真实操作空间范围（0.3~0.7m 量级），首帧不应全为 0。

### 2.2 sample-relative 变换验证

```python
from lerobot.robots.franka_gen_gripper.umi_dataset_transform import apply_umi_sample_relative_transform
import torch

batch = {
    "action": torch.tensor(actions[:50]).unsqueeze(0).float(),          # (1, 50, 7)
    "observation.umi_pose": torch.tensor(umi_poses[0]).unsqueeze(0).float(),  # (1, 6)
}
batch = apply_umi_sample_relative_transform(batch, remove_umi_pose=False)

# 基准帧（umi_pose[0]）经变换后应为单位变换：pos≈0, rotvec≈0
print("base delta (应接近零):", batch["action"][0, 0, :6])
```

---

## Step 3：ACT 策略训练

### 3.1 训练配置文件

创建 `train_umi_act.yaml`（或使用 draccus CLI 参数）：

```yaml
dataset:
  repo_id: yourname/pick_and_place
  root: /path/to/output_dataset
  episodes: null          # null = 使用全部 episode
  video_backend: pyav

policy:
  type: act

  input_features:
    observation.images.camera0:
      type: VISUAL
      shape: [3, 480, 640]
    observation.state:
      type: STATE
      shape: [1]          # 仅 gripper_width

  output_features:
    action:
      type: ACTION
      shape: [7]          # pos3 + rotvec3 + gripper1

  # ACT 超参数
  chunk_size: 15
  n_action_steps: 15
  n_obs_steps: 1
  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 7
  feedforward_dim: 3200
  dropout: 0.1

  normalization_mapping:
    VISUAL: MEAN_STD
    STATE:  MIN_MAX
    ACTION: MIN_MAX

training:
  num_epochs: 1000
  batch_size: 16
  grad_accumulation_steps: 1
  lr: 1e-4
  lr_scheduler: cosine
  lr_warmup_steps: 500
  save_checkpoint: true
  save_freq: 25

output_dir: /path/to/checkpoints/act_umi
```

### 3.2 添加 sample-relative 变换

在训练脚本的 DataLoader collate 函数中注入 `apply_umi_sample_relative_transform`：

```python
from lerobot.robots.franka_gen_gripper.umi_dataset_transform import apply_umi_sample_relative_transform

def collate_fn(batch_list):
    batch = default_collate(batch_list)
    batch = apply_umi_sample_relative_transform(batch)
    # 变换后 observation.umi_pose 已被删除，action 变为 sample-relative 7D
    return batch

dataloader = DataLoader(dataset, collate_fn=collate_fn, ...)
```

> `apply_umi_sample_relative_transform` 以 `observation.umi_pose`（当前帧绝对位姿）为基准，将 action chunk 各帧转为相对增量，之后从 batch 中删除 `observation.umi_pose` 键。

### 3.3 启动训练

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=yourname/pick_and_place \
    --dataset.root=/path/to/output_dataset \
    --policy.chunk_size=15 \
    --policy.n_action_steps=15 \
    --training.batch_size=16 \
    --output_dir=/path/to/checkpoints/act_umi
```


---

## Step 4：真机轨迹回放验证

在策略训练前，先用数据集原始轨迹直接驱动 Franka，确认坐标变换链路正确。

```bash
python -m lerobot.scripts.lerobot_replay_umi_franka \
    --dataset-root /path/to/output_dataset \
    --episode 0 \
    --robot-ip 192.168.1.104 \
    --camera-extrinsic-yaml /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml \
    --fps 20
```

**重要注意事项**：
- `world_flange` 格式回放内部自动使用 episode 首帧作为固定参考（`ee_at_t0`），`--reference` 参数对此格式无效
- 回放前确保工作空间无障碍物
- 按 ENTER 确认后开始运动，Ctrl-C 随时中止

加上夹爪控制：

```bash
python -m lerobot.scripts.lerobot_replay_umi_franka \
    --dataset-root /path/to/output_dataset \
    --episode 0 \
    --robot-ip 192.168.1.104 \
    --use-gripper \
    --yes    # 跳过确认提示
```

---

## Step 5：策略推理部署

使用异步推理架构：策略服务端（GPU 机器）+ 机器人客户端（NUC）通过 gRPC 通信。

### 5.1 启动策略服务端

在 GPU 机器上：

```bash
cd src/lerobot/async_inference

python policy_server.py \
    --host 0.0.0.0 \
    --port 8081 \
    --fps 15
```

### 5.2 启动机器人客户端

在连接 Franka 的 NUC 上：

```bash
cd src/lerobot/async_inference

python franka_umi_async_client.py \
    --pretrained_name_or_path /path/to/checkpoints/act_umi/checkpoints/000500/pretrained_model \
    --server_address <GPU_IP>:8081 \
    --robot.robot_ip 192.168.1.104 \
    --robot.camera_extrinsic_yaml_path /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml \
    --t0_mode per_chunk \
    --reference ee_at_t0 \
    --use_gripper \
    --fps 15 \
    --task "pick and place"
```

**关键参数说明**：

| 参数 | 参考值         | 说明 |
|---|-------------|---|
| `--t0_mode` | `per_chunk` | 每收到一个新 chunk 刷新 t₀；chunk 内固定，与训练 collate 一致 |
| `--reference` | `ee_at_t0`  | world_flange 格式策略输出的 delta 解释方式 |
| `--fps` | 15          | 控制循环频率，应低于数据集录制帧率 |
| `--actions_per_chunk` | 15          | 每次从动作队列取出并执行的步数 |

---