# UMI + Franka 数据集训练指南

本文档描述从 DAS-Gripper mcap 数据到 ACT 策略训练、真机部署的完整流程，基于 UMI（Universal Manipulation Interface）论文的 **sample-relative t₀** 约定实现。

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

UMI 论文的关键设计：每次推理窗口的 **最后一帧观测** 作为当前 t₀，动作 chunk 是相对于此时刻 EE 位姿的增量序列。

```
t₀ = 推理时刻机器人 EE 当前位姿（滑动刷新，非 episode 固定值）
Δ_k = inv(W_T_F(t₀)) @ W_T_F(t₀ + k·dt)
```

**训练/推理必须保持一致**：
- 训练时：`apply_umi_sample_relative_transform` 将数据集绝对位姿转为 sample-relative
- 推理时：每步 `capture_t0()` 刷新参考系，策略输出直接作为 delta_k 使用

### world_flange 数据集格式

| 字段 | 维度 | 内容 |
|---|---|---|
| `action` | 7D | `[pos_x_W, pos_y_W, pos_z_W, rotvec_x_W, rotvec_y_W, rotvec_z_W, gripper]`（VIO 世界系绝对法兰位姿 + 夹爪宽度） |
| `observation.state` | 1D | `[gripper_width]`（仅夹爪宽度，避免绝对位姿作为 state 引入冗余） |
| `observation.umi_pose` | 6D | `[pos_x_W, pos_y_W, pos_z_W, rotvec_x_W, rotvec_y_W, rotvec_z_W]`（同一帧绝对位姿，供 sample-relative processor 使用） |
| `observation.images.camera0` | H×W×3 | RGB 图像（鱼眼中心镜头） |

> **旧格式（ee_at_t0_flange）**：action/state 均为 8D `[pos_xyz, quat_xyzw, gripper]`，以 episode 首帧为固定 t₀。新项目推荐使用 world_flange 格式。

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
FrankaUMIClient + per_step t₀ 刷新 → 真机推理
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
  chunk_size: 100
  n_action_steps: 100
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

> `apply_umi_sample_relative_transform` 以 `observation.umi_pose`（当前帧绝对位姿）为基准，将 action chunk 各帧转为相对增量，之后从 batch 中删除 `observation.umi_pose` 键（策略不需要此输入）。

### 3.3 启动训练

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=yourname/pick_and_place \
    --dataset.root=/path/to/output_dataset \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --training.batch_size=16 \
    --output_dir=/path/to/checkpoints/act_umi
```

**恢复训练**：

```bash
lerobot-train \
    --config_path=/path/to/checkpoints/act_umi/checkpoints/000050/pretrained_model/train_config.json \
    --resume=true
```

---

## Step 4：真机轨迹回放验证

在策略训练前，先用数据集原始轨迹直接驱动 Franka，确认坐标变换链路正确。

### 4.1 dry-run（不连接机器人）

检查计算出的目标位姿是否合理：

```bash
python -m lerobot.scripts.lerobot_replay_umi_franka \
    --dataset-root /path/to/output_dataset \
    --episode 0 \
    --dry-run \
    --camera-extrinsic-yaml /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml
```

输出示例（world_flange 格式会自动以首帧为 t₀）：

```
frame    0: pos=[0.0, 0.0, 0.0] rotvec=[0.0, 0.0, 0.0] grip=0.0850
frame  150: pos=[0.0412, -0.0081, -0.0234] rotvec=[...] grip=0.0210
frame  299: pos=[0.0018, 0.0003, -0.0009] rotvec=[...] grip=0.0850
```

### 4.2 真机回放

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
    --t0_mode per_step \
    --reference ee_at_t0 \
    --use_gripper \
    --fps 15 \
    --task "pick and place"
```

**关键参数说明**：

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--t0_mode` | `per_step` | 每步刷新 t₀，与 world_flange 数据集对齐 |
| `--reference` | `ee_at_t0` | world_flange 格式策略输出的 delta 解释方式 |
| `--fps` | 15 | 控制循环频率，应低于数据集录制帧率 |
| `--actions_per_chunk` | 50 | 每次从动作队列取出并执行的步数 |

**空跑验证**（不执行运动）：

```bash
python franka_umi_async_client.py \
    --pretrained_name_or_path /path/to/checkpoint \
    --dry_run
```

---

## 手眼标定文件格式

```yaml
# /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml
camera_transform:
  translation:
    x: 0.0423
    y: -0.0310
    z: 0.0712
  rotation:
    x:  0.6532
    y: -0.6532
    z:  0.2706
    w:  0.2706
  parent_frame: panda_link8          # 必须为此值
  child_frame:  camera_color_optical_frame
```

> `parent_frame` 必须是 `panda_link8`。若使用不同法兰连杆，需修改 `load_flange_to_camera_extrinsic` 中的校验逻辑。

---

## 常见问题排查

### 策略只在起始位置小幅晃动

**根因**：训练时 t₀ 约定与推理不一致（episode-level vs sample-level）。

**排查**：
1. 确认数据集使用 `--pose-format=world_flange` 生成（action 首帧不应全为 0）
2. 确认训练 DataLoader 中调用了 `apply_umi_sample_relative_transform`
3. 确认推理时 `t0_mode=per_step`

### 回放姿态发散

**根因**：常见两个原因。

1. `world_flange` 绝对位姿被当成 delta 直接发送（坐标系混用）
2. 循环体未解析当前帧位姿（`_row_to_se3_and_gripper` 调用缺失）

**排查**：使用 `--dry-run` 检查首帧输出，`pos` 应全为 `[0.0, 0.0, 0.0]`（episode-relative delta 的首帧等于单位变换）。

### `TypeError: only 0-dimensional arrays can be converted to Python scalars`

**根因**：LeRobot 将 `shape=(1,)` 特征映射为 HF 标量 `Value`，传入 `np.array([x])` 会失败。

**修复**：传入 numpy 标量 `np.float32(x)` 而非 1D 数组。

参见 `lerobot/datasets/utils.py:581`：`shape == (1,)` → `datasets.Value(dtype=...)`.

### `'camera_transform' key not found`

手眼标定 YAML 格式不正确，确保顶层键为 `camera_transform`，子键包含 `translation`、`rotation`、`parent_frame`、`child_frame`。

### 推理帧率低于目标

- 降低 `--fps`（客户端控制循环频率）
- 增大 `--actions_per_chunk`（减少 gRPC 请求频次）
- 关闭 `--gripper-camera-count`（减少串口/USB 负载）
