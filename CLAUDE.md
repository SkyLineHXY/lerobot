# CLAUDE.md

本文件为 Claude Code 提供此仓库的工作指导。

## 常用命令

```bash
# 安装
pip install -e ".[dev,test]"
pip install -e ".[smolvla,aloha,pusht]"          # 策略相关
pip install -e ".[hopejr,lekiwi,unitree_g1,reachy2]"  # 机器人相关
pip install -e ".[all]"                           # 除 groot / unitree_g1 外全量

# 代码质量
pre-commit run --all-files
pre-commit install

# 测试
git lfs install && git lfs pull
pytest -sv ./tests
pytest -sv tests/test_specific_feature.py::test_name

# 端到端训练与评估（Makefile）
make test-end-to-end DEVICE=cpu
make test-act-ete-train DEVICE=cpu

# 训练 / 评估
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
lerobot-eval  --policy.path=<checkpoint_dir>/pretrained_model --env.type=aloha
lerobot-train --config_path=<output_dir>/checkpoints/000002/pretrained_model/train_config.json --resume=true

# UMI Franka 数据回放
python -m lerobot.scripts.lerobot_replay_umi_franka \
    --dataset.repo_id=<hf_user/dataset_name> \
    --robot.robot_ip=192.168.1.104 \
    --reference ee_at_t0    # ee_at_t0 | camera_t0 | abs_pos_world_rot

# 其他 CLI
lerobot-record / teleoperate / calibrate / replay / find-cameras / find-port
lerobot-dataset-viz / edit-dataset / train-tokenizer / info
```

## 架构概览

### 包结构（`src/lerobot/`）

| 模块 | 用途 |
|---|---|
| `policies/` | ML 策略（ACT、Diffusion、SmolVLA、Pi0、TDMPC、VQBeT 等） |
| `configs/` | draccus dataclass 配置系统 |
| `datasets/` | LeRobotDataset — Parquet + MP4，HF Hub 集成 |
| `robots/` | 物理机器人硬件抽象层 |
| `motors/` | 低级电机总线驱动（Feetech、Dynamixel、Damiao、Robstride） |
| `cameras/` | 相机驱动（OpenCV、RealSense、ZMQ） |
| `teleoperators/` | 遥操作设备（SO-100/101 leader、手柄、键盘、手机） |
| `scripts/` | CLI 入口脚本 |
| `envs/` | 仿真环境（Aloha、PushT、LIBERO、MetaWorld） |
| `processor/` | 观测→策略输入预/后处理流水线 |
| `rl/` | 强化学习工具（SAC、在线缓冲区、W&B 日志） |
| `transport/` | 异步推理 / gRPC 传输层 |
| `async_inference/` | 异步推理服务 |
| `model/` | 共享模型架构组件 |
| `utils/` | 通用工具函数 |

### 支持的策略

| Policy | 类型 | 说明 |
|---|---|---|
| `act` | 模仿学习 | Action Chunking with Transformers |
| `diffusion` | 模仿学习 | 扩散策略 |
| `tdmpc` | 模型预测控制 | Temporal Difference MPC |
| `vqbet` | 模仿学习 | VQ-BeT |
| `sac` | 强化学习 | Soft Actor-Critic |
| `smolvla` | VLA | SmolVLM 视觉语言动作模型 |
| `pi0` / `pi05` / `pi0_fast` | VLA | π0 系列 |
| `groot` | VLA | Gr00t N1 |
| `wall_x` / `xvla` / `sarm` / `rtc` | VLA/实时控制 | 其他策略 |

### 支持的机器人

`so100_follower` / `so101_follower` / `bi_so100_follower` / `koch_follower` / `lekiwi` / `piper` / `dual_piper` / `hope_jr` / `franka` / `franka_gen_gripper` / `gen_gripper` / `earthrover_mini_plus` / `reachy2` / `unitree_g1`

### 关键接口

**策略**：继承 `PreTrainedPolicy`（`policies/pretrained.py`）。必须定义 `config_class`、`name`、`forward()`（训练）、`select_action()`（推理）。通过 `HubMixin` 集成 HF Hub，检查点为 `model.safetensors` + `config.json`。

**机器人**：继承 `Robot`（`robots/robot.py`）。必须定义 `observation_features`、`action_features`、`connect()`、`disconnect()`、`get_observation()`、`send_action()`。校准文件在 `~/.cache/huggingface/lerobot/calibration/`。

**配置系统**：基于 `draccus` dataclass，`--policy.type=act` 等 CLI 标志选择注册的配置/模型类。每个策略目录含 `configuration_<name>.py` 和 `modeling_<name>.py`。

**数据集**：Parquet（状态/动作）+ MP4（图像），支持 HF Hub 流式加载。`configs/types.py` 定义 `FeatureType` 枚举（`STATE`、`VISUAL`、`ENV`、`ACTION`、`REWARD`、`LANGUAGE`）。

### UMI ee6d 坐标变换（`robots/franka_gen_gripper/`）

实现了 UMI 论文的 EE-at-t₀ 坐标变换，核心文件：`umi_transforms.py`、`franka_gen_gripper.py`、`config_franka_gen_gripper.py`。

**三种变换模式**（混用会导致系统性旋转误差）：

| 模式 | delta_k 语义 | 推理公式 |
|---|---|---|
| `ee_at_t0` | Δ_k = inv(ᵂT_F(t₀))·ᵂT_F(t_k) | `T_BE_t0 @ Δ_k` |
| `camera_t0` | ΔC_k = inv(ᵂT_C(t₀))·ᵂT_C(t_k) | `T_BE_t0 @ T_FC @ ΔC_k @ inv(T_FC)` |
| `abs_pos_world_rot` | 位置相对 t₀ + 姿态绝对 | 混合 |

**数据集预处理**（将相机轨迹转为 EE-at-t₀ 法兰格式）：
```python
from lerobot.robots.franka_gen_gripper import compute_ee_at_t0_deltas, load_flange_to_camera_extrinsic
T_FC = load_flange_to_camera_extrinsic("path/to/camera_transform.yaml")
deltas = compute_ee_at_t0_deltas(W_T_C_list, T_FC)
```

hand-eye 标定文件默认路径：`/home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml`（`parent_frame: panda_link8`）。数学推导详见 `examples/umi_gripper/docs/UMI ee6d 位姿变换推理.md`。

---

## 代码风格

- 行长度：110 字符；代码检查器：`ruff`（规则集 `E, W, F, I, B, C4, T20, N, UP, SIM`）
- Python ≥ 3.12；Docstring 使用 Google 规范
- `mypy` 严格检查仅限：`configs/`、`optim/`、`cameras/`、`motors/`、`transport/`、`envs/`

## Git 规范

每次完整更新后立即 `git add` + `git commit`：
- 只添加与本次修改直接相关的文件，禁用 `git add -A` / `git add .`
- commit message 使用中文，简明描述变更内容和目的
- 禁止跳过 pre-commit hook（禁用 `--no-verify`）
- 未经明确要求不执行 `git push`

## 语言要求

- **代码注释**（含行内注释、docstring）：中文
- **Markdown 文件**：中文
