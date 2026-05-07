# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation (from source)
```bash
pip install -e ".[dev,test]"
# For specific hardware/policy extras:
pip install -e ".[smolvla,aloha,pusht]"
pip install -e ".[pi,groot,wallx,sarm,xvla]"   # 策略相关
pip install -e ".[hopejr,lekiwi,unitree_g1,reachy2]"  # 机器人相关
pip install -e ".[all]"  # 除 groot 和 unitree_g1 外的所有依赖
```

### Code Quality
```bash
pre-commit run --all-files     # Run all linters/formatters (ruff, typos, etc.)
pre-commit install             # Install hooks to run automatically on commit
```

### Testing
```bash
# Ensure test artifacts are available first:
git lfs install && git lfs pull

pytest -sv ./tests                             # Full test suite
pytest -sv tests/test_specific_feature.py      # Single test file
pytest -sv tests/test_specific_feature.py::test_name  # Single test
```

### End-to-End Training & Eval (uses Makefile)
```bash
make test-end-to-end DEVICE=cpu          # All policies
make test-act-ete-train DEVICE=cpu       # ACT policy train
make test-act-ete-eval  DEVICE=cpu       # ACT policy eval
make test-diffusion-ete-train DEVICE=cpu # Diffusion policy train
make test-smolvla-ete-train DEVICE=cpu   # SmolVLA policy train
make test-tdmpc-ete-train DEVICE=cpu     # TDMPC policy train
```

### CLI Entry Points (installed as scripts)
```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
lerobot-eval  --policy.path=<checkpoint_dir>/pretrained_model --env.type=aloha
lerobot-record --robot.type=so100_follower --dataset.repo_id=<hf_user/dataset_name>
lerobot-teleoperate --robot.type=so100_follower
lerobot-calibrate --robot.type=so100_follower
lerobot-replay --robot.type=so100_follower --dataset.repo_id=<hf_user/dataset_name>
lerobot-find-cameras       # 查找可用摄像头
lerobot-find-port          # 查找设备串口
lerobot-find-joint-limits  # 查找关节限位
lerobot-setup-motors       # 配置电机参数
lerobot-setup-can          # 配置 CAN 总线
lerobot-dataset-viz        # 可视化数据集
lerobot-edit-dataset       # 编辑数据集
lerobot-train-tokenizer    # 训练 tokenizer
lerobot-info               # Show environment info
```

### UMI Franka 数据回放（非安装入口，直接调模块）
```bash
python -m lerobot.scripts.lerobot_replay_umi_franka \
    --dataset.repo_id=<hf_user/dataset_name> \
    --robot.robot_ip=192.168.1.104 \
    --reference ee_at_t0    # ee_at_t0 | camera_t0 | abs_pos_world_rot
```

### Resume Training
```bash
lerobot-train --config_path=<output_dir>/checkpoints/000002/pretrained_model/train_config.json --resume=true
```

## Architecture Overview

### Package Layout (`src/lerobot/`)

| Module | Purpose |
|---|---|
| `policies/` | 所有 ML 策略实现（ACT、Diffusion、SmolVLA、Pi0、TDMPC、VQBeT、Gr00t、Wall-X 等） |
| `configs/` | 基于 `draccus` 的 dataclass 配置系统 |
| `datasets/` | `LeRobotDataset` — Parquet + MP4 格式，HF Hub 集成 |
| `robots/` | 物理机器人硬件抽象层 |
| `motors/` | 低级电机总线驱动（Feetech、Dynamixel、Damiao、Robstride） |
| `cameras/` | 相机驱动（OpenCV、RealSense、ZMQ） |
| `teleoperators/` | 遥操作设备（SO-100/101 leader、手柄、键盘、手机） |
| `scripts/` | 所有 CLI 入口脚本（含 `lerobot_replay_umi_franka.py`） |
| `envs/` | 仿真环境封装（Aloha、PushT、LIBERO、MetaWorld） |
| `processor/` | 观测到策略输入之间的预/后处理流水线 |
| `rl/` | 强化学习工具（SAC、在线缓冲区、W&B 日志） |
| `optim/` | 优化器和学习率调度配置 |
| `transport/` | 异步推理 / gRPC 传输层 |
| `async_inference/` | 异步推理服务（grpc + matplotlib） |
| `data_processing/` | 数据处理工具函数 |
| `model/` | 共享模型架构组件 |
| `utils/` | 通用工具函数库 |
| `templates/` | 新策略/机器人的代码模板 |

### Examples (`examples/`)

| 目录 | 内容 |
|---|---|
| `examples/umi_gripper/` | UMI Gen Gripper 硬件验证脚本、相机诊断工具 |
| `examples/umi_gripper/docs/` | UMI ee6d 位姿变换推理文档（坐标系定义、SE(3) 公式、Franka 实现） |

### Supported Policies (`policies/`)

| Policy | 类型 | 说明 |
|---|---|---|
| `act` | 模仿学习 | Action Chunking with Transformers |
| `diffusion` | 模仿学习 | 扩散策略 |
| `tdmpc` | 模型预测控制 | Temporal Difference MPC |
| `vqbet` | 模仿学习 | VQ-BeT |
| `sac` | 强化学习 | Soft Actor-Critic |
| `smolvla` | VLA | SmolVLM 视觉语言动作模型 |
| `pi0` | VLA | π0 策略 |
| `pi05` | VLA | π0.5 策略 |
| `pi0_fast` | VLA | π0 快速版本 |
| `groot` | VLA | Gr00t N1（Eagle2 视觉语言模型） |
| `wall_x` | VLA | Qwen2.5-VL 基础策略 |
| `xvla` | VLA | xVLA 视觉语言模型 |
| `sarm` | VLA | Spatial Action Representation |
| `rtc` | 实时控制 | Real-Time Control 策略 |

### Supported Robots (`robots/`)

| Robot | 说明 |
|---|---|
| `so100_follower` | SO100 跟随臂 |
| `so101_follower` | SO101 跟随臂 |
| `bi_so100_follower` | 双臂 SO100 |
| `koch_follower` | Koch 跟随臂 |
| `lekiwi` | Lekiwi 移动机器人 |
| `piper` | Piper 单臂 |
| `dual_piper` | 双臂 Piper |
| `hope_jr` | HOPE Jr. 机器人 |
| `franka` | Franka Emika 机械臂 |
| `franka_gen_gripper` | Franka + Gen2 夹爪 |
| `gen_gripper` | 通用夹爪 |
| `earthrover_mini_plus` | EarthRover Mini Plus 移动底座 |
| `reachy2` | Reachy 2 人形机器人 |
| `unitree_g1` | Unitree G1 人形机器人 |

### Configuration System

LeRobot 使用 `draccus`（基于 dataclass 的 CLI 解析器）。所有配置均为普通 Python `@dataclass` 对象。主训练配置为 `configs/train.py` 中的 `TrainPipelineConfig`，由以下部分组成：
- `DatasetConfig` — 数据集加载参数
- `PreTrainedConfig`（每个策略的子类）— 模型超参数
- `EvalConfig`、`WandBConfig`、`PeftConfig` — 可选设置

策略通过 `PreTrainedConfig` 注册为 `draccus.ChoiceRegistry` — `--policy.type=act` CLI 标志选择对应的配置/模型类。每个策略目录包含：
- `configuration_<name>.py` — 继承 `PreTrainedConfig` 的 `@dataclass` 配置
- `modeling_<name>.py`（或类似文件）— 继承 `PreTrainedPolicy` 的 `nn.Module`

### Policy Interface

所有策略继承自 `PreTrainedPolicy`（`policies/pretrained.py`），继承自 `nn.Module`：
- 必须定义 `config_class`（`PreTrainedConfig` 子类）和 `name`（字符串标识符）
- 核心方法：`forward()` 用于训练损失，`select_action()` 用于推理
- 通过 `HubMixin` 集成 HF Hub：`from_pretrained()` / `push_to_hub()` / `save_pretrained()`
- 检查点保存为 `model.safetensors` + `config.json`

### Robot Interface

所有机器人继承自 `Robot`（`robots/robot.py`）：
- 必须定义 `observation_features`、`action_features` 属性及 `connect()`、`disconnect()`、`get_observation()`、`send_action()` 方法
- 校准文件存储于 `~/.cache/huggingface/lerobot/calibration/robots/<robot_name>/<id>.json`

### Dataset Format

`LeRobotDataset` 将 episode 存储为：
- Parquet 文件（状态/动作/奖励张量）
- MP4 视频（图像观测）
- HF Hub 上的 `meta/` 目录中的数据集元数据
- 支持从 HF Hub 流式加载，无需完整下载

### Feature Type System

`configs/types.py` 定义 `FeatureType` 枚举：`STATE`、`VISUAL`、`ENV`、`ACTION`、`REWARD`、`LANGUAGE`。策略输入/输出描述为 `PolicyFeature(type=FeatureType.VISUAL, shape=(C, H, W))`，用于连接数据集特征与策略的 `input_features`/`output_features` 字典。

### UMI ee6d 坐标变换系统（`robots/franka_gen_gripper/`）

`franka_gen_gripper` 实现了 UMI（Universal Manipulation Interface）论文的 EE-at-t₀ 坐标变换：

**核心文件**：
- `umi_transforms.py` — SE(3) 数学工具与三种变换模式
- `franka_gen_gripper.py` — `send_umi_action(delta_k, T_BE_t0)` 推理接口
- `config_franka_gen_gripper.py` — `camera_extrinsic_yaml_path` 指向 hand-eye 标定 YAML

**两种 ee6d 约定**（混用会导致系统性旋转误差）：

| 模式 | delta_k 语义 | 推理公式 | 适用场景 |
|---|---|---|---|
| `ee_at_t0` | Δ_k = inv(ᵂT_F(t₀))·ᵂT_F(t_k) | `T_BE_t0 @ Δ_k` | 推荐；数据集预处理后格式 |
| `camera_t0` | ΔC_k = inv(ᵂT_C(t₀))·ᵂT_C(t_k) | `T_BE_t0 @ T_FC @ ΔC_k @ inv(T_FC)` | 原始 SLAM 相机轨迹 |
| `abs_pos_world_rot` | 位置相对 t₀ + 姿态绝对 | 混合 | 特殊预处理变体 |

**数据集预处理**（将相机轨迹转为 EE-at-t₀ 法兰格式）：
```python
from lerobot.robots.franka_gen_gripper import compute_ee_at_t0_deltas, load_flange_to_camera_extrinsic
T_FC = load_flange_to_camera_extrinsic("path/to/camera_transform.yaml")
deltas = compute_ee_at_t0_deltas(W_T_C_list, T_FC)  # Δ_k 列表，Δ_0 = I
```

**hand-eye 标定文件**（默认路径）：
```
/home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml
parent_frame: panda_link8  # 必须为此值
child_frame:  camera_color_optical_frame
```

数学推导详见 `examples/umi_gripper/docs/UMI ee6d 位姿变换推理.md`。

---

## Code Style

- 行长度：110 字符
- 代码检查器：`ruff`，规则集 `E, W, F, I, B, C4, T20, N, UP, SIM`
- Python ≥ 3.12
- Docstring 使用 Google 规范
- `mypy` 已启用但大部分宽松；严格检查仅在 `configs/`、`optim/`、`cameras/`、`motors/`、`transport/`、`envs/` 上执行

## Git 规范

每次完成一次完整的代码仓库更新后，必须立即执行一次 `git add` 和 `git commit`，将变更提交到本地仓库。

- 只添加与本次修改直接相关的文件，不使用 `git add -A` 或 `git add .`
- commit message 须简明描述本次变更的内容和目的，使用中文编写
- 不得跳过 pre-commit hook（禁止使用 `--no-verify`）
- 未经用户明确要求，不执行 `git push`

## Language Requirements

- **代码注释**：所有代码注释（包括行内注释、函数/类的 docstring）均须使用中文编写。
- **Markdown 文件**：新建或修改的 Markdown 文件内容均须使用中文编写。
