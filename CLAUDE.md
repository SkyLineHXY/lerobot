# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

这是 LeRobot 的一个 fork（`origin` = `SkyLineHXY/lerobot`），在上游之外有两块自研内容：
**`rlt/` 在线强化学习** 和 **Piper / DualPiper 主从遥操作数据采集**。改动落在这两块时，
务必先读下面对应的章节 —— 那里记的都是踩过的坑，光看代码看不出来。

## 常用命令

```bash
# 安装
pip install -e ".[dev,test]"
pip install -e ".[smolvla,aloha,pusht]"          # 策略相关
pip install -e ".[hopejr,lekiwi,unitree_g1,reachy2]"  # 机器人相关
pip install -e ".[all]"                           # 除 groot / unitree_g1 外全量

# 代码质量
pre-commit run --all-files
pre-commit run --files <path1> <path2>            # 只查改动的文件，快得多
pre-commit install

# 测试
git lfs install && git lfs pull                   # 不拉 LFS，涉及 artifacts 的测试会失败
pytest -sv ./tests
pytest -sv tests/scripts/test_record_piper.py::test_name    # 单个测试
pytest -q tests/rlt tests/scripts                 # 按目录跑；tests/ 结构镜像 src/lerobot/

# 端到端训练与评估（Makefile）
make test-end-to-end DEVICE=cpu
make test-act-ete-train DEVICE=cpu

# 训练 / 评估
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
lerobot-eval  --policy.path=<checkpoint_dir>/pretrained_model --env.type=aloha
lerobot-train --config_path=<output_dir>/checkpoints/000002/pretrained_model/train_config.json --resume=true

# 其他 CLI
lerobot-record / teleoperate / calibrate / replay / find-cameras / find-port / setup-can
lerobot-dataset-viz / edit-dataset / train-tokenizer / info
```

### Piper 数据采集与回放（本 fork 新增）

```bash
# CAN 口上电后先起来：_l = leader（主臂），_f = follower（从臂）
./can_config.sh          # 按 USB 物理口重命名并激活 CAN；./find_all_can_port.sh 查端口

# 采集。第一次上机务必先 dry_run + init_mode=none，只读不下发
lerobot-record-piper --config_path examples/piper/record_piper.yaml \
    --collection.dry_run=true --collection.init_mode=none
lerobot-record-piper --config_path examples/piper/record_piper.yaml       # 单臂
lerobot-record-piper --config_path examples/piper/record_dual_piper.yaml  # 双臂

# 回放：画 state/action 曲线、拼多路相机视频，可选真机重放
lerobot-replay-piper --root=<dataset_root> --repo_id=<repo_id> --episode=0

# 遥操作链路体检（不写数据集，只量每一拍花在哪）
python -m lerobot.rlt.teleop_check --config_path examples/rlt/teleop_check.yaml --dry_run=true

# RLT 在线强化学习
lerobot-rlt-train-token / lerobot-rlt-train-online / lerobot-rlt-eval-token
python -m lerobot.rlt.train_online --config_path examples/rlt/mock_online.yaml   # 无硬件冒烟
```

## 架构概览

### 包结构（`src/lerobot/`）

| 模块 | 用途 |
|---|---|
| `policies/` | ML 策略（ACT、Diffusion、SmolVLA、Pi0、TDMPC、VQBeT、RLT 等） |
| `configs/` | draccus dataclass 配置系统 |
| `datasets/` | LeRobotDataset — Parquet + MP4，HF Hub 集成；`repair.py` 崩溃修复 |
| `robots/` | 物理机器人硬件抽象层 |
| `motors/` | 低级电机总线驱动（Feetech、Dynamixel、Damiao、Robstride、Piper） |
| `cameras/` | 相机驱动（OpenCV、RealSense、ZMQ、HIK） |
| `teleoperators/` | 遥操作设备（SO-100/101 leader、Piper leader、手柄、键盘、手机） |
| `scripts/` | CLI 入口脚本 |
| `rlt/` | **在线 RL 的外围**：训练循环、chunk 回放缓冲、环境、人工干预（模型本体在 `policies/rlt/`） |
| `envs/` | 仿真环境（Aloha、PushT、LIBERO、MetaWorld） |
| `processor/` | 观测→策略输入预/后处理流水线 |
| `rl/` | 强化学习工具（SAC、在线缓冲区、W&B 日志） |
| `transport/` | 异步推理 / gRPC 传输层 |
| `async_inference/` | 异步推理服务 |
| `model/` | 共享模型架构组件 |
| `utils/` | 通用工具函数；`status_view.py` 为 OpenCV 操作员视图 |

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

> ⚠️ **入口脚本里不能写 `from __future__ import annotations`。**
> `lerobot.configs.parser.wrap()` 直接读 `inspect.getfullargspec(fn).annotations`；PEP 563 一开，
> draccus 拿到的是字符串而不是 dataclass 类型，配置解析会以很难懂的方式挂掉。
> `teleop_check.py`、`lerobot_record_piper.py` 顶部都有这条注释，别顺手"补上"它。

**数据集**：Parquet（状态/动作）+ MP4（图像），支持 HF Hub 流式加载。`configs/types.py` 定义 `FeatureType` 枚举（`STATE`、`VISUAL`、`ENV`、`ACTION`、`REWARD`、`LANGUAGE`）。

---

## 写 LeRobot v3 数据集的硬约束

自己拼帧写数据集（而不是走 `lerobot-record`）时，下面每一条都在实际调试中咬过人：

**`add_frame(frame)` 的校验**（`datasets/utils.py::validate_frame`）：
- `task` 是 **frame dict 里的一个键**，不是单独的函数参数；
- 键集合必须**恰好等于** `set(features) - DEFAULT_FEATURES` —— 不能少也不能多。
  尤其**不要**自己塞 `timestamp`，它会被判成 extra key 而报错；
- 数值特征必须是 dtype/shape 精确匹配的 `np.ndarray`。`int64` 标量列要写成
  `np.array([i], dtype=np.int64)`（形状 `(1,)`），裸 int 过不了；
- 图像是 `np.uint8` HWC **RGB** —— `RealSenseCameraConfig` / `OpenCVCameraConfig` 的
  `color_mode` 默认就是 RGB，所以写数据集直接透传，反倒是喂给 cv2 显示时才需要转 BGR。

**`build_dataset_frame` 只认 1-D float32 和图像**，`int64` 的列会被静默跳过 —— 需要
`subtask_index` / `back_event` 这类整型列时只能手工构造 features 和 frame。

**`meta/subtasks.parquet` 没有写入方**。仓库代码只在 `__getitem__` 里读它
（`meta.subtasks.iloc[idx].name`）。光有 `subtask_index` 列，读回来的子任务是空的 ——
写库的脚本必须自己产出这个 parquet（见 `lerobot_record_piper._write_subtasks_parquet`）。

**耐久性是不对称的**：元数据每集刷盘，而 episode 数据走一个常驻 `pq.ParquetWriter`，
footer 只在 `close()` 时才写。`kill -9` 会留下"元数据说有 3 集、数据只有 2 集"的目录，
而 `LeRobotDataset.__init__` 遇到不一致会**回退去访问 HuggingFace Hub**，本地 repo_id 直接挂住。
所以：
- 每存一集调 `dataset.meta._flush_metadata_buffer()`（缓冲默认 10 集）；
- 每 N 集调 `dataset._close_writer()` 并置 `dataset._writer_closed_for_reading = True`
  （不置这个标志，下一集会在同一路径重开 writer 把已写内容截断）；
- 主循环包在 `with VideoEncodingManager(dataset):` 里收尾编码器；
- 续采时 `repair_dataset_consistency(root)`（`datasets/repair.py`）**必须在构造
  `LeRobotDataset` 之前**调用。它保留最长的合法前缀，多余的进 `_quarantine_*` 而不是删掉。

**其他两处**：`LeRobotDatasetMetadata.create` 用的是 `root.mkdir(exist_ok=False)`，目录已存在
直接 `FileExistsError`；续采构造出来的 `episode_buffer` 是 `None`，要补一句
`dataset.episode_buffer = dataset.create_episode_buffer()`，否则没加帧就丢弃会解引用 `None`。

---

## Piper 主从遥操作与数据采集

主脚本 `scripts/lerobot_record_piper.py`（采集）/ `lerobot_replay_piper.py`（回放），
配置在 `examples/piper/record_piper.yaml`、`record_dual_piper.yaml`。
底层复用 `Piper`（从臂）+ `PiperLeader`（主臂，带重力补偿），双臂 = 两组实例，**不走 `DualPiper` 类**。

**CAN 带宽是硬约束，不是调优项。** gs_usb 写一帧 ≈ 2.3 ms，一次 `JointCtrl` 是 3 帧。
`PiperMotorsBus.write()` 每拍固定发 MotionCtrl_2 + JointCtrl + GripperCtrl = **5 帧 ≈ 11.5 ms/臂**，
双臂 23 ms —— 30 Hz 的 33 ms 预算去掉大半，这就是双臂卡顿的根因。采集脚本的 `ArmPair.send()`
因此绕开 `send_action()`，把 5 帧压到常态 3 帧：模式帧按秒级间隔重发（固件记得住当前模式），
夹爪只在目标真的变了才发。

**主臂的重力补偿靠 `JointMitCtrl(kp=0, kd=0)`**（`teleoperators/piper_leader/gravity_compensation.py`），
只发力矩。删掉主臂就拖不动了 —— 从臂侧的 MIT 阻抗控制已按要求移除，但**主臂这支必须留着**。

**`Piper.connect()` 默认 `calibrate=True`**，会立刻把从臂开回 home 位。要自己决定初始化方式时
必须传 `calibrate=False`，否则 `init_mode` 还没生效手臂已经动过了。

**限速 `rate_limit_joints()`（`rlt/piper_env.py`）是整体等比缩放，不是逐关节裁剪** ——
它保持方向不变。脚本结束会打印限速饱和率，饱和率高说明记录的动作已经不等于人的意图，
应调大 `max_joint_step_rad` 重采。

**action 记什么**（`collection.action_source`）：
- `follower_next`（默认）—— 从臂在**下一拍**的实测位姿。示教时从臂并不能 100% 跟上主臂，
  记主臂指令等于记了个没被执行的动作。代价是每集丢掉最后一帧。
- `leader` —— 限速后实际下发给从臂的目标。
- ⚠️ 没有"当拍从臂位姿"这个选项：那会让 `action[t] ≡ state[t]`，策略学到恒等映射，推理时手臂永远不动。

**关节命名**：单臂 `joint_1.pos..joint_7.pos`（6 关节 + 夹爪）；双臂左 `joint_1..7`、右 `joint_8..14`，
与 `DualPIPERConfig` 已有约定一致。

---

## 固定节拍控制回路的实时性

30 Hz 掉到 21 Hz 的实测案例，四个原因各自独立，改一个不够：

1. **`async_read()` 会清掉 `new_frame_event`**，于是下一次调用必然阻塞到相机吐出**新**一帧
   （60 fps 最多等 16.7 ms，相机打嗝就等满 200 ms 超时）。固定节拍的回路要的是"现在最新那帧"，
   用非阻塞的 **`read_latest(max_age_ms=...)`**（RealSense / OpenCV / ZMQ / HIK 都有）。
2. **`add_frame` 会阻塞在流式编码器队列上**（实测 p95 63 ms / max 214 ms）。改成投进有界
   `queue.Queue`，由专门的写帧线程落盘；队列满时**丢帧**而不是拖住控制回路 —— 回路一卡，从臂跟随就抖。
3. **`cv2.imshow` / `waitKey` 一次可能阻塞几十毫秒**，而 cv2 的 GUI 又必须在主线程泵。
   只能把**控制回路挪到后台线程**，主线程留给 GUI，而不是反过来。
4. 双臂的读写要并行提交（`ThreadPoolExecutor`）并 fire-and-forget，在下一拍开头再收。

诊断先看脚本结束打印的分段 p50/p95/max 和写帧队列峰值：队列长期贴着上限 = 视频编码真的跟不上，
那就降分辨率或换 `video_codec`，继续调别处没用。

---

## 代码风格

- 行长度 110；`ruff` 规则集 `E, W, F, I, B, C4, T20, N, UP, SIM`
- Python ≥ 3.12；Docstring 用 Google 规范
- `mypy` 严格检查仅限：`configs/`、`optim/`、`cameras/`、`motors/`、`transport/`、`envs/`
- **注释（含行内注释、docstring）与 Markdown 文件一律用中文**
- **不要写冗余注释**：只解释"为什么"和不明显的约束，别复述代码在做什么
- 收尾路径（`finally` / `disconnect` / `close`）里每一步单独 try/except 并记日志 ——
  收尾时冒出的异常会顶掉真正那个错误，上机排障时看到的就成了个无关的异常

## 测试

- `tests/` 结构镜像 `src/lerobot/`；硬件相关的测试一律用假的 bus / robot / dataset，不碰真机
- 需要真数据集的测试直接建 `LeRobotDataset`，别 mock 它 —— 单独钉住"写"和"读"两端会漏掉
  schema 对不上的问题（往返测试就抓到过 `meta/subtasks.parquet` 缺失和图像通道顺序）

## Git 规范

每次完整更新后立即 `git add` + `git commit`：
- 只添加与本次修改直接相关的文件，**禁用 `git add -A` / `git add .`**；
  工作区常有用户自己预暂存的改动，`git commit` 不带路径会把它们一起卷进去
- commit message 使用中文，简明描述变更内容和目的
- 禁止跳过 pre-commit hook（禁用 `--no-verify`）
- 未经明确要求不执行 `git push`
