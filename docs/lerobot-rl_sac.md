# LeRobot RL 框架分析与复现指南

## Context

用户要求对 lerobot 仓库内的 RL 训练栈做一次系统性梳理：
- `src/lerobot/rl/actor.py` —— Actor 服务进程
- `src/lerobot/rl/gym_manipulator.py` —— Gym 环境与 Processor 流水线装配
- `src/lerobot/rl/learner.py` —— Learner 服务进程
- `docs/source/hilserl.mdx` —— 真机 HIL-SERL 教程
- `docs/source/hilserl_sim.mdx` —— 仿真版本教程

目标是搞清楚**实现机制 + 关键配置 + 复现路径**。本文档作为知识库，便于后续修改 RL 相关代码或排错时快速定位。

---

## 一、整体架构（Distributed Actor–Learner over gRPC）

LeRobot 的 RL 实现是 **HIL-SERL 论文**（Luo et al., 2024）的工程化版本，核心思路是**异步分布式 SAC + 人在回路**：

```
┌──────────────────────────────┐                 ┌──────────────────────────────┐
│        ACTOR PROCESS         │                 │        LEARNER PROCESS       │
│  ───────────────────────────  │                 │  ───────────────────────────  │
│  1. 与真实/仿真机器人交互   │  ──Transitions─▶│  1. 累积 ReplayBuffer       │
│  2. 用 policy.actor 推理     │  ──Interactions▶│  2. 每步采样、跑 SAC 更新   │
│  3. 接收 teleop 干预         │                 │  3. 周期性广播 actor 参数   │
│  4. 采样动作 → 发送 transit. │ ◀──Parameters── │  4. 写检查点 + WandB        │
└──────────────────────────────┘                 └──────────────────────────────┘
                gRPC（4 MB 分块流式传输 + 自动重试）
```

两端各自有 `policy = make_policy(...)` 实例：**learner 拿全套优化器训练，actor 只跑 forward**，参数差距通过 `parameters_queue + StreamParameters` 推送同步，避免 200× 速度损失（GIL 注释见 `learner.py:271-273`）。

并发模式由 `policy.concurrency.{actor,learner}` 选 `"threads"` 或 `"processes"`：
- threads → 复用同一 gRPC channel，零拷贝队列
- processes → 用 `torch.multiprocessing.Queue`、spawn start method，channel 在子进程内独立创建

---

## 二、文件级功能拆解

### 2.1 `src/lerobot/rl/actor.py`（actor 进程）

| 函数 | 作用 |
|---|---|
| `actor_cli()` (L103) | 入口；读 config，建立与 learner 的 gRPC 连接（`establish_learner_connection` 重试 30 次×2s），起 3 个并发 worker：`receive_policy / send_transitions / send_interactions`，主线程跑 `act_with_policy` |
| `act_with_policy()` (L210) | RL 主循环：构建 env + 两套 processor → 每步 `policy.select_action()` → `step_env_and_process_transition()` → 累积 transition → `done/truncated` 时 push 到 `transitions_queue`，并触发 `update_policy_parameters()` 拉新参数；按 `cfg.env.fps` 用 `precise_sleep` 控制节拍 |
| `receive_policy()` (L458) | 持续从 learner 端 server-streaming RPC `StreamParameters` 收 actor/discrete_critic 权重，丢进 `parameters_queue` |
| `send_transitions()` / `send_interactions()` (L510 / L560) | 把队列内容用 `client-streaming RPC` + `send_bytes_in_chunks` 切成 ≤2 MB 块发给 learner |
| `update_policy_parameters()` (L652) | 用 `get_last_item_from_queue` 丢弃过期参数只取最新；分别加载 `policy.actor` 与（若存在）`policy.discrete_critic` 的 state_dict |
| `push_transitions_to_transport_queue()` (L684) | 把 transition 搬到 CPU、检查 NaN、`transitions_to_bytes` 后入队 |
| `get_frequency_stats()` / `log_policy_frequency_issue()` | 实时监控策略推理 FPS 是否 ≥ `cfg.env.fps`，否则 warning |

**关键细节**：
- 每个子进程都重新初始化日志文件 + ProcessSignalHandler（SIGINT/SIGTERM 触发 `shutdown_event`，二次按 Ctrl+C 强退）
- `lru_cache(maxsize=1)` 装饰的 `learner_service_client` 保证 gRPC channel 单例（HTTP/2 多路复用）

### 2.2 `src/lerobot/rl/gym_manipulator.py`（环境 + 流水线）

| 类 / 函数 | 作用 |
|---|---|
| `RobotEnv` (L122) | 极简 `gym.Env`：观测 = `agent_pos + pixels`；动作 = 连续 (3,) 或 (4,) 含夹爪；`step()` 仅写关节、不算 reward；reward / done / truncated 全部由后续 processor 负责 |
| `make_robot_env()` (L303) | 区分 `"gym_hil"` 仿真分支与真机分支；真机要求 `cfg.robot` 与 `cfg.teleop` 都已配置 |
| `make_processors()` (L357) | **核心**：根据 config 装配 env_processor 与 action_processor，详见 §三 |
| `step_env_and_process_transition()` (L522) | 把"动作流水线 → env.step → 观测流水线"串起来。会**累加** processor 注入的 reward / done / truncated（如 `RewardClassifierProcessorStep` 的 success_reward） |
| `control_loop()` (L574) | 用于 `mode="record"` 离线数据采集：循环喂中性动作 + 等 teleop 干预，按 episode 写 `LeRobotDataset`（含 video/observation/action/reward/done） |
| `replay_trajectory()` (L746) | `mode="replay"` 离线轨迹回放调试 |
| `main()` (L774) | gym_manipulator CLI 入口，记录数据集 / 测试 reward classifier 都走它 |

**注意**：当前文件顶部 `from joint_observations_processor import ...` 是个**裸导入**，依赖 PYTHONPATH 中存在 `joint_observations_processor`（猜测来自外部 fork），是潜在的破坏点。

### 2.3 `src/lerobot/rl/learner.py`（learner 进程）

| 函数 | 作用 |
|---|---|
| `train_cli()` / `train()` (L106 / L122) | 入口；处理 resume 逻辑、初始化 WandBLogger、起 communication_process 跑 gRPC 服务 |
| `start_learner()` (L604) | 在子进程/线程内创建 `LearnerService`，绑定到 `learner_host:learner_port`，等 shutdown_event |
| `add_actor_information_and_train()` (L251) | **核心训练循环**：<br>① `process_transitions` 把 actor 发来的 transition 喂入 `replay_buffer`（且若是 intervention 也写 offline buffer）<br>② `process_interaction_messages` 写 wandb<br>③ `len(replay_buffer) ≥ online_step_before_learning` 才开始训练<br>④ 每个优化步：执行 `utd_ratio` 次 critic 更新（中间 `utd_ratio-1` 次 + 最后 1 次记录 loss），每 `policy_update_freq` 步做一次 actor + temperature 更新；如有离散动作还要更新 discrete_critic<br>⑤ `policy_parameters_push_frequency` 秒推一次参数<br>⑥ `update_target_networks()` 软更新（tau）<br>⑦ 周期性 `save_training_checkpoint`（保存模型 + 优化器 + replay_buffer 转 LeRobotDataset） |
| `make_optimizers_and_scheduler()` (L761) | 为 actor / critic_ensemble / temperature / (可选) discrete_critic 各建 Adam；shared_encoder=True 时 actor 优化器跳过 `encoder.*` 参数，避免重复更新 |
| `handle_resume_logic()` / `load_training_state()` (L816 / L873) | 通过 `checkpoints/last/pretrained_model/train_config.json` 还原配置 + step + interaction_step |
| `initialize_replay_buffer()` / `initialize_offline_replay_buffer()` (L932 / L975) | 新建空 buffer 或 `from_lerobot_dataset()` 还原 |
| `get_observation_features()` (L1017) | 当 `freeze_vision_encoder=True` 时预先 `get_cached_image_features` 给 actor / critic / discrete_critic 复用，避免三遍 vision forward |
| `check_nan_in_transition()` / `push_actor_policy_to_queue()` | NaN 防护与权重广播 |

---

## 三、Processor 流水线详解（最容易踩坑的地方）

`make_processors()` 装配两条 `DataProcessorPipeline`（`/home/zzq/lerobot/src/lerobot/processor/pipeline.py`）。

### 3.1 env_pipeline（处理观测、reward、done）

按顺序：

1. **`VanillaObservationProcessorStep`** —— `agent_pos→OBS_STATE`、`pixels→OBS_IMAGES.{cam}`，HWC uint8 → BCHW float32/255
2. **`JointVelocityProcessorStep`** *(可选)* —— 有限差分加入关节速度（依赖 fork 模块 `joint_observations_processor`）
3. **`MotorCurrentProcessorStep`** *(可选)* —— 直接读电机电流寄存器
4. **`ForwardKinematicsJointsToEEObservation`** *(可选)* —— 把状态从关节空间替换为 EE 6D 位姿（IK 启用时常配）
5. **`ImageCropResizeProcessorStep`** *(可选)* —— `crop_params_dict` + `resize_size`（推荐 128×128 / 64×64）
6. **`TimeLimitProcessorStep`** *(可选)* —— `control_time_s × fps` 步后置 `truncated=True`；本仓 patch（commit 83322371）支持 `control_time_s ≤ 0` 不限时
7. **`GripperPenaltyProcessorStep`** *(可选)* —— 阻止"已开还开/已关还关"，写入 `complementary_data[discrete_penalty]`
8. **`RewardClassifierProcessorStep`** *(可选)* —— 用预训练 ResNet10 分类器自动给 reward；阈值过则 `done=terminate_on_success`
9. **`AddBatchDimensionProcessorStep`** + **`DeviceProcessorStep`** —— batch 维 + 搬到 GPU

### 3.2 action_pipeline（处理动作 + 干预）

按顺序：

1. **`AddTeleopActionAsComplimentaryDataStep`** —— `teleop_device.get_action()` 写入 `complementary_data["teleop_action"]`
2. **`AddTeleopEventsAsInfoStep`** —— 把 success/intervention/rerecord 等事件写入 `info`
3. **`InterventionActionProcessorStep`** —— `IS_INTERVENTION=True` 时把 teleop_action 替换 policy action；按 success / terminate_episode 设 reward / done
4. **IK 子流水线** *(`inverse_kinematics` 配置存在时)*：
   - `MapTensorToDeltaActionDictStep`：tensor → `{delta_x,y,z[,gripper]}`
   - `MapDeltaActionToRobotActionStep`：包装 `enabled` + 缩放（`end_effector_step_sizes`）
   - `EEReferenceAndDelta`：上升沿 latch 参考位姿 + Δ 累积
   - `EEBoundsAndSafety`：clip 到 `end_effector_bounds`，>`max_ee_step_m` 抛错
   - `GripperVelocityToJoint`：速度→位置，`discrete_gripper=True` 把 {0,1,2} 映射为 {-clip_max,0,clip_max}
   - `InverseKinematicsRLStep`：调 `RobotKinematics.inverse_kinematics(q_curr, T_des)` 解关节，缓存 `IK_solution` 给下一步
5. **`RobotActionToPolicyActionProcessorStep`** —— 把关节 dict 拼回 tensor 发 robot

**关键工作流**：actor 收集到的 `transition[ACTION]` 实际是**经过 IK 的关节动作**（即 `executed_action`），而 `state` 是 EE 空间观测。这与 §四 SAC 训练 batch 的格式呼应。

---

## 四、SAC 策略实现要点（`src/lerobot/policies/sac/`）

### 4.1 网络组件（`modeling_sac.py`）

| 组件 | 作用 |
|---|---|
| `actor` (`Policy`) | TanhMultivariateNormalDiag → `rsample()` 采样连续动作；不输出离散维度 |
| `critic_ensemble` (`CriticEnsemble`) | `num_critics` 个 Q 头（默认 2），训练时取 `min` 防过估 |
| `critic_target` | online critic 的软拷贝，按 `tau=critic_target_update_weight=0.005` 更新 |
| `discrete_critic` *(可选)* | DQN 头，用 `argmax` 选离散夹爪动作；同样有 target 副本 |
| `log_alpha` | 温度 α 的对数参数化，`temperature = exp(log_alpha)` |
| `SACObservationEncoder` | 共享或独立的视觉 + 状态 encoder；支持 `freeze_vision_encoder` |

### 4.2 损失函数（`forward(batch, model=...)`）

- **critic loss**：标准 SAC，`td = r + γ(1-done)·(min_target_Q - α·log_p_next)`，对所有 ensemble 求 MSE 后 sum；含离散动作时把最后一维剥掉
- **actor loss**：`(α·log_p - min_q).mean()`
- **temperature loss**：`-α·(log_p + target_entropy).mean()`（`target_entropy` 默认 `-(action_dim+离散维)/2`）
- **discrete critic loss**：Double-DQN，online 选 argmax + target 估值，可加 `discrete_penalty`

### 4.3 关键超参（`configuration_sac.py`）

| 类别 | 字段 | 默认 |
|---|---|---|
| 数据 | `online_step_before_learning` | 100 |
| | `online_buffer_capacity` / `offline_buffer_capacity` | 100k / 100k |
| | `async_prefetch` | False（建议开） |
| 优化频率 | `utd_ratio` | 1（提高可加速） |
| | `policy_update_freq` | 1 |
| 学习率 | `actor_lr / critic_lr / temperature_lr` | 3e-4 |
| SAC | `temperature_init` | 1.0（实战推荐 1e-2，见教程） |
| | `discount` | 0.99 |
| | `critic_target_update_weight` (tau) | 0.005 |
| 离散 | `num_discrete_actions` | None（启用夹爪 DQN 时设 3） |
| Encoder | `shared_encoder` / `freeze_vision_encoder` | True / True |
| 通讯 | `actor_learner_config.learner_host:port` | 127.0.0.1:50051 |
| | `policy_parameters_push_frequency` | 4 s（推荐 1-2 s） |
| | `queue_get_timeout` | 2 s |
| 部署 | `concurrency.actor / learner` | "threads" / "threads" |
| | `storage_device` | "cpu"（GPU 充裕可改 "cuda"） |

---

## 五、底层基础设施

### 5.1 `ReplayBuffer`（`src/lerobot/rl/buffer.py`）

- 环形缓冲区，`storage_device="cpu"` 节省显存，sample 时再搬到 `device`
- `optimize_memory=True`：next_state 与 state 共享内存，按 `(idx+1)%capacity` 读
- `use_drq=True` 时对图像观测一次性 random_shift（DrQ 数据增强）
- `get_iterator(async_prefetch=True)`：daemon 线程 + Queue 解耦采样和训练
- `from_lerobot_dataset()`：把离线演示转成 transition 灌进 buffer
- `to_lerobot_dataset()`：检查点时把当前 buffer 反序列化为 dataset 落盘
- `concatenate_batch_transitions()`：online+offline batch 拼接（learner 中 `batch_size //= 2`）

### 5.2 gRPC 传输（`src/lerobot/transport/`）

- `services.proto`：`Ready / StreamParameters / SendTransitions / SendInteractions` 4 个 RPC，消息结构 `{TransferState, bytes data}` + `enum {BEGIN, MIDDLE, END}`
- `CHUNK_SIZE=2MB, MAX_MESSAGE_SIZE=4MB`，自动重试 5 次（指数退避）
- 序列化：state / transitions 用 `torch.save/load(weights_only=True)`，interaction message 用 `pickle`

### 5.3 `LearnerService`（`src/lerobot/rl/learner_service.py`）

- `MAX_WORKERS=3, SHUTDOWN_TIMEOUT=10s`
- `StreamParameters` 用 `get_last_item_from_queue` 丢弃过期权重，确保 actor 永远拉到最新
- `seconds_between_pushes` 节流避免带宽爆炸

### 5.4 `ProcessSignalHandler`（`src/lerobot/rl/process.py`）

- 自适应 threads vs processes：从对应模块取 Event
- 注册 SIGINT/SIGTERM/SIGHUP/SIGQUIT，第二次信号直接 `sys.exit(1)` 防止挂死

### 5.5 `WandBLogger.log_dict`（`src/lerobot/rl/wandb_utils.py`）

- 支持多个独立步轴：`custom_step_key="Optimization step"` 和 `"Interaction step"` 各走各的
- 用 `wandb.define_metric(..., hidden=True)` 注册自定义步轴

---

## 六、复现 HIL-SERL 训练的端到端流程

### 6.1 仿真 (`gym_hil`，最快验证 RL 栈)

```bash
pip install -e ".[hilserl]"

# 1. 写 config，task = "PandaPickCubeGamepad-v0"，env.name = "gym_hil"
#    参考：huggingface.co/datasets/lerobot/config_examples/.../gym_hil/train_config.json

# 2. 起 learner（先）
python -m lerobot.rl.learner --config_path path/to/train_gym_hil_env.json

# 3. 起 actor（另一终端）
python -m lerobot.rl.actor --config_path path/to/train_gym_hil_env.json

# 4. wandb dashboard 看 Episodic reward / Intervention rate / Optimization frequency
```

### 6.2 真机（SO-100 / SO-101 / Franka 等）

完整 6 步：

1. **`lerobot-find-joint-limits`** 找关节/EE 边界 → 写入 `processor.inverse_kinematics.end_effector_bounds`
2. **录数据集**（`gym_manipulator.py mode=record`）：
   - gamepad / 键盘 / leader 干预 → 完成任务 → 按"成功"键
   - 输出 `LeRobotDataset` 到 HF Hub 或本地
3. **`crop_dataset_roi.py`** 框选 ROI → 输出 `crop_params_dict`，写回 config
4. *(可选)* **训练 Reward Classifier**：
   - 录数据时设 `terminate_on_success=false` 多收正样本
   - `lerobot-train --config_path reward_classifier_train_config.json`
   - 训完把 `pretrained_path` 写入 config
5. **起 learner**：`python -m lerobot.rl.learner --config_path train_config.json`
6. **起 actor**：`python -m lerobot.rl.actor --config_path train_config.json`
   - 训练初期密集干预 → 后期减少；目标是干预率随 step 单调下降
   - 监控 wandb：`Episodic reward / Intervention rate / Policy frequency [Hz] / Optimization frequency loop [Hz]`

### 6.3 调参建议（来自 `hilserl.mdx`）

- `temperature_init` 建议 `1e-2`（默认 1.0 会让人类干预失效）
- `policy_parameters_push_frequency` 改 1-2 s 提鲜度
- `storage_device="cuda"` 提升 learner 吞吐
- `async_prefetch=True` 让 sample 与训练并行
- `utd_ratio>1` 时注意 critic 容易过拟合，配合 `num_subsample_critics` 用

### 6.4 Resume

```bash
lerobot-train --config_path <output_dir>/checkpoints/last/pretrained_model/train_config.json --resume=true
# learner 与 actor 各自 resume；replay_buffer 也会从 dataset 还原
```

---

## 七、本仓的本地修改（`git status` 提示的 RL 相关变动）

| 文件 | 变更性质 |
|---|---|
| `src/lerobot/rl/actor.py` | 模块导入用 `from rl_queue import Empty`（注意：原版应是 `from queue import Empty`，本仓做了重命名/封装；潜在依赖问题） |
| `src/lerobot/rl/buffer.py` | 已修改（待 diff 确认） |
| `src/lerobot/rl/gym_manipulator.py` | `TimeLimitProcessorStep` 在 `control_time_s ≤ 0` 时跳过；新增 `joint_observations_processor` 模块导入；`control_loop` 中 `if terminated:` 而非 `terminated or truncated` |
| `src/lerobot/rl/learner_service.py` | 已修改（待 diff 确认） |
| `src/lerobot/rl/wandb_utils.py` | 已修改 |

---

## 八、关键文件路径速查表

| 主题 | 路径 |
|---|---|
| Actor 进程 | `src/lerobot/rl/actor.py` |
| Learner 进程 | `src/lerobot/rl/learner.py` |
| gRPC service 实现 | `src/lerobot/rl/learner_service.py` |
| Replay Buffer | `src/lerobot/rl/buffer.py` |
| 信号处理 | `src/lerobot/rl/process.py` |
| 队列工具 | `src/lerobot/rl/rl_queue.py` |
| W&B 日志 | `src/lerobot/rl/wandb_utils.py` |
| Gym 环境装配 | `src/lerobot/rl/gym_manipulator.py` |
| SAC 模型 | `src/lerobot/policies/sac/modeling_sac.py` |
| SAC 配置 | `src/lerobot/policies/sac/configuration_sac.py` |
| 训练 Pipeline 配置 | `src/lerobot/configs/train.py` (`TrainRLServerPipelineConfig`) |
| Processor pipeline 基础 | `src/lerobot/processor/pipeline.py` |
| Transition 数据结构 | `src/lerobot/processor/core.py`、`converters.py` |
| Teleop / 干预 step | `src/lerobot/processor/hil_processor.py` |
| 观测 step | `src/lerobot/processor/observation_processor.py` |
| Action 转换 step | `src/lerobot/processor/delta_action_processor.py`、`gym_action_processor.py`、`policy_robot_bridge.py` |
| EE / IK step (SO100) | `src/lerobot/robots/so100_follower/robot_kinematic_processor.py` |
| Teleop 事件枚举 | `src/lerobot/teleoperators/utils.py` |
| 传输工具 | `src/lerobot/transport/utils.py`、`services.proto` |
| 真机教程 | `docs/source/hilserl.mdx` |
| 仿真教程 | `docs/source/hilserl_sim.mdx` |

---

## Verification（如需验证理解是否正确）

仅做最小化烟雾测试，无需真实硬件：

```bash
# 1. 用 gym_hil 跑一次完整 actor+learner，确认通讯链路通：
python -m lerobot.rl.learner --config_path <train_gym_hil_env.json>
python -m lerobot.rl.actor   --config_path <train_gym_hil_env.json>
# 期望：wandb 出现 Episodic reward 曲线；几分钟内 Optimization step 涨到几百

# 2. 单独跑 gym_manipulator 验证 processor 流水线：
python -m lerobot.rl.gym_manipulator --config_path <env_config.json>
# mode=null：纯交互；mode=record：录数据

# 3. 看 actor & learner 日志：
tail -f outputs/<run>/logs/actor_*.log
tail -f outputs/<run>/logs/learner_*.log
```
