# RLT：基于 RL Token 的 VLA 在线强化学习

RLT 在冻结的 SmolVLA 之上增加一个轻量级 actor-critic，通过**在线强化学习（Online RL）+ 人工示教纠正**，将基础策略进一步优化到可实际使用的水平。

核心模型组件——包括 RL token、chunk actor-critic、controller 和相关配置——位于：

```text
lerobot/policies/rlt/
```

本目录则包含围绕 RLT 的完整训练与运行基础设施：训练入口、chunk 级 replay buffer、环境封装、异步 learner，以及人在回路（Human-in-the-Loop, HIL）支持。

---

## 1. 两个阶段

```text
阶段 1  train_rl_token
  演示数据集 ──► 冻结的 SmolVLA 前缀嵌入 ──► RL token 编码/解码器
                                              │
                                              └─► outputs/rl_token/
                                                    rl_token.pt
                                                    policy_preprocessor.json   ← 归一化统计量
                                                    policy_postprocessor.json

阶段 2  train_online
  ┌─ rollout 主线程 ────────────────────┐      ┌─ learner 线程 ──────────┐
  │  obs ─► preprocessor ─► compute_x   │      │  sample(batch)          │
  │  x ─► actor(x, ref) ─► 归一化动作   │◄────►│  critic/actor 梯度步    │
  │  ─► postprocessor ─► env.step       │      │  发布 actor 权重        │
  │  人可随时按空格接管                 │      └─────────────────────────┘
  └─────────────────┬───────────────────┘                  ▲
                    └──────► ChunkReplayBuffer ────────────┘
```

阶段 2 以及真机运行时，都必须使用**完全相同的统计量**进行归一化和反归一化。否则，`z_rl` 和参考 chunk 会发生静默漂移：系统通常不会报错，但训练基本无法收敛。

---

## 2. 代码架构

```text
rlt/
├── train_rl_token.py     阶段 1：RL token 表征学习训练入口
├── eval_rl_token.py      阶段 1：表征质量诊断（重建 / 表征坍缩 / 线性可解码 / 时序结构）
├── train_online.py       阶段 2：在线强化学习入口
├── rollout.py            RolloutWorker：规划一个 chunk，并执行 ChunkRecord
├── learner.py            LearnerThread + ActorMirror：异步梯度更新与权重发布
├── backends.py           ThreadBackend / ProcessBackend：两种并发模式的统一接口
├── replay_buffer.py      ChunkReplayBuffer：chunk 级 transition，支持 stride 子采样
├── collect_libero.py     在 LIBERO 中通过键盘 / 手柄示教，采集为 LeRobotDataset
├── teleop_check.py       真机 HIL 链路体检（不训练、不加载 VLA）
├── gravity_probe.py      领臂重力补偿调参
├── envs/
│   ├── base.py           ChunkEnv 协议 + 归一化边界（ActionNormalizer）
│   ├── __init__.py       make_chunk_env() 工厂
│   ├── mock.py           抽象 6-DoF 任务：只验证管道，不提供有效算法信号
│   ├── libero.py         LIBERO 仿真：上真机前的完整闭环验证
│   └── piper.py          真实 Piper 机械臂环境
├── distributed/          多进程后端（见 5.4），仅在 mode=processes 时导入
│   ├── messages.py       线协议：buffer 操作的编解码
│   ├── client.py         rollout 侧：RemoteBufferSink / RemoteActorMirror
│   └── learner_proc.py   learner 侧：gRPC 服务 + buffer + 梯度循环 + checkpoint
└── teleop/
    ├── keys.py           键盘后端（termios / pynput）+ 算子按键状态机
    ├── base.py           InterventionResult / InterventionManager 契约
    ├── device.py         DeviceIntervention：键盘 / 手柄 / SpaceMouse
    └── piper_leader.py   PiperLeaderIntervention：领臂遥操作
```

---

## 3. 数据流

```text
env.reset() ─► obs
   │
   ├─ obs_to_batch()          原始观测 ─► stage-1 preprocessor ─► 模型 batch
   │                          （LIBERO 还需先经过 LiberoProcessorStep，将嵌套状态摊平）
   ▼
controller.plan_chunk(batch)
   ├─ SmolVLA 前缀前向 ─► 选 token ─► RL token ─► z_rl
   ├─ x = concat(z_rl, 本体感知)
   ├─ ref_full = 通过 flow matching 采样得到的 VLA 参考 chunk
   └─ action_chunk = actor(x, ref[:C])        ← 零初始化残差，第 0 步与 base VLA 逐位一致
   ▼
逐步执行 C 步：
env.step(action[j]) ─► postprocessor 反归一化 ─► 物理命令
   ▼
ChunkRecord ─► buffer.add_chunk()
```

**归一化边界只有一处，但包含两个方向：**

- **出**：`postprocessor` 将归一化动作还原为物理量，例如关节弧度，或 LIBERO 中的 `[-1, 1]` delta EE。
- **入**：`ActionNormalizer`（`envs/base.py`）将人工遥操作产生的物理命令映射回归一化空间。

人工动作必须和策略动作处于**同一个动作空间**。否则，actor 的 BC 项会被拉向一个坐标系错位的目标，即使整个训练流程不会报错。

---

## 4. Replay Buffer 如何写入数据

`ChunkReplayBuffer` 采用**单写者、单读者**模型：

- rollout 主线程：唯一写者；
- learner 线程：唯一读者。

### 4.1 滞后一个 chunk

`add_chunk(rec)` 并不会立即生成 transition。

chunk `k` 首先存入 `_pending`，直到 chunk `k+1` 到达后，才通过：

```text
_emit_pair(k, k+1)
```

生成训练 transition。

原因在于 replay buffer 支持 stride 子采样。对于 `offset > 0` 的样本，其动作 / 奖励窗口可能横跨两个连续 chunk，因此必须同时持有 chunk `k` 和 chunk `k+1`。

因此：

- `total_added` 是**成批增长**的，而不是逐条增长；一次最多写入约 `chunk_len / stride` 条；
- episode 结束时通过 `_flush_terminal` 处理尚未配对的 pending chunk；
- termination 与 truncation 的处理方式不同，详见 4.3。

### 4.2 一条 transition 存什么

| 字段 | 含义 |
|---|---|
| `x` | RL 状态：`z_rl ++ 本体感知` |
| `action` | 实际执行的动作 chunk，形状为 `(C, d)` |
| `ref` | 当前状态对应的参考 chunk：来自 VLA，或被人工纠正替换 |
| `reward_disc` | `sum_j gamma^(j-1) * r_j` |
| `x_next` | `k` 个控制步之后的 RL 状态 |
| `ref_next` | `x_next` 对应的参考 chunk |
| `done` | 当前窗口内是否发生终止 |
| `actual_steps` | 当前窗口内**实际执行**的步数，`k ≤ C` |

`actual_steps` 是一个关键字段。

chunk 可能因为成功或截断而提前结束，因此 critic 在 bootstrap 时必须使用 `gamma^k`，而不是固定的 `gamma^C`。

未执行的尾部动作只用于补齐张量形状：通过重复最后一个动作将 tensor pad 到固定长度；真正的有效长度由 `actual_steps` 记录。

### 4.3 终止 vs 截断

- **终止（termination）**：bootstrap 被 mask 掉，因此 `x_next` 的具体值不再重要。
- **截断（truncation）**：bootstrap **不能**被 mask，因此必须提供真实的 post-episode 状态。

如果截断时拿不到真实的 post-episode 状态，就直接丢弃整条 transition。

不能简单地把 `x_next` 指回 `x`：这样会把 Bellman backup 变成自环，静默地将这些状态的价值钉在错误的位置，典型结果就是 value 学不起来。

### 4.4 丢弃 episode 的语义

操作员按下丢弃键，语义是：

> “这一整个 episode 的数据都不可信。”

典型情况包括：

- reset 没摆好；
- 成功 / 失败判定标错；
- 物体被意外碰倒；
- 其它导致整集数据失真的异常。

因此，`discard_episode()` 会回退**整个 episode**，而不只是尚未配对的 pending chunk。

当前实现有一个明确限制：回退写指针只在 buffer **尚未绕圈覆盖旧数据**时成立。

一旦 ring buffer 写满并发生 wrap-around，回退位置之后的槽位可能已经存放有效的历史数据，此时直接缩小 `size` 会误删旧样本。因此，这种情况当前选择保留数据并明确告警，而不是冒险执行错误回退。

在实际配置下，这一限制通常不会触发：buffer 容量约 20 万条，而单个 episode 通常只有约 `10²` 条 transition。

---

## 5. 并发模型

### 5.1 当前实现

当前采用**单进程、双线程**架构：

- **rollout = 主线程**
  - 持有 env、VLA 和遥操作设备；
  - 是 replay buffer 的唯一写者。

- **learner = 后台线程**
  - 持有 agent；
  - 按 UTD 节奏从 buffer 采样并执行梯度更新；
  - 定期将 actor 权重发布到 `ActorMirror`。

rollout **从不直接访问正在训练中的 actor 权重**。

它只会在 chunk 边界从 `ActorMirror` 拉取一份带版本号的参数快照。因此，一个 chunk 从规划到执行始终使用同一套自洽参数，不会出现 chunk 中途模型权重变化的问题。

### 5.2 这套并发解决了什么，又没有解决什么

| 机制 | 双线程 | 多进程 |
|---|---|---|
| GIL 争用（buffer gather vs MuJoCo / 图像预处理） | 存在 | 消除 |
| `.item()` 导致共享 CUDA 默认流同步 | 存在 | 消除（独立 CUDA context） |
| GPU 算力时间片竞争 | 存在 | **仍然存在** |

具体来说，`RLTAgent.update` 每个梯度步会调用约 3～4 次 `.item()`。每次调用都可能触发 CUDA 同步，抽干当前默认流。

与此同时，rollout 的 `plan_chunk` 本身包含：

- 2 次 VLM 前向；
- flow matching 采样。

如果 learner 恰好进行一轮梯度突发，rollout 的推理就会排在 learner 后面等待。

例如：

```text
utd = 5
stride = 2
```

时，一次 burst 可能对应约 25 个梯度步。在 30 Hz 真机控制下，这种停顿正是异步训练架构原本希望避免的问题。

当前已经做了以下缓解：

- `max_updates_per_burst`，默认 `32`  
  用于限制一次 learner burst 的规模。这里**按 transition 数截断，而不是按 update 数截断**，否则会破坏以下不变量：

  ```text
  总更新数 == utd × 总 transition 数
  ```

- `publish_every`，默认 `10`  
  没有必要每次梯度更新都 clone 整个 actor `state_dict`。`ActorMirror` 本身只会在 chunk 边界被 rollout 同步，因此降低发布频率可以避免大量无意义的参数复制。

- `sample()` 的 host gather 放在锁内，pinned memory 和 `non_blocking` GPU 拷贝放在锁外  
  尽量缩短 replay buffer 的临界区。

但仍然存在一个固有限制：

> 单进程意味着 rollout 和 learner 共享同一个 CUDA context。

只要 learner 吃满 GPU，VLA 前向仍然必须等待。

这正是多进程后端要解决的问题，见 5.4。

### 5.4 多进程后端（`concurrency.mode=processes`）

rollout 进程持 VLA + env + 遥操作，learner 进程持 replay buffer + actor/critic。
复用 `transport/services.proto` 已有的 `SendTransitions`（op 上行）和
`StreamParameters`（actor 权重下行），proto 不改，服务端也直接复用
`lerobot.rl.learner_service.LearnerService`——它本来就只是搬字节。

```bash
python -m lerobot.rlt.train_online --config_path examples/rlt/libero_online_processes.yaml
# 依赖：pip install -e ".[async]"
```

**上行的是 buffer 操作，不是算好的 transition。** buffer 那套逻辑（滞后一个 chunk、
终止 vs 截断、整集回退）足够微妙，两份实现必然会漂。所以 `ChunkReplayBuffer` 原样
搬到 learner 进程，网上跑的是方法调用本身：`start / chunk / end / discard`。gRPC 流
保序，rollout 又在开下一集之前发 `discard`，所以 learner 侧的回退看到的状态和单进程
版完全一致——`tests/rlt/test_distributed.py` 就是逐字段对比这两条路径。

线上全是 tensor 和基本类型的普通 dict，接收端用 `torch.load(weights_only=True)` 解，
不走 pickle。

**checkpoint 移到了 learner 侧**：它同时拥有 agent 和 buffer，所以不存在
`state_dict()` 撞上 optimizer 半步的窗口，线程版那套 `pause()` / `resume()` 在这里
根本不需要。

| | 线程 | 进程 |
|---|---|---|
| GIL 争用 | 有 | 消除 |
| `.item()` 抽干共享 CUDA 流 | 有 | 消除（各自 context） |
| GPU 算力时间片 | 有 | **仍然有** |
| 单独限制 learner 的 torch 线程数 | 做不到（全局设置会连 VLA 一起限） | 可以 |

最后一行是意外收获，但收益很大：actor-critic 是 2 层 MLP，torch 默认按核数开
intra-op 线程，同步开销远大于收益。本机实测 **48 线程 6.7 updates/s，1 线程
163 updates/s（24 倍）**，顺带也不会把核占满饿死 rollout。默认
`learner_torch_threads: 1`。

**线程版仍是默认，也是对照基线。** RL 上一旦不涨，"进程版有权重陈旧 bug" 和
"这个任务本来就难" 从曲线上分不开，所以先用线程版拿到可用结果，再切进程版提速。

### 5.3 已修复的竞态条件

| 问题 | 原表现 / 当前处理 |
|---|---|
| buffer 无锁 | `discard_episode()` 回退写指针时 learner 可能同时 `sample()`，采到即将被覆盖的数据；现已加锁 |
| `len()` 与 `sample()` 之间 buffer 被清空 | 可能触发 `torch.randint(0, 0)`；现在 `sample()` 会在锁内复检并返回 `None` |
| learner 异常被吞 | 线程死亡后训练永久空转，但日志仍输出陈旧均值；现在主循环通过 `raise_if_failed()` 重新抛出异常 |
| checkpoint 撕裂 | 保存时 learner 可能正处于 optimizer step；现在 checkpoint 前后成对调用 `pause()` / `resume()` |
| `metrics()` 并发遍历 | deque 在遍历过程中同时 append，可能触发 `"mutated during iteration"`；现在先复制快照再统计 |

---

## 6. 人在回路

### 6.1 算子按键

仿真和真机使用完全一致的算子键位：

| 键 | 作用 |
|---|---|
| `空格` | 切换人工接管状态（电平状态，不是单次边沿事件） |
| `s` | 标记本集成功——**唯一的奖励来源**，`r_T = 1` |
| `f` | 标记本集失败并结束，不计成功 |
| `r` | critical-phase 交接：将控制权从 base VLA 交给 RL policy |
| `←` / `退格` | 丢弃当前 episode，详见下方键位冲突说明 |
| `Esc` | 结束运行 |

键盘后端按以下顺序尝试，默认配置为：

```yaml
keyboard_backend: auto
```

1. **`termios`**
   - 要求 stdin 是真实终端；
   - **优先使用**；
   - 只有当前终端获得焦点时按键才生效。

   这对于真机运行尤其重要，因为它可以避免操作员在其它窗口输入文字时误触发控制指令。

2. **`pynput`**
   - 使用全局 X11 键盘钩子；
   - 当 stdin 是管道时，例如 IDE 控制台、`nohup`、`roslaunch`，通常只能使用这个后端；
   - 缺点是**全局捕获**。

   例如，操作员在另一个窗口输入字母 `s`，也可能被记录为“episode 成功”。

> **键位冲突**
>
> 键盘遥操作使用方向键控制机械臂，而默认的 `left` 同时又是“丢弃 episode”。
>
> 两个模块读取的是同一条全局按键流，因此无法通过调用顺序解决冲突，唯一可靠的方法是重新绑定按键。
>
> 所以，所有启用键盘遥操作的配置都应显式设置：
>
> ```yaml
> discard_key: backspace
> ```

### 6.2 遥操作设备

| 设备 | 通道 | 说明 |
|---|---|---|
| `keyboard` | 方向键控制 xy，Shift / 右 Shift 控制 z，`u/o`、`i/k`、`j/l` 控制 roll / pitch / yaw，Ctrl 控制夹爪 | `pynput` 模式下需要 DISPLAY |
| `gamepad` | 左摇杆控制 xy，右摇杆控制 z，旋转轴编号可配置 | 不同手柄的轴编号不同，使用前务必确认 |
| `spacemouse` | 原生 6-DoF | 操作体验最好，但需要额外硬件 |

三类设备走**完全相同的代码路径**。

`DeviceIntervention` 首先读取设备声明的：

```text
action_features["names"]
```

然后按照 env 声明的 `action_names` **按名称重新组装动作向量**。

设备没有提供的通道会自动补 `0`。因此：

- 3-DoF 键盘；
- 6-DoF SpaceMouse；

都可以安全驱动 7 维 LIBERO 动作空间，而不会因为维度顺序不同导致通道错位。

有两个转换必须特别注意。

#### 夹爪编码

teleop 侧：

```text
0 = 闭合
1 = 保持
2 = 张开
```

robosuite 侧：

```text
-1 = 张开
+1 = 闭合
```

其中，`保持`不能简单映射为 `0`。

正确语义是：

> 重复上一次夹爪命令。

如果直接发送 `0`，夹爪可能在抓取过程中自行松开。

#### 接管状态的读取

人工接管开关由：

```text
KeyboardEventListener
```

统一管理，而**不通过**设备自己的：

```text
get_teleop_events()
```

读取。

原因是 `KeyboardEndEffectorTeleop.get_action()` 会清空 `get_teleop_events()` 依赖的 `current_pressed`。

因此，两者无论谁先调用，另一个都有可能读到空状态。将接管状态独立到 `KeyboardEventListener` 后，可以彻底避免这一冲突。

### 6.3 人工纠正如何进入训练

发生人工接管时，人工动作会同时替换：

1. **真正执行的动作**；
2. replay buffer 中保存的**参考 chunk**。

这一设计对应论文 Sec. V 的 “Rollout”。

这是一个关键细节：actor 的 BC 项会因此拟合**人的纠正动作**，而不是继续拟合 VLA 原本那次失败的参考动作。

换句话说，人工接管不仅改变当前 rollout，也会直接改变后续 actor 的监督目标。

---

## 7. 常用命令

```bash
# 阶段 0：采集演示数据
# 可选；如果官方 LIBERO 数据集已经足够，可直接跳过
lerobot-rlt-collect-libero --config_path examples/rlt/collect_libero.yaml

# 阶段 1：训练 RL token
lerobot-rlt-train-token --config_path examples/rlt/libero_rl_token.yaml

# 阶段 1：表征质量诊断
lerobot-rlt-eval-token --config_path examples/rlt/libero_rl_token.yaml

# 阶段 2 基线：
# 在 RL 训练之前，先测量冻结 SmolVLA 的成功率
lerobot-eval --config_path examples/rlt/eval_libero_smolvla.yaml

# 阶段 2：在线 RL

# 无 GPU / 无机器人的管道冒烟测试
python -m lerobot.rlt.train_online \
  --config_path examples/rlt/mock_online.yaml

# LIBERO 仿真，无人在回路
python -m lerobot.rlt.train_online \
  --config_path examples/rlt/libero_online.yaml

# LIBERO 仿真 + 人工示教纠正
python -m lerobot.rlt.train_online \
  --config_path examples/rlt/libero_online_teleop.yaml

# Piper 真机在线 RL
python -m lerobot.rlt.train_online \
  --config_path examples/rlt/piper_online.yaml

# 真机上机前必做：HIL 链路体检
python -m lerobot.rlt.teleop_check \
  --config_path examples/rlt/teleop_check.yaml \
  --dry_run=true

# 真机上机前必做：领臂重力补偿调参
python -m lerobot.rlt.gravity_probe --port can1
```

### 配置清单

| 文件 | 用途 |
|---|---|
| `collect_libero.yaml` | LIBERO 示教数据采集 |
| `rl_token.yaml` / `libero_rl_token.yaml` | 阶段 1：RL token 训练 |
| `eval_libero_smolvla.yaml` | 阶段 2 前的冻结 VLA 基线评估 |
| `mock_online.yaml` | 无 GPU / 无机器人的完整管道冒烟测试 |
| `libero_online.yaml` | LIBERO 仿真在线 RL，无人在回路 |
| `libero_online_teleop.yaml` | LIBERO 仿真在线 RL + 键盘 / 手柄示教纠正 |
| `libero_online_processes.yaml` | 同上但 learner 跑在独立进程（gRPC），见 5.4 |
| `piper_online.yaml` | Piper 真机在线 RL + 领臂示教 |
| `teleop_check.yaml` | 真机 HIL 链路体检 |

---

## 8. 排障

### 「mode=processes 时进程无限自我复制 / 一开就卡死」

自己写的驱动脚本没有 `if __name__ == "__main__":` 保护。多进程后端用 `spawn`
（CUDA context 不能安全 fork），子进程会重新导入主模块，没有保护就会再次执行建后端
的代码，递归拉起进程。

仓库自带的入口（`python -m lerobot.rlt.train_online`、`lerobot-rlt-train-online`）
都有这个保护，不受影响。

---

### 「mode=processes 连不上 learner，每个 RPC 都是 UNAVAILABLE，报的端口还不是我配的」

gRPC 默认会走 `http_proxy` / `https_proxy`。客户端已显式关掉
`grpc.enable_http_proxy`，如果仍然出现，先确认没有别的中间件在劫持本机回环。

---

### 「mode=processes 时 learner 只做了几十次更新就再也不动了」

先看 learner 进程自己的 stderr：它的异常不会出现在 rollout 的日志里，rollout 只会
报 `RLT learner process exited with code N`。

---


### 「按键完全没反应」

如果 stdin 不是终端，同时系统中又没有 DISPLAY，那么 `termios` 和 `pynput` 两个后端都无法正常启动。

日志中会输出明确告警。

解决方式：

- 在真实终端中运行；
- 或在 PyCharm / VSCode 的运行配置中启用终端模拟。

---

### 「方向键推机械臂的同时，把 episode 丢了」

说明 `discard_key` 仍然使用默认值：

```yaml
discard_key: left
```

键盘遥操作本身也使用方向键，因此两者发生冲突。

所有启用键盘遥操作的配置都应改为：

```yaml
discard_key: backspace
```

---

### 「训练在跑，但 loss 一直不动，buffer 也在涨」

过去，这通常意味着 learner 线程已经静默死亡，但主线程仍在继续 rollout，并输出旧的 metrics。

现在主循环会主动检测 learner 状态，并抛出：

```text
RuntimeError: RLT learner thread died
```

learner 中最初的异常会保存在 `__cause__` 上，因此直接查看 traceback 即可定位真正原因。

---

### 「成功率完全不涨」

建议按以下顺序排查。

1. **检查 RL token 表征是否可用**

   运行：

   ```bash
   lerobot-rlt-eval-token --config_path examples/rlt/libero_rl_token.yaml
   ```

   重点关注：

   - 重建质量；
   - 表征是否坍缩；
   - 关键状态是否线性可解码；
   - 时序结构；
   - 尤其是任务进度 `t/T` 的线性可解码性。

   在稀疏奖励设置下，critic 在很大程度上是在学习：

   > “当前状态距离成功还有多远？”

   如果 `z_rl` 中完全没有任务进度信息，value learning 会非常困难。

2. **确认冻结 VLA 本身不是 0% 成功率**

   使用：

   ```text
   eval_libero_smolvla.yaml
   ```

   先测量 RL 之前的 frozen VLA baseline。

   如果 base policy 本身完全无法完成任务，在线 RL 很难仅靠稀疏成功奖励将策略从零推起来。

3. **确认阶段 1 和阶段 2 使用完全相同的 processors**

   尤其检查：

   ```text
   policy_preprocessor.json
   policy_postprocessor.json
   ```

   是否与对应的 `rl_token.pt` 来自同一个阶段 1 输出目录。

---

### 「真机上动作方向不对 / 幅度离谱」

优先怀疑**归一化不匹配**。

可以在上 RL 训练之前运行：

```bash
python -m lerobot.rlt.teleop_check \
  --config_path examples/rlt/teleop_check.yaml \
  --rl_token=outputs/rl_token
```

该模式会执行归一化 / 反归一化的 round-trip 自检。

这类问题应尽量在 `teleop_check.py` 阶段解决，而不是等到在线 RL 过程中再 debug。

---

### 「仿真里人工示教的动作看起来太猛」

调小：

```yaml
teleop.position_scale
teleop.rotation_scale
```

设备输出和 LIBERO 动作空间本身都位于 `[-1, 1]`，因此这两个参数本质上只是人工遥操作的增益系数。

---

### 「数据集读不回来 / parquet footer 报错」

通常意味着采集过程被 `kill -9` 等方式强制中断，导致 parquet footer 没有正常写入。

可以使用：

```python
repair_dataset_consistency(root)
```

进行修复。

`collect_libero.py` 的续采路径会自动调用这一修复逻辑。

正常结束采集时，脚本会调用：

```python
dataset.finalize()
```

确保 footer 和数据集元信息完整写入。
