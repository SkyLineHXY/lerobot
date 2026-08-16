# RLT：基于 RL token 的 VLA 在线强化学习

在冻结的 SmolVLA 之上加一个轻量 actor-critic，用在线 RL + 人工示教纠正把策略推到
可用水平。模型本身（RL token、chunk actor-critic、controller、配置）
`lerobot/policies/rlt/`；这个包是围绕它的一切：训练入口、chunk 级 replay buffer、
环境、异步 learner、人在回路。

---

## 1. 两个阶段

```
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

**阶段 1 的产出目录是一份契约**：`rl_token.pt` 必须和当时的 processors 放在一起。
阶段 2 和真机都要用完全相同的统计量做归一化/反归一化，否则 z_rl 和参考 chunk 会
悄悄漂移——不报错，只是训不出来。

---

## 2. 代码架构

```
rlt/
├── train_rl_token.py     RLtoken 阶段 1 表征学习训练入口
├── eval_rl_token.py      阶段 1 表征质量诊断（重建/表征坍缩/线性可解码/时序结构）
├── train_online.py       阶段 2 在线强化学习入口
├── rollout.py            RolloutWorker：规划一个 chunk、执行 ChunkRecord
├── learner.py            LearnerThread + ActorMirror：异步梯度步与权重发布
├── replay_buffer.py      ChunkReplayBuffer：chunk 级 transition，stride 子采样
├── collect_libero.py     用键盘/手柄在 LIBERO 里示教，采集成 LeRobotDataset
├── teleop_check.py       真机 HIL 链路体检（不训练、不加载 VLA）
├── gravity_probe.py      领臂重力补偿调参
├── envs/
│   ├── base.py           ChunkEnv 协议 + 归一化边界（ActionNormalizer）
│   ├── __init__.py       make_chunk_env() 工厂
│   ├── mock.py           抽象 6-DoF 任务，只验管道，不是算法信号
│   ├── libero.py         LIBERO 仿真：上真机前的完整闭环验证
│   └── piper.py          真实 Piper 机械臂
└── teleop/
    ├── keys.py           键盘后端（termios / pynput）+ 算子按键状态机
    ├── base.py           InterventionResult / InterventionManager 契约
    ├── device.py         DeviceIntervention：键盘/手柄/SpaceMouse
    └── piper_leader.py   PiperLeaderIntervention：领臂遥操作
```

---

## 3. 数据流

```
env.reset() ─► obs
   │
   ├─ obs_to_batch()          原始观测 ─► stage-1 preprocessor ─► 模型 batch
   │                          （LIBERO 还要先过 LiberoProcessorStep 摊平嵌套状态）
   ▼
controller.plan_chunk(batch)
   ├─ SmolVLA 前缀前向 ─► 选 token ─► RL token ─► z_rl
   ├─ x = concat(z_rl, 本体感知)
   ├─ ref_full = flow matching 采样的 VLA 参考 chunk
   └─ action_chunk = actor(x, ref[:C])        ← 零初始化残差，第 0 步与 base VLA 逐位相同
   ▼
逐步执行 C 步：env.step(action[j]) ─► postprocessor 反归一化 ─► 物理命令
   ▼
ChunkRecord ─► buffer.add_chunk()
```

**归一化边界只有一处**，两个方向：

- 出：`postprocessor` 把归一化动作还原成物理量（关节弧度 / LIBERO 的 [-1,1] delta EE）。
- 入：`ActionNormalizer`（`envs/base.py`）把人工遥操作的物理命令折回归一化空间。
  人的动作要和策略的动作住在同一个空间里，否则 actor 的 BC 项会被拉向一个错位的目标。

---

## 4. Replay buffer 是怎么写进去的

单写者（rollout 即主线程），单读者（learner 线程）。

### 4.1 滞后一个 chunk

`add_chunk(rec)` 并不立即产出 transition。chunk k 先存进 `_pending`，等 chunk k+1
到达才由 `_emit_pair(k, k+1)` 产出。原因是 stride 子采样：offset > 0 的动作/奖励
窗口横跨两个连续 chunk。所以：

- `total_added` 是**成批跳动**的，一次最多 `chunk_len / stride` 行；
- episode 结束走 `_flush_terminal`，终止和截断的处理不同（见 4.3）。

### 4.2 一条 transition 存什么

| 字段 | 含义 |
|---|---|
| `x` | RL 状态 = z_rl ++ 本体感知 |
| `action` | 实际执行的动作 chunk (C, d) |
| `ref` | 该状态下的参考 chunk（VLA 的，或被人工纠正替换掉的） |
| `reward_disc` | `sum_j gamma^(j-1) * r_j` |
| `x_next` | k 个控制步之后的 RL 状态 |
| `ref_next` | `x_next` 处的参考 chunk |
| `done` | 窗口内是否终止 |
| `actual_steps` | 窗口内**真正执行**的步数 k ≤ C |

`actual_steps` 是关键：chunk 会因为成功或截断提前结束，critic 必须用 `gamma^k`
而不是 `gamma^C` 做 bootstrap。未执行的尾部只是重复最后一个动作把张量形状补齐，
真实长度由 `actual_steps` 记着。

### 4.3 终止 vs 截断

- **终止**：bootstrap 被 mask 掉，`x_next` 无所谓。
- **截断**：bootstrap **不**被 mask，所以必须要有真实的 post-episode 状态。拿不到
  就整条丢弃——把 `x_next` 指回 `x` 会让 backup 变成自环，悄悄把这些状态的价值钉死在 0。

### 4.4 丢弃 episode 的语义

操作员按丢弃键的意思是"这一集是垃圾"（复位没摆好、判定标错、物体被碰倒），所以
`discard_episode()` 回退整集，而不只是未配对的那个 pending chunk。

限制：回退写指针只在 buffer 从未绕圈时成立。一旦写满，回退位置之后的槽位放的是
有效的旧数据，缩小 `size` 会删错东西。这种情况下保留并明说——真实运行（20 万容量、
每集 ~10² 条）到不了那里。

---

## 5. 并发模型

### 5.1 现状

一个进程，两个线程：

- **rollout = 主线程**：持有 env、VLA、遥操作设备；是 buffer 的唯一写者。
- **learner = 后台线程**：持有 agent；按 UTD 节奏采样并做梯度步，把 actor 权重
  发布出去。rollout 从不直接碰训练权重，它只在 chunk 边界从 `ActorMirror` 拉一份
  版本化快照，所以一个 chunk 一定是用一套自洽的参数规划出来的。

### 5.2 这套并发买到了什么，没买到什么

| 机制 | 线程 | 多进程 |
|---|---|---|
| GIL 争用（buffer gather vs MuJoCo/图像预处理） | 存在 | 消除 |
| `.item()` 抽干共享的 CUDA 默认流 | 存在 | 消除（各自独立 context） |
| GPU 算力时间片 | 存在 | **仍然存在** |

具体来说：`RLTAgent.update` 每个梯度步取 3~4 次 `.item()`，每次都把默认流抽干；
rollout 的 `plan_chunk`（2 次 VLM 前向 + flow matching）排在 learner 的梯度突发
后面。`utd=5` + stride-2 时一次突发约 25 个梯度步，30 Hz 真机上这正是异步设计
本想消除的停顿。

已做的缓解：

- `max_updates_per_burst`（默认 32）截断单次突发。**按 transition 数截断**，
  不是按更新数——否则会破坏「总更新数 == utd × 总 transition 数」这个不变量。
- `publish_every` 默认 10。每个梯度步 clone 整个 actor state_dict 是纯浪费，
  mirror 本来就只在 chunk 边界同步。
- `sample()` 锁内做 host 端 gather，锁外做 pinned + `non_blocking` 拷贝。

仍然存在的固有限制：单进程共享 CUDA context，learner 吃满 GPU 时 VLA 前向照样等。
多进程 + gRPC 版本正在计划中（复用 `transport/services.proto` 已有的
`StreamParameters` / `SendTransitions`）。

### 5.3 已修掉的竞态

| 问题 | 表现 |
|---|---|
| buffer 无锁 | `discard_episode()` 回退写指针时 learner 正在 `sample()`，采到马上被覆写的行 |
| `len()` 与 `sample()` 之间被清空 | `torch.randint(0, 0)` 直接抛错；现在 `sample()` 锁内复检并返回 `None` |
| learner 异常被吞 | 线程死掉，训练**永久空转**且日志照打陈旧均值；现在主循环 `raise_if_failed()` 抛出 |
| checkpoint 撕裂 | 保存时 learner 正在 optimizer step；现在 `pause()` / `resume()` 成对调用 |
| `metrics()` 并发遍历 | deque 被同时 append，抛 "mutated during iteration"；现在先快照 |

---

## 6. 人在回路

### 6.1 算子键（仿真与真机完全一致）

| 键 | 作用 |
|---|---|
| `空格` | 切换接管（电平，不是边沿） |
| `s` | 本集成功——**唯一的奖励来源**（r_T = 1） |
| `f` | 本集失败并结束（不计成功） |
| `r` | critical-phase 交接：把控制权从 base VLA 交给 RL policy |
| `←` / `退格` | 丢弃本集（见下方冲突说明） |
| `Esc` | 结束运行 |

两个键盘后端，按序尝试（`keyboard_backend: auto`）：

1. `termios` —— 需要 stdin 是真终端。**首选**，因为只有该终端获得焦点时按键才生效，
   这是真机上安全的行为。
2. `pynput` —— 全局 X11 钩子，stdin 是管道时（IDE 控制台、nohup、roslaunch）唯一
   能用的。代价是**全局捕获**：在别的窗口里敲 `s` 也会被记成"成功"。

> **键位冲突**：键盘遥操作用方向键推机械臂，与默认的 `left` = 丢弃冲突。两个读者
> 看的是同一条全局按键流，唯一的解法是改绑。所以带键盘遥操作的配置里都写
> `discard_key: backspace`。

### 6.2 遥操作设备

| 设备 | 通道 | 说明 |
|---|---|---|
| `keyboard` | 方向键 xy，Shift/右Shift z，`u/o` `i/k` `j/l` roll/pitch/yaw，Ctrl 夹爪 | 需要 DISPLAY（pynput） |
| `gamepad` | 左摇杆 xy，右摇杆 z，旋转轴号可配 | 轴编号各家手柄不同，务必先确认 |
| `spacemouse` | 原生 6-DoF | 最顺手，但要额外硬件 |

三种设备走**同一条代码路径**：`DeviceIntervention` 读设备的
`action_features["names"]`，按**名字**装配成 env 声明的 `action_names` 顺序，设备
没有的通道补 0。所以 3-DoF 键盘和 6-DoF SpaceMouse 都能驱动 7 维的 LIBERO，不会错位。

两个必须知道的转换：

- **夹爪编码**：teleop 侧是 `0=闭合 / 1=保持 / 2=张开`，robosuite 侧是连续量
  `-1=张开 / +1=闭合`。`保持`必须重复上一次命令，发 0 会让夹爪在抓取途中自己松开。
- **接管开关走 `KeyboardEventListener`，不走设备的 `get_teleop_events()`**。
  `KeyboardEndEffectorTeleop.get_action()` 会清掉 `get_teleop_events()` 要读的
  `current_pressed`，两者无论谁先调都有一个读到空。

### 6.3 人工纠正进入训练的方式

被接管的动作同时替换**执行动作**和 buffer 里存的**参考 chunk**（论文 Sec. V
"Rollout"）。这是关键：actor 的 BC 项因此拉向人的纠正，而不是拉向 VLA 那次失败的尝试。

---

## 7. 常用命令

```bash
# 阶段 0：采演示数据（可选，官方 libero 数据集够用就跳过）
lerobot-rlt-collect-libero --config_path examples/rlt/collect_libero.yaml

# 阶段 1：训 RL token
lerobot-rlt-train-token --config_path examples/rlt/libero_rl_token.yaml
lerobot-rlt-eval-token  --config_path examples/rlt/libero_rl_token.yaml   # 表征质量诊断

# 阶段 2 基线：RL 之前先量一下冻结 VLA 的成功率
lerobot-eval --config_path examples/rlt/eval_libero_smolvla.yaml

# 阶段 2：在线 RL
python -m lerobot.rlt.train_online --config_path examples/rlt/mock_online.yaml          # 冒烟
python -m lerobot.rlt.train_online --config_path examples/rlt/libero_online.yaml        # 仿真，无人
python -m lerobot.rlt.train_online --config_path examples/rlt/libero_online_teleop.yaml # 仿真 + 示教
python -m lerobot.rlt.train_online --config_path examples/rlt/piper_online.yaml         # 真机

# 真机上机前必做
python -m lerobot.rlt.teleop_check --config_path examples/rlt/teleop_check.yaml --dry_run=true
python -m lerobot.rlt.gravity_probe --port can1
```

### 配置清单

| 文件 | 用途 |
|---|---|
| `collect_libero.yaml` | LIBERO 示教采集 |
| `rl_token.yaml` / `libero_rl_token.yaml` | 阶段 1 |
| `eval_libero_smolvla.yaml` | 阶段 2 前的冻结 VLA 基线 |
| `mock_online.yaml` | 无 GPU/机器人的管道冒烟 |
| `libero_online.yaml` | 仿真在线 RL，无人在回路 |
| `libero_online_teleop.yaml` | 仿真在线 RL + 键盘/手柄示教纠正 |
| `piper_online.yaml` | 真机在线 RL + 领臂示教 |
| `teleop_check.yaml` | 真机 HIL 链路体检 |

---

## 8. 排障

**「按键完全没反应」**
stdin 不是终端且没有 DISPLAY，两个后端都起不来——日志里会有明确告警。在真实终端里
跑，或在 PyCharm/VSCode 的运行配置里打开终端模拟。

**「方向键推机械臂的同时把 episode 丢了」**
`discard_key` 还是默认的 `left`。带键盘遥操作的配置要改成 `backspace`。

**「训练在跑但 loss 一直不动、buffer 也在涨」**
以前是 learner 线程静默死亡的症状。现在会在主循环抛 `RuntimeError: RLT learner
thread died`，原始异常挂在 `__cause__` 上。

**「成功率完全不涨」**
按顺序排查：① `lerobot-rlt-eval-token` 看 z_rl 是不是可用表征（尤其是进度
`t/T` 的线性可解码性——稀疏奖励下 critic 本质在学"离成功还有多远"）；
② 用 `eval_libero_smolvla.yaml` 确认冻结 VLA 的基线不是 0；③ 确认阶段 1 和阶段 2
用的是同一份 processors。

**「真机上动作方向不对 / 幅度离谱」**
归一化不匹配。`teleop_check.py` 加 `--rl_token=outputs/rl_token` 会做归一化往返
自检，这比在 RL 训练里debug 便宜得多。

**「仿真里人工示教的动作看起来太猛」**
调小 `teleop.position_scale` / `rotation_scale`。设备输出和 LIBERO 动作空间都是
[-1, 1]，这两个参数纯粹是增益。

**「数据集读不回来 / parquet footer 报错」**
采集被 kill -9 掐断了。`repair_dataset_consistency(root)` 能修（`collect_libero.py`
的续采路径会自动调），正常结束时脚本会调 `dataset.finalize()` 写 footer。
