# π0（PI0）模型架构深度分析

> 对应源码：`src/lerobot/policies/pi0/`
> 上游参考实现：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

---

## 目录

1. [整体架构概述](#1-整体架构概述)
2. [核心组件拆解](#2-核心组件拆解)
   - [PaliGemma 视觉语言模型（前缀流）](#21-paligemma-视觉语言模型前缀流)
   - [Action Expert Gemma（后缀流）](#22-action-expert-gemma后缀流)
   - [双流联合注意力机制](#23-双流联合注意力机制)
3. [关键配置参数详解](#3-关键配置参数详解)
   - [模型结构参数](#31-模型结构参数)
   - [训练优化参数](#32-训练优化参数)
   - [微调控制参数](#33-微调控制参数)
   - [Flow Matching 参数](#34-flow-matching-参数)
4. [注意力 Mask 深度分析](#4-注意力-mask-深度分析)
   - [Mask 的数据结构](#41-mask-的数据结构)
   - [前缀流（Prefix）的 Mask 设计](#42-前缀流prefix的-mask-设计)
   - [后缀流（Suffix）的 Mask 设计](#43-后缀流suffix的-mask-设计)
   - [完整的联合 Mask 构造](#44-完整的联合-mask-构造)
   - [推理阶段的 Mask 处理](#45-推理阶段的-mask-处理)
5. [训练流程：Flow Matching](#5-训练流程flow-matching)
6. [推理流程：ODE 求解去噪](#6-推理流程ode-求解去噪)
7. [数据预处理管道](#7-数据预处理管道)
8. [梯度检查点（Gradient Checkpointing）机制](#8-梯度检查点gradient-checkpointing机制)
9. [模型架构总结图](#9-模型架构总结图)

---

## 1. 整体架构概述

π0 是一个基于 **Flow Matching** 的视觉-语言-动作（VLA）策略模型，核心设计思路是：

- **前缀流（Prefix Stream）**：由 PaliGemma（SigLIP 视觉编码器 + Gemma 语言模型）组成，负责编码图像和语言指令，产生上下文感知的 token 嵌入序列。
- **后缀流（Suffix Stream）**：由一个独立的 Action Expert Gemma 组成，负责接受机器人状态、噪声动作序列及扩散时间步，输出去噪后的速度场预测。
- **联合注意力（Joint Attention）**：两个 Transformer 共享同一个注意力层，前缀 token 和后缀 token 被拼接在一起做跨流注意力，实现视觉/语言上下文对动作生成的条件控制。

```
输入图像 ─→ SigLIP 视觉塔 ─→ 图像 token 嵌入 ─┐
语言指令 ─→ Gemma 嵌入层  ─→ 语言 token 嵌入 ─┤
                                                 ├─→ 【前缀 Embedding 序列】─┐
机器人状态 ─→ state_proj ──────────────────────┐  │                         │
噪声动作   ─→ action_in_proj ──┐                │  │  联合 Transformer 注意力│
扩散时间步 ─→ 正弦位置编码    ─┤ action_time_mlp│  │  （18层 Gemma layers）  │
                                └────────────────┘  │                         │
                                ─→ 【后缀 Embedding 序列】──────────────────────┘
                                                              │
                                              后缀 token 输出（取最后 chunk_size 个）
                                                              │
                                              action_out_proj ─→ 速度场预测 v_t
```

---

## 2. 核心组件拆解

### 2.1 PaliGemma 视觉语言模型（前缀流）

**类**：`PaliGemmaWithExpertModel.paligemma`（`PaliGemmaForConditionalGenerationWithPiGemma`）

#### 视觉编码器（SigLIP）
- 输入：图像张量，形状 `[B, C, H, W]`，归一化到 `[-1, 1]`
- 输出：视觉 token 嵌入，形状 `[B, N_img, hidden_size]`
  - 对于 `224×224` 分辨率，SigLIP 产生 `256` 个图像 patch token
- 特殊处理：`embed_image` 方法将 pooler_output 乘以 `sqrt(hidden_size)` 进行缩放，与 PaliGemma 规范一致
- 精度：始终保持 `float32`（即使整体模型使用 `bfloat16`），避免精度损失

#### 语言编码器（Gemma 嵌入层）
- 输入：tokenizer 产出的 token id，形状 `[B, T_lang]`
- 输出：语言 token 嵌入 × `sqrt(lang_emb_dim)`（标准 Gemma 缩放）
- tokenizer：使用 `google/paligemma-3b-pt-224`，`max_length=48`，右侧 padding

#### Gemma 变体规格

| 变体 | 隐藏维度 | 层数 | MLP 维度 | 注意力头数 | KV 头数 | 头维度 |
|------|---------|------|---------|-----------|--------|--------|
| `gemma_300m` | 1024 | 18 | 4096 | 8 | 1 | 256 |
| `gemma_2b` | 2048 | 18 | 16384 | 8 | 1 | 256 |

默认配置：paligemma 使用 `gemma_2b`，action expert 使用 `gemma_300m`。

### 2.2 Action Expert Gemma（后缀流）

**类**：`PaliGemmaWithExpertModel.gemma_expert`（`PiGemmaForCausalLM`）

Action Expert 的 token 嵌入层被置为 `None`（`self.gemma_expert.model.embed_tokens = None`），因为动作序列不经过 token 化，而是通过专用投影层生成嵌入。

后缀流嵌入由三部分拼接而成：

| 组件 | 投影层 | 形状 | 说明 |
|------|--------|------|------|
| 机器人状态 | `state_proj`: `max_state_dim → width` | `[B, 1, width]` | 单个 token 代表当前机器人关节状态 |
| 噪声动作 + 时间步 | `action_in_proj` + `action_time_mlp_in/out` | `[B, chunk_size, width]` | 每个动作步骤对应一个 token |

**时间步编码**：使用正弦-余弦位置编码，频率范围由 `min_period=4e-3` 到 `max_period=4.0` 对数均匀分布，编码维度等于 action expert 的 `width`。

**动作-时间融合 MLP**：
```
[action_emb; time_emb]  (cat along last dim, dim=2×width)
    → action_time_mlp_in (2×width → width)
    → SiLU 激活
    → action_time_mlp_out (width → width)
```

### 2.3 双流联合注意力机制

在训练阶段（`inputs_embeds=[prefix_embs, suffix_embs]`），两个模型的 token 序列被拼接，**共享同一个注意力矩阵**进行计算，实现跨流信息融合：

```python
# compute_layer_complete 中：
query_states = torch.cat([q_paligemma, q_expert], dim=2)  # 沿 seq_len 维拼接
key_states   = torch.cat([k_paligemma, k_expert], dim=2)
value_states = torch.cat([v_paligemma, v_expert], dim=2)
# 单次 eager_attention_forward 计算所有 token 的注意力
att_output, _ = eager_attention_forward(...)
# 然后按各自序列长度切分，分别通过各自的 o_proj、MLP、残差
```

每一层的输出被切分回前缀部分和后缀部分，各自通过：
1. `o_proj`（输出投影）
2. 第一个残差连接（带 gated residual）
3. `post_attention_layernorm`
4. MLP（GeGLU 激活）
5. 第二个残差连接

---

## 3. 关键配置参数详解

### 3.1 模型结构参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `paligemma_variant` | `"gemma_2b"` | 前缀流（VLM）的 Gemma 变体，决定视觉语言主干的容量 |
| `action_expert_variant` | `"gemma_300m"` | 后缀流（动作专家）的 Gemma 变体，容量显著小于主干 |
| `dtype` | `"float32"` | 模型参数精度；`bfloat16` 可减少约 50% 显存，但 LayerNorm 和视觉塔始终保持 float32 |
| `max_state_dim` | `32` | 机器人状态向量填充到的最大维度，短于此值的向量用零补齐 |
| `max_action_dim` | `32` | 动作向量填充到的最大维度，输出时截取实际维度 |
| `chunk_size` | `50` | 单次前向传播预测的动作步骤数（又称 action horizon） |
| `n_action_steps` | `50` | 每次推理实际执行的动作步骤数（≤ chunk_size） |
| `image_resolution` | `(224, 224)` | 图像输入分辨率，PaliGemma 要求正方形 |
| `empty_cameras` | `0` | 额外添加的空摄像头数量（全零图像，mask=0），用于兼容不同机器人配置 |
| `tokenizer_max_length` | `48` | 语言指令 token 序列的最大长度（padding 到此长度） |

### 3.2 训练优化参数

#### `gradient_checkpointing: bool = False`

**作用**：以重计算换显存，在反向传播时不保存中间激活值，而是在需要时重新计算。

**启用后的行为**（`gradient_checkpointing_enable()`）：
- `paligemma.model.language_model.gradient_checkpointing = True`
- `paligemma.model.vision_tower.gradient_checkpointing = True`
- `gemma_expert.model.gradient_checkpointing = True`

**在联合注意力层的实现**：每一个 `compute_layer_complete`（包含一完整的 attention + MLP + 残差的层计算）被包装为 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`，最终的 `compute_final_norms` 也同样被 checkpoint 包装。

**显存收益**：对于 18 层的 Gemma，开启后可节省大约 60-70% 的激活显存，代价是约 20-30% 的计算时间增加。

**注意**：此参数仅在 `self.training == True` 时生效，推理阶段不会触发 checkpoint。

---

#### `compile_model: bool = False`

**作用**：使用 `torch.compile` 对模型进行 JIT 编译优化，可提升推理和训练吞吐量。

**编译范围**：同时编译 `sample_actions`（推理路径）和 `forward`（训练路径）。

**`compile_mode: str = "max-autotune"`**：
- `"default"`：标准编译，平衡编译时间和性能
- `"reduce-overhead"`：优化小 batch/重复调用场景
- `"max-autotune"`：最大自动调优，编译时间长但运行性能最优（适合大规模训练）

**注意**：`compile_model` 还会设置 `torch.set_float32_matmul_precision("high")`，允许使用 TF32 加速矩阵乘法。

---

#### `device: str | None = None`

模型部署设备。`None` 表示自动检测。`get_safe_dtype` 函数会针对不同设备做类型安全检查：
- CPU：不支持 `bfloat16`，自动降级为 `float32`
- MPS：不支持 `float64`，自动降级为 `float32`

### 3.3 微调控制参数

#### `freeze_vision_encoder: bool = False`

**作用**：冻结 SigLIP 视觉塔的所有参数，训练时只更新语言和动作相关参数。

**实现**：
```python
self.paligemma.model.vision_tower.eval()  # 始终保持 eval 模式（不更新 BN 等）
for param in self.paligemma.model.vision_tower.parameters():
    param.requires_grad = False
```

**适用场景**：
- 数据量不足以微调视觉编码器时
- 希望保留预训练视觉特征，仅适配语言和动作分支

---

#### `train_expert_only: bool = False`

**作用**：冻结整个 PaliGemma（VLM 主干）的所有参数，只训练 Action Expert Gemma 以及相关投影层（`state_proj`、`action_in_proj`、`action_out_proj`、`action_time_mlp_*`）。

**实现**：
```python
self.paligemma.eval()  # 整个 VLM 保持 eval 模式
for param in self.paligemma.parameters():
    param.requires_grad = False
```

**适用场景**：
- 将预训练好的 VLM 视为固定特征提取器
- 仅需适配新的机器人平台，动作空间发生变化

**与 `freeze_vision_encoder` 的区别**：
- `freeze_vision_encoder`：只冻结视觉塔，语言模型仍可训练
- `train_expert_only`：冻结整个 PaliGemma（包括语言模型），只训练动作专家分支

### 3.4 Flow Matching 参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `num_inference_steps` | `10` | 推理时的去噪步数（ODE 离散化步数） |
| `time_sampling_beta_alpha` | `1.5` | 训练时间步采样的 Beta 分布 α 参数 |
| `time_sampling_beta_beta` | `1.0` | 训练时间步采样的 Beta 分布 β 参数 |
| `time_sampling_scale` | `0.999` | 时间步缩放因子，避免采样到精确的 0 或 1 |
| `time_sampling_offset` | `0.001` | 时间步偏移量 |
| `min_period` | `4e-3` | 正弦位置编码的最小周期（对应最高频率） |
| `max_period` | `4.0` | 正弦位置编码的最大周期（对应最低频率） |

时间步采样：`t ~ Beta(1.5, 1.0) * 0.999 + 0.001`，倾向于在 `t≈1`（噪声端）附近多采样，符合 Flow Matching 的训练策略。

---

## 4. 注意力 Mask 深度分析

π0 的注意力 mask 设计是其最关键的架构特征，实现了"前缀双向注意力 + 后缀因果注意力 + 跨流单向注意力"的混合模式。

### 4.1 Mask 的数据结构

系统中使用两种 mask：

| 名称 | 类型 | 形状 | 含义 |
|------|------|------|------|
| `pad_masks` | `bool[B, N]` | 批次×序列长度 | `True` 表示有效 token，`False` 表示填充 token |
| `att_masks`（`mask_ar`） | `int/float[B, N]` | 批次×序列长度 | 控制注意力边界：`0` 表示与前一个 token 共享注意力范围，`1` 表示新的因果分组起点 |

### 4.2 前缀流（Prefix）的 Mask 设计

前缀序列的组成：

```
[img_token_1, ..., img_token_256, | img_token_1', ..., | lang_token_1, ..., lang_token_T]
      att_mask: [0, 0, ..., 0,           0, ..., 0,       0, 0, ..., 0]
```

**所有前缀 token 的 `att_masks` 均为 `0`**。

这意味着累计和（`cumsum`）在整个前缀范围内单调不减（全为 0 后全为 0），任意两个前缀 token 之间满足 `cumsum[i] <= cumsum[j]`，即**前缀 token 之间是完全双向注意力（Bidirectional Attention / Prefix-LM）**。

图像 token 和语言 token 可以互相看到彼此，这对多模态理解至关重要。

### 4.3 后缀流（Suffix）的 Mask 设计

后缀序列的组成：

```
[state_token, action_token_1, action_token_2, ..., action_token_49]
  att_mask: [      1,              1,               0,    ...,    0    ]
```

关键设计：
- `state_token`：`att_masks = 1`（新的因果分组起点）
- `action_token_1`（第一个动作 token）：`att_masks = 1`（新的因果分组起点）
- `action_token_2` 到 `action_token_49`：`att_masks = 0`（与前一个 token 共享注意力范围）

**累计和分析**：
```
state_token:     cumsum = 1
action_token_1:  cumsum = 2
action_token_2:  cumsum = 2  ← 与 action_token_1 相同！
action_token_3:  cumsum = 2  ← 与前两个相同
...
action_token_49: cumsum = 2
```

由于所有动作 token 的 `cumsum` 值相同（均为 2），根据 `cumsum[i] <= cumsum[j]` 的规则：
- **所有动作 token 之间可以互相注意（双向注意力）**
- `state_token`（cumsum=1）只能被 cumsum≥1 的 token 注意到（即全部后缀 token 都可以注意到 state）
- **但后缀 token 不能注意到前缀 token**（见下节）

这是 π0 的核心设计：**动作序列内部是全局双向注意力，不是自回归因果注意力**，这允许模型在预测每个动作步骤时考虑整个动作 chunk 的全局信息，更适合连续动作的去噪预测。

### 4.4 完整的联合 Mask 构造

```python
# make_att_2d_masks 的实现逻辑
cumsum = torch.cumsum(att_masks, dim=1)                    # [B, N_total]
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]    # [B, N_total, N_total]
pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
return att_2d_masks & pad_2d_masks
```

拼接后的完整序列（N_total = N_prefix + N_suffix）：

```
att_masks = [0, 0, ..., 0  | 1,  1,  0,  0, ..., 0]
             ←─ prefix ──→   ←────── suffix ────────→

cumsum    = [0, 0, ..., 0  | 1,  2,  2,  2, ..., 2]
```

完整注意力矩阵（行=query token，列=key token）：


```
              prefix tokens    | state  act_1  act_2 ... act_49
             ─────────────────┼──────────────────────────────
prefix  q   |  ✓(双向)        |   ✗     ✗     ✗  ...   ✗
state   q   |  ✓(单向：可看前缀) |  ✓     ✗     ✗  ...   ✗
act_1   q   |  ✓(单向：可看前缀) |  ✓     ✓     ✓  ...   ✓  ← act_1 可看所有 action token（因 cumsum 相同）
act_2   q   |  ✓(单向：可看前缀) |  ✓     ✓     ✓  ...   ✓
...
act_49  q   |  ✓(单向：可看前缀) |  ✓     ✓     ✓  ...   ✓
```

**小结：注意力模式**

| 关系 | 注意力类型 |
|------|-----------|
| 前缀 token 之间 | 全双向（Bidirectional） |
| 前缀 → 后缀 | 不可见（前缀不能看后缀） |
| 后缀 → 前缀 | 可见（后缀可以看前缀所有 token） |
| 后缀 token 之间 | 全双向（所有动作 token 互相可见） |
| state → action | state 不能看 action token |
| action → state | 所有动作 token 都可以看 state |

### 4.5 推理阶段的 Mask 处理

推理时采用 **KV Cache 加速**，分为两个阶段：

**阶段一：前缀预计算**
```python
# 只处理前缀流（suffix=None）
_, past_key_values = self.paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
```
前缀 KV 缓存 `past_key_values` 保存了所有图像和语言 token 的 key/value，后续每次去噪步骤可以直接复用，无需重新计算。

**阶段二：去噪步骤（`denoise_step`）**
```python
# 后缀 token 对前缀的注意力 mask：直接用 prefix_pad_masks 展开
prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
# 后缀 token 之间的 mask
suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
# 拼接：每个后缀 token 可以看所有前缀 token（用 KV cache），且后缀内双向
full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
```

位置 ID 从前缀偏移处开始：
```python
prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]  # 前缀有效 token 数
position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
```

**4D Mask 转换**：
```python
att_2d_masks_4d = att_2d_masks[:, None, :, :]          # 增加 head 维度
# 将 True/False 转换为加性偏置值（用于 softmax 前相加）
return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
# OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38（近似负无穷）
```

这是标准的加性 attention mask 做法：masked 位置加上约 `-∞` 值，经过 softmax 后概率趋近于 0。

---

## 5. 训练流程：Flow Matching

Flow Matching 是 π0 的动作生成范式，用于训练模型预测从噪声到动作的速度场。

### 训练前向传播步骤

```python
# 1. 采样噪声（标准正态）
noise = N(0, I)  # shape: [B, chunk_size, max_action_dim]

# 2. 采样时间步 t ~ Beta(1.5, 1.0) * 0.999 + 0.001，值域约 [0.001, 1.0]
t ∈ [0, 1]

# 3. 在直线轨迹上插值，生成含噪动作
x_t = t * noise + (1 - t) * actions

# 4. 训练目标：预测速度场 u_t（即 noise - action 方向）
u_t = noise - actions

# 5. 模型预测速度场 v_t
v_t = model(x_t, t, images, language, state)

# 6. MSE 损失
loss = MSE(u_t, v_t)
```

**物理直觉**：在 `t=1` 时，`x_t = noise`（纯噪声）；在 `t=0` 时，`x_t = actions`（真实动作）。训练目标是让模型学会"噪声→动作"方向上的速度场，推理时从 `t=1` 沿速度场积分到 `t=0` 即可得到预测动作。

---

## 6. 推理流程：ODE 求解去噪

推理时使用一阶欧拉法求解 ODE，从噪声 `x_1` 积分到动作 `x_0`：

```python
dt = -1.0 / num_steps  # 负方向步长（从 t=1 到 t=0）

x_t = noise  # 初始化为纯噪声
for step in range(num_steps):  # 默认 10 步
    time = 1.0 + step * dt   # t: 1.0 → 0.0
    v_t = model.denoise_step(x_t, time, ...)  # 预测速度场
    x_t = x_t + dt * v_t    # 欧拉积分
    
return x_t  # 预测的动作序列
```

`num_inference_steps=10` 在精度和速度之间取得平衡，更多步数可提升动作质量但增加推理延迟。

---

## 7. 数据预处理管道

**预处理流水线**（`make_pi0_pre_post_processors`）：

```
原始观测数据
    ↓ RenameObservationsProcessorStep    # 特征重命名（兼容预训练配置）
    ↓ AddBatchDimensionProcessorStep     # 单步推理时补充 batch 维度
    ↓ Pi0NewLineProcessor               # 在语言指令末尾添加换行符（PaliGemma tokenizer 要求）
    ↓ TokenizerProcessorStep            # PaliGemma tokenizer，max_length=48，右侧 padding
    ↓ DeviceProcessorStep               # 将 tensor 移到模型所在设备
    ↓ NormalizerProcessorStep           # 状态/动作归一化（MEAN_STD），图像归一化（IDENTITY，由模型内部处理）
```

**图像预处理**（`_preprocess_images`）：
1. 检查格式（channels-first or channels-last）
2. 使用 `resize_with_pad_torch` 等比缩放并填充黑边到 `224×224`
3. 归一化：`[0, 1] → [-1, 1]`（SigLIP 要求）

**后处理流水线**：
```
模型输出动作（归一化空间）
    ↓ UnnormalizerProcessorStep         # 反归一化到原始动作空间
    ↓ DeviceProcessorStep(cpu)          # 移回 CPU
```

---

## 8. 梯度检查点（Gradient Checkpointing）机制

π0 中 gradient checkpointing 的实现细节：

```python
# PI0Pytorch.gradient_checkpointing_enable() 被调用后：

# 对于联合注意力层（training 阶段）：
for layer_idx in range(num_layers):  # 18 层
    if use_gradient_checkpointing:
        inputs_embeds = torch.utils.checkpoint.checkpoint(
            compute_layer_complete,   # 一完整层的计算（attention + MLP + 残差）
            layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond,
            use_reentrant=False,      # 推荐的现代用法，兼容 autograd 图
            preserve_rng_state=False, # 不保存 RNG 状态（节省内存，随机性 OK）
            paligemma=self.paligemma,
            gemma_expert=self.gemma_expert,
        )

# 最终的 LayerNorm 也被 checkpoint：
outputs_embeds = torch.utils.checkpoint.checkpoint(
    compute_final_norms, inputs_embeds, adarms_cond,
    use_reentrant=False, preserve_rng_state=False,
)
```

**显存节省原理**：`checkpoint` 不保存中间激活（attention 输出、MLP 中间值等），反向传播需要时重新执行前向计算。对于 18 层模型，理论显存节省约等于 `1 - 1/sqrt(18) ≈ 76%`（实际取决于激活张量大小）。

**`_apply_checkpoint` 辅助方法**：`embed_prefix` 和 `embed_suffix` 内部的子计算（图像嵌入、语言嵌入、state/action 投影）也可通过 `_apply_checkpoint` 包装，进一步节省显存。

---

## 9. 模型架构总结图

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    PI0Policy (PreTrainedPolicy)              │
                    │                                                             │
                    │  ┌───────────────────────────────────────────────────────┐ │
                    │  │                   PI0Pytorch                          │ │
                    │  │                                                       │ │
                    │  │  ┌─────────────────────────────────────────────────┐ │ │
                    │  │  │          PaliGemmaWithExpertModel                │ │ │
                    │  │  │                                                  │ │ │
                    │  │  │  ┌───────────────────┐  ┌──────────────────┐   │ │ │
  图像 ─────────────┼──┼──┼─▶│   SigLIP 视觉塔   │  │  Gemma 嵌入层   │◀──┼─┼─┼── 语言 token
                    │  │  │  │   (256 patch emb) │  │  (lang token emb)│   │ │ │
                    │  │  │  └────────┬──────────┘  └────────┬─────────┘   │ │ │
                    │  │  │           │  前缀嵌入序列          │             │ │ │
                    │  │  │           └──────────┬────────────┘             │ │ │
                    │  │  │                      │ [B, N_prefix, width]     │ │ │
                    │  │  │                      ▼                          │ │ │
                    │  │  │  ┌─────────────────────────────────────────┐   │ │ │
  机器人状态 ────────┼──┼──┼─▶│    联合 Transformer（18 层 Gemma）       │   │ │ │
  （state_proj）    │  │  │  │    ← prefix tokens + suffix tokens →   │   │ │ │
  噪声动作+时间步 ───┼──┼──┼─▶│    注意力 Mask：Prefix-LM + 双向 Action │   │ │ │
  （action_in_proj  │  │  │  └──────────────────────┬──────────────────┘   │ │ │
   + MLP）          │  │  │                         │                      │ │ │
                    │  │  │                         ▼ 取最后 chunk_size 个  │ │ │
                    │  │  │              ┌──────────────────────┐          │ │ │
                    │  │  │              │  action_out_proj      │          │ │ │
                    │  │  │              │  (width → max_action) │          │ │ │
                    │  │  │              └──────────────────────┘          │ │ │
                    │  │  └─────────────────────────────────────────────────┘ │ │
                    │  └───────────────────────────────────────────────────────┘ │
                    │                           │                               │
                    │              Flow Matching Loss / 预测动作                 │
                    └─────────────────────────────────────────────────────────────┘
```

---

## 参考

- 原始论文：[π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/download/pi0.pdf)
- 上游实现：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- 核心文件：
  - `src/lerobot/policies/pi0/configuration_pi0.py`：配置参数定义
  - `src/lerobot/policies/pi0/modeling_pi0.py`：模型实现
  - `src/lerobot/policies/pi0/processor_pi0.py`：数据预处理管道
  - `src/lerobot/utils/constants.py`：`OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38`
