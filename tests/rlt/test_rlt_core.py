"""L0 单元测试：RLT 的模型、缓冲区与损失口径。

这些测试全部不依赖硬件、也不加载 SmolVLA，用于在上真机之前锁住
`replay_buffer` 的时间对齐、residual actor 的恒等性以及 TD 目标的数值。
"""
import torch

from lerobot.policies.rlt import (
    ActorCriticConfig,
    ChunkActor,
    OnlineRLConfig,
    RLTAgent,
    RLTokenConfig,
    RLTokenModule,
)
from lerobot.rlt.replay_buffer import ChunkRecord, ChunkReplayBuffer

C, D, X_DIM = 4, 3, 5
STRIDE = 2


def _ac_cfg(**kw):
    base = dict(chunk_len=C, action_dim=D, proprio_dim=2, rl_token_dim=X_DIM - 2, hidden_dim=16)
    base.update(kw)
    return ActorCriticConfig(**base)


def _online_cfg(**kw):
    return OnlineRLConfig(ac=_ac_cfg(), device="cpu", batch_size=4, **kw)


def _buffer(discount=0.9, capacity=64):
    return ChunkReplayBuffer(
        capacity=capacity,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=discount,
        stride=STRIDE,
        device="cpu",
    )


def _record(tag: float, rewards=None, done=False, done_step=None, n_xs=None):
    """一个 chunk 的记录；用 tag 把每个 chunk 的张量区分开，便于断言来源。"""
    n_offsets = len(range(0, C, STRIDE)) if n_xs is None else n_xs
    return ChunkRecord(
        xs=torch.arange(n_offsets, dtype=torch.float32).reshape(-1, 1).repeat(1, X_DIM) + tag,
        actions=torch.arange(C, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag,
        rewards=torch.zeros(C) if rewards is None else rewards,
        ref_full=torch.arange(C + C, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag,
        done=done,
        done_step=done_step,
    )


# --------------------------------------------------------------- RL token
def test_rl_token_shapes_and_masking():
    cfg = RLTokenConfig(vla_width=8, d_model=6, n_heads=2, n_encoder_layers=1, n_decoder_layers=1)
    module = RLTokenModule(cfg)
    z = torch.randn(2, 5, cfg.vla_width)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False  # 第二个样本只有 3 个有效 token

    loss, z_rl, metrics = module.reconstruction_loss(z, mask)
    assert z_rl.shape == (2, cfg.d_model)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "recon_rel_err" in metrics and "z_rl_std" in metrics

    # 被 mask 掉的 token 不应影响损失
    z2 = z.clone()
    z2[1, 3:] = 1e3
    module.eval()  # 冻结统计量，否则大值会改变归一化
    loss_a, _, _ = module.reconstruction_loss(z, mask)
    loss_b, _, _ = module.reconstruction_loss(z2, mask)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_rl_token_loss_decreases():
    torch.manual_seed(0)
    cfg = RLTokenConfig(vla_width=8, d_model=8, n_heads=2, n_encoder_layers=1, n_decoder_layers=1)
    module = RLTokenModule(cfg)
    opt = torch.optim.Adam(module.parameters(), lr=1e-3)
    z = torch.randn(4, 6, cfg.vla_width)

    first = last = None
    for step in range(60):
        loss, _, _ = module.reconstruction_loss(z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first


# ------------------------------------------------------------------ actor
def test_actor_starts_at_the_vla_reference():
    """零初始化的 residual 保证第一条真机动作就是 VLA 的参考动作。"""
    actor = ChunkActor(_ac_cfg())
    x = torch.randn(4, X_DIM)
    ref = torch.rand(4, C, D) * 1.6 - 0.8
    assert torch.allclose(actor.mu(x, ref), ref, atol=1e-6)


def _saturate(actor):
    """打破零初始化，让 residual 顶到上界。"""
    with torch.no_grad():
        last = [m for m in actor.net if isinstance(m, torch.nn.Linear)][-1]
        last.weight.fill_(5.0)
        last.bias.fill_(5.0)


def test_residual_box_is_the_only_bound():
    """动作只相对参考受限，不设绝对上界——这样才不依赖归一化模式的取值范围。"""
    actor = ChunkActor(_ac_cfg(max_residual=0.1))
    _saturate(actor)
    x = torch.randn(4, X_DIM)
    ref = torch.full((4, C, D), 0.95)

    assert actor.residual(x, ref).abs().max() <= 0.1 + 1e-6
    mu = actor.mu(x, ref)
    assert torch.all((mu - ref).abs() <= 0.1 + 1e-6)


def test_a_reference_beyond_one_survives_untouched():
    """SmolVLA 用 MEAN_STD 归一化，|a|>1 只是超过 1 个标准差，占 LIBERO 动作的 27%。

    以前这里有一刀 clamp(±1)，会把这些动作静默截断——每步丢掉约 4mm 平移，夹爪闭合
    只剩 71% 力度，而且 warmup 执行未截断的参考、RL 一接手就开始截断，
    "零初始化 => step 0 等于 base VLA" 实际上不成立。
    """
    actor = ChunkActor(_ac_cfg())
    x = torch.randn(4, X_DIM)
    ref = torch.full((4, C, D), 2.5)
    assert torch.allclose(actor.mu(x, ref), ref, atol=1e-6)


def test_exploration_noise_is_not_clipped_either():
    actor = ChunkActor(_ac_cfg(action_std=0.05))
    x = torch.randn(256, X_DIM)
    ref = torch.full((256, C, D), 3.0)
    sampled = actor.sample(x, ref)
    assert sampled.max() > 3.0, "噪声被削成单向就不再是零均值扰动"
    assert sampled.min() < 3.0


def test_ref_dropout_masks_input_but_not_the_residual_base():
    actor = ChunkActor(_ac_cfg(ref_dropout=1.0))
    ref = torch.rand(8, C, D)
    assert torch.count_nonzero(actor.apply_ref_dropout(ref)) == 0
    # 即使输入被完全 drop，输出仍以完整 ref 为基准
    assert torch.allclose(actor.mu(ref_in=torch.zeros_like(ref), x=torch.randn(8, X_DIM), ref_chunk=ref), ref, atol=1e-6)


# ----------------------------------------------------------- replay buffer
def test_offsets_are_time_aligned_across_chunks():
    """offset o 的 x_next 必须正好是 C 步之后的状态，即 chunk k+1 的 offset o。"""
    buf = _buffer()
    k0, k1 = _record(0.0), _record(100.0)
    buf.add_chunk(k0)
    buf.add_chunk(k1)

    assert len(buf) == len(range(0, C, STRIDE))  # k0 的每个 offset 一条
    for i, o in enumerate(range(0, C, STRIDE)):
        assert torch.allclose(buf.x[i], k0.xs[i])
        assert torch.allclose(buf.x_next[i], k1.xs[i])
        # 动作窗口跨两个 chunk：k0[o:] ++ k1[:o]
        expected = torch.cat([k0.actions[o:], k1.actions[:o]], dim=0)
        assert torch.allclose(buf.action[i], expected)
        assert buf.actual_steps[i] == C


def test_discounted_chunk_return():
    gamma = 0.9
    buf = _buffer(discount=gamma)
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    buf.add_chunk(_record(0.0, rewards=rewards))
    buf.add_chunk(_record(100.0))
    expected = gamma**1 + gamma**3
    assert abs(buf.reward_disc[0].item() - expected) < 1e-6


def test_early_success_marks_done_and_records_real_step_count():
    buf = _buffer()
    rewards = torch.zeros(C)
    rewards[0] = 1.0
    # chunk k+1 第 1 步就成功
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0, rewards=rewards, done=True, done_step=1, n_xs=1))

    dones = buf.done[: buf.size]
    steps = buf.actual_steps[: buf.size]
    assert dones.max() == 1.0  # 终止落在某个窗口内
    assert steps.min() < C  # 短窗口保留了真实步数，供 gamma^k 使用
    assert steps.max() <= C


def test_truncation_without_a_real_next_state_is_dropped():
    """截断样本仍要 bootstrap，缺少真实 x_next 时宁可丢弃也不能自指。"""
    buf = _buffer()
    buf.add_chunk(_record(0.0, done_step=3, n_xs=2))
    buf.end_episode(x_last=None)
    assert len(buf) == 0

    buf2 = _buffer()
    x_last = torch.full((X_DIM,), 42.0)
    buf2.add_chunk(_record(0.0, done_step=3, n_xs=2))
    buf2.end_episode(x_last=x_last)
    assert len(buf2) > 0
    for i in range(len(buf2)):
        assert buf2.done[i] == 0.0
        assert torch.allclose(buf2.x_next[i], x_last)
        assert not torch.allclose(buf2.x_next[i], buf2.x[i])


def test_truncated_window_bootstraps_against_the_real_next_reference():
    """截断窗口的 bootstrap 生效，ref_next 置零会让 target 动作变成纯残差噪声。"""
    buf = _buffer()
    d_step = 3
    rec = _record(0.0, done_step=d_step, n_xs=2)
    buf.add_chunk(rec)
    buf.end_episode(x_last=torch.full((X_DIM,), 42.0))

    assert len(buf) > 0
    expected = rec.ref_full[d_step : d_step + C]
    for i in range(len(buf)):
        assert torch.allclose(buf.ref_next[i], expected)
        assert not torch.allclose(buf.ref_next[i], torch.zeros_like(expected))


def test_short_reference_is_padded_instead_of_raising():
    """真实 ref_full 总够长（VLA H=50 / 人工平铺 2C），手搓记录不一定。"""
    buf = _buffer()
    rec = _record(0.0, done_step=C, n_xs=2)
    rec.ref_full = rec.ref_full[: C + 2]
    buf.add_chunk(rec)
    buf.end_episode(x_last=torch.full((X_DIM,), 42.0))

    assert len(buf) > 0
    assert torch.allclose(buf.ref_next[0][-1], rec.ref_full[-1])


def test_save_and_load_roundtrip(tmp_path):
    buf = _buffer()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0))
    path = tmp_path / "buf.pt"
    buf.save(path)

    restored = _buffer()
    restored.load(path)
    assert len(restored) == len(buf)
    assert torch.allclose(restored.x[: len(buf)], buf.x[: len(buf)])
    assert torch.allclose(restored.actual_steps[: len(buf)], buf.actual_steps[: len(buf)])


# ------------------------------------------------------------ TD3 updates
def test_td_target_uses_gamma_to_the_actual_step_count():
    cfg = _online_cfg()
    agent = RLTAgent(cfg, device="cpu")
    batch = {
        "x": torch.zeros(2, X_DIM),
        "action": torch.zeros(2, C, D),
        "ref": torch.zeros(2, C, D),
        "reward_disc": torch.tensor([1.0, 1.0]),
        "x_next": torch.zeros(2, X_DIM),
        "ref_next": torch.zeros(2, C, D),
        "done": torch.tensor([0.0, 1.0]),
        "actual_steps": torch.tensor([2.0, 2.0]),
    }
    metrics = agent.update_critic(batch)
    assert torch.isfinite(torch.tensor(metrics["critic_loss"]))
    # done=1 的样本目标就是折扣回报本身，done=0 的样本多一个 bootstrap 项
    assert metrics["target_mean"] >= 1.0


def test_bc_term_uses_the_papers_sum_convention():
    """beta 的量纲必须按 ||a - a~||^2 求和，否则等效 beta 被除以 C*d。"""
    cfg = _online_cfg(bc_beta=1.0)
    agent = RLTAgent(cfg, device="cpu")
    with torch.no_grad():  # 让 actor 输出偏离 ref 一个已知常量
        last = [m for m in agent.actor.net if isinstance(m, torch.nn.Linear)][-1]
        last.bias.fill_(0.05)

    ref = torch.zeros(1, C, D)
    batch = {"x": torch.zeros(1, X_DIM), "ref": ref}
    metrics = agent.update_actor(batch)
    # 每个元素偏差 0.05 -> 求和口径下 bc = C*D*0.05^2
    assert abs(metrics["bc_dist"] - C * D * 0.05**2) < 1e-5


def test_q_bootstrap_is_clamped():
    cfg = _online_cfg()
    agent = RLTAgent(cfg, device="cpu")
    expected = 1.0 / (1.0 - cfg.discount**C)
    assert abs(agent.q_max - expected) < 1e-6
