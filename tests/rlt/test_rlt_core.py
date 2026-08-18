"""L0 单元测试：RLT 的模型、缓冲区与损失口径。

这些测试全部不依赖硬件、也不加载 SmolVLA，用于在上真机之前锁住
`replay_buffer` 的时间对齐、residual actor 的恒等性以及 TD 目标的数值。
"""
import pytest
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


def _record(tag: float, rewards=None, done=False, done_step=None, n_xs=None, from_human=False):
    """一个 chunk 的记录；用 tag 把每个 chunk 的张量区分开，便于断言来源。"""
    n_offsets = len(range(0, C, STRIDE)) if n_xs is None else n_xs
    return ChunkRecord(
        xs=torch.arange(n_offsets, dtype=torch.float32).reshape(-1, 1).repeat(1, X_DIM) + tag,
        actions=torch.arange(C, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag,
        rewards=torch.zeros(C) if rewards is None else rewards,
        ref_full=torch.arange(C + C, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag,
        done=done,
        done_step=done_step,
        from_human=from_human,
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


# ------------------------------------------- 失败的人工修正不该被 BC 模仿
def _human_episode(buf, *, success):
    """一集：策略跑一段 -> 人接管一段 -> 成功或失败终止。"""
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0, from_human=True))
    rewards = torch.zeros(C)
    if success:
        rewards[0] = 1.0
    buf.add_chunk(_record(200.0, rewards=rewards, done=True, done_step=1, n_xs=1))


def test_failed_takeover_loses_its_bc_target():
    """人接管后没做成，那段动作照存（critic 要），但不能再当模仿目标。"""
    buf = _buffer()
    _human_episode(buf, success=False)
    n = len(buf)
    assert n > 0
    zeroed = (buf.bc_weight[:n] == 0).sum().item()
    assert zeroed > 0, "介入产生的 transition 应该被摘掉 BC 权重"
    # 执行动作和奖励原样保留
    assert torch.isfinite(buf.action[:n]).all()


def test_successful_takeover_keeps_its_bc_target():
    """修正做成了，那才是比 VLA 更好的模仿目标（论文 Sec. V）。"""
    buf = _buffer()
    _human_episode(buf, success=True)
    n = len(buf)
    assert n > 0
    assert (buf.bc_weight[:n] == 1).all(), "有奖励的一集不该摘掉任何 BC 权重"


def test_policy_transitions_keep_full_weight_in_a_failed_episode():
    """只摘人工介入那部分；策略自己跑失败的段落仍然要向 VLA 参考对齐。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0, done=True, done_step=2, n_xs=1))
    n = len(buf)
    assert n > 0 and (buf.bc_weight[:n] == 1).all()


def test_truncated_episode_also_withdraws_the_bc_target():
    """跑满步数没做成，和按 f 是一回事。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0, from_human=True))
    buf.add_chunk(_record(100.0, from_human=True))
    buf.end_episode(x_last=torch.full((X_DIM,), 7.0))
    n = len(buf)
    assert n > 0 and (buf.bc_weight[:n] == 0).all()


def test_bc_weight_is_sampled_and_masks_the_actor_loss():
    """权重要真的到达 update_actor，否则前面全白做。

    必须先打破零初始化：actor 出厂时 mu 就等于 ref，bc 恒为 0，加不加权都一样。
    """
    import copy

    buf = _buffer()
    _human_episode(buf, success=False)
    batch = buf.sample(8)
    assert "bc_weight" in batch

    agent = RLTAgent(_online_cfg(bc_beta=1000.0), device="cpu")
    _saturate(agent.actor)
    assert (agent.actor.mu(batch["x"], batch["ref"]) - batch["ref"]).abs().max() > 0

    masked = copy.deepcopy(agent).update_actor({**batch, "bc_weight": torch.zeros(8)})
    kept = copy.deepcopy(agent).update_actor({**batch, "bc_weight": torch.ones(8)})
    assert masked["actor_loss"] < kept["actor_loss"], "权重为 0 时 BC 项必须消失"


def test_new_episode_does_not_inherit_the_previous_outcome():
    buf = _buffer()
    _human_episode(buf, success=True)
    first = len(buf)
    _human_episode(buf, success=False)
    assert (buf.bc_weight[:first] == 1).all(), "上一集的成功不该被下一集的失败改写"
    assert (buf.bc_weight[first : len(buf)] == 0).any()


# --------------------------------------------------------------- 分层采样
def _strat_buffer(capacity=4096, **kw):
    opts = {"sample_strategy": "stratified", "recent_episode_window": 2,
            "recent_ratio": 0.4, "human_ratio": 0.2, "reward_ratio": 0.1}
    opts.update(kw)
    return ChunkReplayBuffer(capacity=capacity, x_dim=X_DIM, chunk_len=C, action_dim=D,
                             discount=0.9, stride=STRIDE, device="cpu", **opts)


def _episode(buf, tag, *, human=False, success=False):
    buf.start_episode()
    buf.add_chunk(_record(tag, from_human=human))
    rewards = torch.zeros(C)
    if success:
        rewards[0] = 1.0
    buf.add_chunk(_record(tag + 1, rewards=rewards, from_human=human, done=True, done_step=1, n_xs=1))


def test_stratified_lifts_rare_groups_above_their_share():
    """人工介入只占 6.8% 时，均匀采样会把它冲淡——分层的意义就在这里。"""
    buf = _strat_buffer()
    _episode(buf, 0.0, human=True, success=False)          # 稀有：人工介入
    for i in range(20):                                     # 大量普通策略数据
        _episode(buf, 100.0 + 10 * i)
    n = len(buf)
    human_share = (buf.from_human[:n] > 0).float().mean().item()
    assert human_share < 0.15, "前提：人工数据本来就很少"

    idx = buf._stratified_indices(512)
    drawn = (buf.from_human[idx] > 0).float().mean().item()
    # human_ratio=0.2 是下界；均匀采样只能给到自然占比
    assert drawn >= 0.18, f"分层后 {drawn:.3f}，human_ratio=0.2 没生效（自然 {human_share:.3f}）"


def test_stratified_favours_the_newest_episodes():
    buf = _strat_buffer(recent_episode_window=2)
    for i in range(12):
        _episode(buf, 100.0 + 10 * i)
    newest = buf.episode_id[: len(buf)].max()
    idx = buf._stratified_indices(512)
    recent = (buf.episode_id[idx] >= newest - 1).float().mean().item()
    assert recent > 0.4, f"最近两集只占 {recent:.3f}，recent_ratio=0.4 没生效"


def test_stratified_lifts_rewarded_transitions():
    buf = _strat_buffer()
    _episode(buf, 0.0, success=True)
    for i in range(20):
        _episode(buf, 100.0 + 10 * i)
    n = len(buf)
    natural = (buf.reward_disc[:n] > 0).float().mean().item()
    idx = buf._stratified_indices(512)
    drawn = (buf.reward_disc[idx] > 0).float().mean().item()
    # reward_ratio=0.1 是下界；均匀采样在这批数据上只有 ~0.03
    assert drawn >= 0.09, f"有奖励样本占比 {drawn:.3f}，reward_ratio=0.1 没生效（自然 {natural:.3f}）"


def test_stratified_always_returns_a_full_batch():
    """池子空/不足时必须用均匀采样补齐，不能返回短批。"""
    buf = _strat_buffer()
    _episode(buf, 0.0)  # 没有人工介入，也没有奖励：两个池子全空
    for size in (1, 7, 64, 256):
        assert buf._stratified_indices(size).shape == (size,)


def test_stratified_indices_stay_in_range():
    buf = _strat_buffer()
    for i in range(5):
        _episode(buf, 100.0 + 10 * i, human=(i == 0), success=(i == 1))
    idx = buf._stratified_indices(256)
    assert idx.min() >= 0 and idx.max() < len(buf)


def test_stratified_sample_returns_the_same_keys_as_uniform():
    buf = _strat_buffer()
    _episode(buf, 0.0, human=True, success=True)
    uniform = _buffer()
    _episode(uniform, 0.0, human=True, success=True)
    assert set(buf.sample(8)) == set(uniform.sample(8))


def test_unknown_sample_strategy_is_rejected():
    with pytest.raises(ValueError, match="uniform or stratified"):
        _strat_buffer(sample_strategy="prioritized")


def test_episode_id_and_human_flag_survive_a_save_load(tmp_path):
    buf = _strat_buffer()
    _episode(buf, 0.0, human=True, success=True)
    _episode(buf, 100.0)
    path = tmp_path / "buf.pt"
    buf.save(path)
    restored = _strat_buffer()
    restored.load(path)
    n = len(buf)
    assert torch.equal(restored.episode_id[:n], buf.episode_id[:n])
    assert torch.equal(restored.from_human[:n], buf.from_human[:n])


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
