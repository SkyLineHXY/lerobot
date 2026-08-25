"""L0 单元测试：RLT 的模型、缓冲区与损失口径。

这些测试全部不依赖硬件、也不加载 SmolVLA，用于在上真机之前锁住 `replay_buffer`
的窗口构建、actor 的直接输出口径以及 TD 目标的数值——即 openpi-RLT 里
`networks.py` / `trainer.py` / `replay.py` 三个文件对应的行为。
"""
import copy

import pytest
import torch

from lerobot.policies.rlt import (
    ActorCriticConfig,
    ChunkActor,
    OnlineRLConfig,
    RLTAgent,
    RLTokenConfig,
    RLTokenModule,
    TransitionSource,
)
from lerobot.rlt.replay_buffer import (
    COLLECTION_PHASE_WARMUP,
    ChunkRecord,
    ChunkReplayBuffer,
)

C, D, X_DIM = 4, 3, 5
STRIDE = 2


def _ac_cfg(**kw):
    base = {"chunk_len": C, "action_dim": D, "proprio_dim": 2,
            "rl_token_dim": X_DIM - 2, "hidden_dim": 16}
    base.update(kw)
    return ActorCriticConfig(**base)


def _online_cfg(**kw):
    kw.setdefault("ac", _ac_cfg())
    return OnlineRLConfig(device="cpu", batch_size=4, **kw)


def _buffer(discount=0.9, capacity=64, **kw):
    return ChunkReplayBuffer(
        capacity=capacity,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=discount,
        stride=STRIDE,
        device="cpu",
        **kw,
    )


def _record(tag: float, *, base=0, n_exec=C, rewards=None, done=False, source=TransitionSource.RL):
    """一个 chunk 的记录；用 tag 把每个 chunk 的张量区分开，便于断言来源。

    anchor 落在绝对步号是 STRIDE 整数倍的位置，和 `RolloutWorker` 的做法一致。
    """
    offsets = [o for o in range(n_exec) if (base + o) % STRIDE == 0]
    return ChunkRecord(
        xs=torch.tensor([[float(tag + o)] * X_DIM for o in offsets]),
        x_offsets=torch.tensor(offsets, dtype=torch.long),
        # Each anchor now carries the reference sampled *at* that anchor, not a
        # shifted slice of the chunk boundary's plan.
        refs=torch.stack([
            torch.arange(C, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag + o + 0.5
            for o in offsets
        ]),
        aligned=torch.tensor([o == 0 for o in offsets], dtype=torch.bool),
        actions=torch.arange(n_exec, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag,
        rewards=torch.zeros(n_exec) if rewards is None else rewards[:n_exec],
        source=int(source),
        done=done,
    )


def _tail(tag: float = 999.0):
    """end_episode 需要的末尾 anchor。"""
    return (
        torch.full((X_DIM,), tag),
        torch.arange(C, dtype=torch.float32).reshape(-1, 1).repeat(1, D) + tag,
    )


def _batch(n=8, **kw):
    out = {
        "x": torch.randn(n, X_DIM),
        "action": torch.randn(n, C, D),
        "ref": torch.randn(n, C, D),
        "rewards": torch.zeros(n, C),
        "x_next": torch.randn(n, X_DIM),
        "ref_next": torch.randn(n, C, D),
        "done": torch.zeros(n),
        "mc_return": torch.zeros(n),
        "mc_valid": torch.zeros(n),
        "aligned": torch.ones(n),
        "source_chunk": torch.full((n, C), int(TransitionSource.RL), dtype=torch.uint8),
    }
    out.update(kw)
    return out


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
def test_actor_emits_the_chunk_directly():
    """residual_scale=0 时 actor 直接输出整条 chunk，参考只是输入特征。"""
    actor = ChunkActor(_ac_cfg(residual_scale=0.0))
    x = torch.randn(4, X_DIM)
    ref = torch.rand(4, C, D) * 1.6 - 0.8
    mu = actor.mu(x, ref)
    assert mu.shape == (4, C, D)
    assert torch.isfinite(mu).all()
    assert not torch.allclose(mu, ref), "直接输出不应恰好复刻参考"


def test_residual_actor_starts_at_the_reference_and_stays_in_the_box():
    """旧的 residual 参数化只作为显式消融保留。"""
    actor = ChunkActor(_ac_cfg(residual_scale=0.3, ref_dropout=0.0, action_std=0.0))
    x = torch.randn(4, X_DIM)
    ref = torch.rand(4, C, D) * 1.6 - 0.8
    assert torch.allclose(actor.mu(x, ref), ref, atol=1e-6), "零初始化的残差层必须复刻参考"

    with torch.no_grad():  # 把 trunk 推到饱和，检查盒子是硬边界
        actor.trunk[-1].weight.normal_(0.0, 50.0)
        actor.trunk[-1].bias.normal_(0.0, 50.0)
    assert (actor.mu(x, ref) - ref).abs().max() <= 0.3 + 1e-6

def test_a_reference_beyond_one_is_not_clipped():
    """SmolVLA 用 MEAN_STD 归一化，|a|>1 只是超过 1 个标准差，不该被静默截断。

    参考实现在网络里既不 tanh 也不 clamp；物理限幅放在环境侧（robosuite 的
    `scale_action`、Piper 的 `rate_limit_joints`），那里量纲是明确的。
    """
    actor = ChunkActor(_ac_cfg())
    x = torch.randn(4, X_DIM)
    ref = torch.full((4, C, D), 2.5)
    assert torch.isfinite(actor.mu(x, ref)).all()


def test_exploration_noise_is_not_clipped_either():
    actor = ChunkActor(_ac_cfg(action_std=0.05))
    x = torch.randn(256, X_DIM)
    ref = torch.full((256, C, D), 3.0)
    mu = actor.mu(x, ref)
    sampled = actor.sample(x, ref)
    assert (sampled > mu).any() and (sampled < mu).any(), "噪声被削成单向就不再是零均值扰动"


def test_ref_dropout_zeroes_the_whole_chunk_per_sample():
    actor = ChunkActor(_ac_cfg(ref_dropout=1.0, residual_scale=0.0, action_std=0.0))
    ref = torch.rand(8, C, D)
    masked = actor.apply_ref_dropout(ref)
    assert torch.count_nonzero(masked) == 0
    # dropout 后没有 `ref_base` 旁路；输出只能由 x 和零 reference 生成。
    x = torch.randn(8, X_DIM)
    dropped = actor.sample(x, masked, deterministic=True)
    assert torch.allclose(dropped, actor.mu(x, torch.zeros_like(ref)))
    assert not torch.allclose(dropped, ref)


# ----------------------------------------------------------- replay buffer
def test_windows_are_time_aligned_and_span_chunks():
    """窗口 start 的 x_next 必须正好是 C 步之后的状态，动作窗口可以跨 chunk。"""
    buf = _buffer()
    k0, k1 = _record(0.0, base=0), _record(100.0, base=C)
    buf.start_episode()
    buf.add_chunk(k0)
    buf.add_chunk(k1)
    x_last, ref_last = _tail()
    n = buf.end_episode(x_last, ref_last)

    assert n == len(range(0, 2 * C, STRIDE))  # start = 0, 2, 4, 6
    all_actions = torch.cat([k0.actions, k1.actions], dim=0)
    all_x = torch.cat([k0.xs, k1.xs, x_last.unsqueeze(0)], dim=0)
    for i, start in enumerate([0, STRIDE, C]):
        assert torch.allclose(buf.x[i], all_x[start // STRIDE])
        assert torch.allclose(buf.x_next[i], all_x[(start + C) // STRIDE])
        assert torch.allclose(buf.action[i], all_actions[start : start + C])

    # 越界的那个窗口补零，并且 bootstrap 到 episode 末尾那个状态
    assert torch.allclose(buf.action[3][: 2 * C - C - STRIDE], all_actions[C + STRIDE :])
    assert torch.allclose(buf.action[3][2 * C - C - STRIDE :], torch.zeros(STRIDE, D))
    assert torch.allclose(buf.x_next[3], x_last)


def test_rewards_are_stored_per_step_and_zero_padded():
    buf = _buffer()
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    buf.start_episode()
    buf.add_chunk(_record(0.0, rewards=rewards))
    buf.add_chunk(_record(100.0, base=C))
    buf.end_episode(*_tail())
    assert torch.allclose(buf.rewards[0], rewards)


def test_early_termination_marks_done_inside_the_window():
    buf = _buffer()
    rewards = torch.zeros(C)
    rewards[0] = 1.0
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0, base=C, n_exec=2, rewards=rewards, done=True))
    buf.end_episode(*_tail(), success=True)

    n = len(buf)
    assert n > 0
    assert buf.done[:n].max() == 1.0  # 终止落在某个窗口内
    assert buf.success[:n].max() == 1


def test_truncation_without_a_real_next_state_drops_everything():
    """截断样本仍要 bootstrap；没有真实 x_next 时宁可一条不存，也不能自指。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0, n_exec=3))
    assert buf.end_episode(None, None) == 0


def test_a_short_episode_still_yields_partial_windows():
    """3 步凑不满 C=4，但奖励只会出现在这种尾巴上，所以不能整条丢掉。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0, n_exec=3))
    x_last, ref_last = _tail(42.0)
    assert buf.end_episode(x_last, ref_last) == 2  # start = 0, 2

    for i in range(2):
        assert buf.done[i] == 0.0
        assert torch.allclose(buf.x_next[i], x_last)
        assert not torch.allclose(buf.x_next[i], buf.x[i])


def test_truncated_window_bootstraps_against_the_real_next_reference():
    """ref_next 置零会让 target 动作落在部署时从未见过的输入上。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    x_last, ref_last = _tail(42.0)
    buf.end_episode(x_last, ref_last)
    assert torch.allclose(buf.ref_next[0], ref_last)
    assert not torch.allclose(buf.ref_next[0], torch.zeros_like(ref_last))


def test_short_reference_is_padded_instead_of_raising():
    """真实的 VLA 参考总够长（H=50），手搓记录不一定。"""
    buf = _buffer()
    rec = _record(0.0)
    rec.refs = rec.refs[:, : C - 1]
    buf.start_episode()
    buf.add_chunk(rec)
    buf.end_episode(*_tail())
    assert len(buf) > 0
    assert torch.allclose(buf.ref[0, : C - 1], rec.refs[0])
    assert torch.allclose(buf.ref[0, C - 1], rec.refs[0, -1])


def _terminal_episode():
    """7 步的一集，奖励只落在最后一步（第 6 步）。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(
        _record(100.0, base=C, n_exec=3, rewards=torch.tensor([0.0, 0.0, 1.0]), done=True)
    )
    buf.end_episode(*_tail(), success=True)
    return buf


def test_the_sparse_reward_lands_in_every_window_that_saw_it():
    """只留 start+C<=n 的窗口时，整集只有 1 条 transition 见过奖励——远远不够。"""
    buf = _terminal_episode()
    n = len(buf)
    assert n == len(range(0, 7, STRIDE))  # start = 0, 2, 4, 6
    # start=4 和 start=6 的窗口都覆盖到第 6 步
    assert int((buf.rewards[:n].sum(dim=-1) > 0).sum()) == C // STRIDE
    assert (buf.done[:n] > 0).any()


def test_mc_return_is_the_discounted_return_to_go():
    buf = _terminal_episode()
    n = len(buf)
    g = buf.discount
    # 奖励在第 6 步，窗口 start = 0, 2, 4, 6
    assert torch.allclose(
        buf.mc_return[:n],
        torch.tensor([g**6, g**4, g**2, 1.0]),
        atol=1e-6,
    )
    assert (buf.mc_valid[:n] > 0).all()


def test_a_truncated_episode_has_no_usable_mc_return():
    """截断只是别人停了看，不是"后面没有奖励"，当成 0 会把 critic 往下拽。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0, base=C))
    buf.end_episode(*_tail())
    assert len(buf) > 0
    assert (buf.mc_valid[: len(buf)] == 0).all()


def test_a_timeout_declared_as_failure_is_terminal():
    """失败超时既是可靠的零回报，也必须屏蔽越过 episode 边界的 bootstrap。"""
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0, n_exec=3))
    buf.end_episode(*_tail(), success=False, truncation_is_failure=True)

    assert len(buf) == 2
    assert (buf.mc_valid[: len(buf)] == 1).all()
    assert (buf.done[: len(buf)] == 1).all()


def test_mc_blend_replaces_the_td_target_when_fully_weighted():
    agent = RLTAgent(_online_cfg(critic_mc_weight=1.0, discount=0.9), device="cpu")
    batch = _batch(n=8)
    batch["mc_return"] = torch.full((8,), 0.7)
    batch["mc_valid"] = torch.ones(8)
    batch["rewards"][:, 0] = 1.0  # TD 目标本来会是非零的另一个值

    for _ in range(400):
        agent.update_critic(batch)
    assert torch.allclose(
        agent.critic.min_q(batch["x"], batch["action"]),
        torch.full((8,), 0.7),
        atol=0.05,
    )


def test_mc_blend_is_skipped_where_it_is_not_valid():
    agent = RLTAgent(_online_cfg(critic_mc_weight=1.0, discount=0.9), device="cpu")
    batch = _batch(n=8)
    batch["mc_return"] = torch.full((8,), 5.0)
    batch["mc_valid"] = torch.zeros(8)
    before = agent.update_critic(batch)["target_mean"]
    assert abs(before) < 1.0  # 5.0 没有渗进来


def test_discard_drops_the_trace_without_touching_stored_rows():
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0))
    buf.add_chunk(_record(100.0, base=C))
    buf.end_episode(*_tail())
    stored, added = len(buf), buf.total_added

    buf.start_episode()
    buf.add_chunk(_record(7.0))
    assert buf.discard_episode() == C
    assert len(buf) == stored and buf.total_added == added
    assert buf.end_episode(*_tail()) == 0


def test_chunk_len_must_be_a_multiple_of_stride():
    with pytest.raises(ValueError, match="multiple of stride"):
        ChunkReplayBuffer(
            capacity=8, x_dim=X_DIM, chunk_len=5, action_dim=D, discount=0.9, stride=2, device="cpu"
        )


def test_human_chunk_preserves_the_vla_reference_by_default():
    """部署时 actor 条件是 VLA proposal；human action 只能替换监督目标。"""
    buf = _buffer()
    rec = _record(0.0, source=TransitionSource.HUMAN)
    buf.start_episode()
    buf.add_chunk(rec)
    buf.end_episode(*_tail())
    assert torch.allclose(buf.ref[0], rec.refs[0])
    assert not torch.allclose(buf.ref[0], buf.action[0])
    assert (buf.source_chunk[0] == int(TransitionSource.HUMAN)).all()
    assert buf.source[0] == int(TransitionSource.HUMAN)
    assert buf.intervention[0] == 1.0


def test_human_ref_override_is_an_explicit_legacy_ablation():
    buf = _buffer(human_ref_override=True)
    rec = _record(0.0, source=TransitionSource.HUMAN)
    buf.start_episode()
    buf.add_chunk(rec)
    buf.end_episode(*_tail())
    assert torch.allclose(buf.ref[0], buf.action[0])


def test_only_the_chunk_boundary_anchor_is_marked_aligned():
    """子采样出来的 anchor 横跨两次规划决策，不能当 actor 的模仿目标。"""
    buf = _buffer()
    buf.start_episode()
    for i in range(4):
        buf.add_chunk(_record(float(i), base=i * C))
    buf.end_episode(*_tail())
    assert buf.aligned[: len(buf)].float().mean() == pytest.approx(STRIDE / C)


def test_window_spanning_human_and_policy_is_mixed():
    buf = _buffer()
    buf.start_episode()
    buf.add_chunk(_record(0.0, n_exec=2, source=TransitionSource.RL))
    buf.add_chunk(_record(100.0, base=2, n_exec=4, source=TransitionSource.HUMAN))
    buf.end_episode(*_tail())
    assert buf.source[0] == int(TransitionSource.MIXED)
    assert buf.intervention[0] == 1.0


# --------------------------------------------------------------- 分层采样
def _strat_buffer(capacity=4096, **kw):
    opts = {"sample_strategy": "stratified", "recent_episode_window": 2,
            "recent_online_ratio": 0.4, "warmup_demo_ratio": 0.3,
            "human_intervention_ratio": 0.2}
    opts.update(kw)
    return _buffer(capacity=capacity, **opts)


def _episode(buf, tag, *, human=False, warmup=False, success=False):
    buf.start_episode(warmup=warmup)
    source = TransitionSource.HUMAN if human else TransitionSource.RL
    buf.add_chunk(_record(tag, source=source))
    buf.add_chunk(_record(tag + 10, base=C, source=source))
    buf.end_episode(*_tail(tag + 900), success=success)


def test_stratified_lifts_human_takeovers_above_their_share():
    """人工介入只占几个百分点时，均匀采样会把它冲淡——分层的意义就在这里。"""
    buf = _strat_buffer()
    _episode(buf, 0.0, human=True)
    for i in range(20):
        _episode(buf, 100.0 + 10 * i)
    n = len(buf)
    natural = (buf.intervention[:n] > 0).float().mean().item()
    assert natural < 0.15, "前提：人工数据本来就很少"

    idx = buf._stratified_indices(512)
    drawn = (buf.intervention[idx] > 0).float().mean().item()
    assert drawn >= 0.18, f"分层后 {drawn:.3f}，human_intervention_ratio=0.2 没生效"


def test_stratified_favours_recent_online_episodes():
    buf = _strat_buffer(recent_episode_window=2)
    for i in range(12):
        _episode(buf, 100.0 + 10 * i)
    newest = buf.episode_id[: len(buf)].max()
    idx = buf._stratified_indices(512)
    recent = (buf.episode_id[idx] >= newest - 1).float().mean().item()
    assert recent > 0.4, f"最近两集只占 {recent:.3f}，recent_online_ratio=0.4 没生效"


def test_stratified_lifts_warmup_transitions():
    buf = _strat_buffer()
    _episode(buf, 0.0, warmup=True)
    for i in range(20):
        _episode(buf, 100.0 + 10 * i)
    n = len(buf)
    natural = (buf.phase[:n] == COLLECTION_PHASE_WARMUP).float().mean().item()
    idx = buf._stratified_indices(512)
    drawn = (buf.phase[idx] == COLLECTION_PHASE_WARMUP).float().mean().item()
    assert drawn >= 0.28, f"warmup 样本占比 {drawn:.3f}（自然 {natural:.3f}）"


def test_stratified_always_returns_a_full_batch():
    """池子空/不足时必须用均匀采样补齐，不能返回短批。"""
    buf = _strat_buffer()
    _episode(buf, 0.0)  # 没有人工介入，也没有 warmup：两个池子全空
    for size in (1, 7, 64, 256):
        assert buf._stratified_indices(size).shape == (size,)


def test_stratified_indices_stay_in_range():
    buf = _strat_buffer()
    for i in range(5):
        _episode(buf, 100.0 + 10 * i, human=(i == 0), warmup=(i == 1))
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


def test_save_and_load_roundtrip(tmp_path):
    buf = _strat_buffer()
    _episode(buf, 0.0, human=True, warmup=True, success=True)
    _episode(buf, 100.0)
    path = tmp_path / "buf.pt"
    buf.save(path)

    restored = _strat_buffer()
    restored.load(path)
    n = len(buf)
    assert len(restored) == n
    for key in ("x", "action", "ref", "rewards", "x_next", "ref_next", "done"):
        assert torch.allclose(getattr(restored, key)[:n], getattr(buf, key)[:n])
    for key in ("source_chunk", "source", "phase", "success", "episode_id"):
        assert torch.equal(getattr(restored, key)[:n], getattr(buf, key)[:n])

    # Serialized tensor views must not retain the capacity-sized backing
    # storage of the preallocated replay buffer.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in buf._COLUMNS:
        tensor = saved[key]
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def test_a_pre_openpi_buffer_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"x": torch.zeros(2, X_DIM), "size": 2, "ptr": 2}, path)
    with pytest.raises(ValueError, match="predates the current schema"):
        _buffer().load(path)


# ------------------------------------------------------------ TD3 updates
def test_td_target_bootstraps_with_gamma_to_the_chunk_length():
    cfg = _online_cfg(discount=0.9)
    agent = RLTAgent(cfg, device="cpu")
    rewards = torch.zeros(2, C)
    rewards[:, 0] = 1.0
    batch = _batch(2, rewards=rewards, done=torch.tensor([0.0, 1.0]))

    with torch.no_grad():
        a_next = agent.actor_target.mu(batch["x_next"], batch["ref_next"])
        q_next = agent.critic_target.min_q(batch["x_next"], a_next)
    metrics = agent.update_critic(batch)

    # done=1 的样本目标就是折扣回报本身；done=0 的多一个 gamma^C 的 bootstrap
    lo = 1.0 + (0.9**C) * (q_next.min().item() - 3 * cfg.ac.action_std)
    hi = 1.0 + (0.9**C) * (q_next.max().item() + 3 * cfg.ac.action_std)
    assert lo <= metrics["target_mean"] * 2 - 1.0 <= hi


def test_td_target_comes_from_the_target_actor():
    agent = RLTAgent(_online_cfg(ac=_ac_cfg(action_std=0.0)), device="cpu")
    batch = _batch(8)
    before = agent.update_critic(batch)["target_mean"]
    with torch.no_grad():  # 只动 target actor，target 必须跟着变
        for p in agent.actor_target.parameters():
            p.add_(1.0)
    after = agent.update_critic(batch)["target_mean"]
    assert abs(after - before) > 1e-6


def test_targets_move_only_on_actor_update_steps():
    cfg = _online_cfg(tau=0.5, actor_update_period=2)
    agent = RLTAgent(cfg, device="cpu")
    batch = _batch(8)
    snapshot = copy.deepcopy(agent.critic_target.state_dict())

    agent.update(batch)  # critic only
    assert all(
        torch.equal(v, snapshot[k]) for k, v in agent.critic_target.state_dict().items()
    )
    agent.update(batch)  # critic + actor + both target EMAs
    assert any(
        not torch.equal(v, snapshot[k]) for k, v in agent.critic_target.state_dict().items()
    )


def test_human_bc_targets_executed_actions_without_replacing_actor_input():
    """human 步拟合执行动作，policy 步拟合 VLA ref。"""
    cfg = _online_cfg(
        ac=_ac_cfg(action_std=0.0, ref_dropout=0.0, residual_scale=0.0),
        human_bc_weight=1.0,
        awr_weight=0.0,
    )
    agent = RLTAgent(cfg, device="cpu")
    n = 8
    source = torch.full((n, C), int(TransitionSource.RL), dtype=torch.uint8)
    source[:, :2] = int(TransitionSource.HUMAN)
    batch = _batch(n, source_chunk=source)

    with torch.no_grad():
        a = agent.actor.mu(batch["x"], batch["ref"])
    metrics = agent.update_actor(batch, bc_weight=1.0, q_weight=0.0)
    target = torch.where(source.eq(int(TransitionSource.HUMAN)).unsqueeze(-1),
                         batch["action"], batch["ref"])
    assert abs(metrics["bc_penalty"] - (a - target).pow(2).mean().item()) < 1e-5
    assert metrics["human_mask_ratio"] == pytest.approx(2 / C)


def test_reference_bc_uses_every_fresh_anchor_even_when_behavior_spans_plans():
    """aligned 只约束行为模仿；每个 anchor 的新鲜 VLA ref 都是合法 BC 目标。"""
    cfg = _online_cfg(
        ac=_ac_cfg(action_std=0.0, ref_dropout=0.0, residual_scale=0.0),
        human_bc_weight=1.0,
        awr_weight=0.0,
    )
    agent = RLTAgent(cfg, device="cpu")
    n = 8
    aligned = torch.zeros(n)
    aligned[:3] = 1.0
    batch = _batch(n, aligned=aligned)
    with torch.no_grad():
        a = agent.actor.mu(batch["x"], batch["ref"])
    metrics = agent.update_actor(batch, bc_weight=1.0, q_weight=0.0)
    expected = (a - batch["ref"]).pow(2).mean().item()
    assert metrics["bc_penalty"] == pytest.approx(expected, abs=1e-5)
    assert metrics["aligned_ratio"] == pytest.approx(3 / n)


def test_human_steps_are_upweighted_in_the_bc_term():
    """接管步在 BC 里按 human_bc_weight 加权——12% 的占比在平均里活不下来。"""
    n, ac = 8, _ac_cfg(action_std=0.0, ref_dropout=0.0, residual_scale=0.0)
    source = torch.full((n, C), int(TransitionSource.RL), dtype=torch.uint8)
    source[:, :2] = int(TransitionSource.HUMAN)
    batch = _batch(n, source_chunk=source)

    penalties = {}
    for w in (1.0, 5.0):
        agent = RLTAgent(_online_cfg(ac=ac, human_bc_weight=w, awr_weight=0.0), device="cpu")
        torch.manual_seed(0)
        with torch.no_grad():
            a = agent.actor.mu(batch["x"], batch["ref"])
        target = torch.where(source.eq(int(TransitionSource.HUMAN)).unsqueeze(-1),
                             batch["action"], batch["ref"])
        sq = (a - target).pow(2).mean(dim=-1)
        step_w = 1.0 + (w - 1.0) * (source == int(TransitionSource.HUMAN)).float()
        expected = ((sq * step_w).sum() / step_w.sum()).item()
        penalties[w] = agent.update_actor(batch, bc_weight=1.0, q_weight=0.0)["bc_penalty"]
        assert penalties[w] == pytest.approx(expected, abs=1e-5)
    assert penalties[5.0] != pytest.approx(penalties[1.0], abs=1e-4)


def test_human_bc_learns_the_correction_without_changing_the_conditioning_reference():
    """端到端回归：同一个 VLA 输入下，actor 确实向 human execution 收敛。"""
    torch.manual_seed(0)
    cfg = _online_cfg(
        ac=_ac_cfg(action_std=0.0, ref_dropout=0.0, residual_scale=0.0),
        actor_lr=3e-3,
        human_bc_weight=1.0,
        awr_weight=0.0,
    )
    agent = RLTAgent(cfg, device="cpu")
    n = 8
    source = torch.full((n, C), int(TransitionSource.HUMAN), dtype=torch.uint8)
    batch = _batch(n, source_chunk=source)
    batch["action"] = batch["ref"] + 0.25
    original_ref = batch["ref"].clone()

    with torch.no_grad():
        before = (agent.actor.mu(batch["x"], batch["ref"]) - batch["action"]).pow(2).mean()
    for _ in range(100):
        agent.update_actor(batch, bc_weight=1.0, q_weight=0.0)
    with torch.no_grad():
        after = (agent.actor.mu(batch["x"], batch["ref"]) - batch["action"]).pow(2).mean()

    assert after < before * 0.1
    assert torch.equal(batch["ref"], original_ref), "human target 不得改写 actor 条件输入"


def test_warmup_and_online_use_different_loss_weights():
    cfg = _online_cfg(
        ac=_ac_cfg(residual_scale=0.0),
        actor_update_period=1,
        warmup_bc_weight=10.0, warmup_q_weight=0.0,
        online_bc_weight=0.0, online_q_weight=1.0,
        weight_ramp_updates=0,
        awr_weight=0.0,
    )
    batch = _batch(8)
    warm = copy.deepcopy(RLTAgent(cfg, device="cpu")).update(batch, warmup=True)
    online = copy.deepcopy(RLTAgent(cfg, device="cpu")).update(batch, warmup=False)
    assert warm["actor_loss"] > 0 and warm["bc_penalty"] > 0
    assert warm["bc_weight"] == 10.0 and online["bc_weight"] == 0.0
    # 在线相只剩论文 Eq. (5) 的固定权重 Q 项。
    assert online["actor_loss"] == pytest.approx(
        -online["actor_q_weight"] * online["actor_q"], abs=1e-6
    )


def _flatten_critic(agent):
    """把 critic 压成常数函数，这样 advantage 就等于回报本身，断言才不掺 V 的噪声。"""
    for q in agent.critic.qs:
        torch.nn.init.zeros_(q.trunk[-1].weight)
        torch.nn.init.zeros_(q.trunk[-1].bias)


def test_awr_weights_track_the_advantage_and_ignore_truncated_rows():
    """回报高于状态价值的窗口权重更大；被截断的 episode（mc_valid=0）保持权重 1。"""
    agent = RLTAgent(_online_cfg(awr_beta=1.0, awr_weight_max=20.0), device="cpu")
    _flatten_critic(agent)
    n = 16
    batch = _batch(n, mc_return=torch.linspace(0.0, 1.0, n), mc_valid=torch.ones(n))

    w, _ = agent._awr_weights(batch)
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5), "权重归一化到均值 1"
    assert torch.all(w[1:] >= w[:-1] - 1e-6), "权重必须随 advantage 单调不减"
    assert w[-1] > w[0]

    batch["mc_valid"] = torch.zeros(n)
    w, _ = agent._awr_weights(batch)
    assert torch.allclose(w, torch.ones(n)), "没有可用回报时不该有任何偏好"


def test_awr_weights_are_clipped():
    """一条离群的高回报窗口不该单独主导整个 batch。"""
    agent = RLTAgent(_online_cfg(awr_beta=5.0, awr_weight_max=3.0), device="cpu")
    _flatten_critic(agent)
    n = 16
    mc = torch.zeros(n)
    mc[0] = 50.0
    w, _ = agent._awr_weights(_batch(n, mc_return=mc, mc_valid=torch.ones(n)))
    # 裁剪发生在归一化之前，所以裁剪后的最大值 / 最小值必须落在 max/min 的比值内
    assert w.max() / w.min() <= 3.0 / torch.exp(torch.tensor(-5.0)) + 1e-3


def test_awr_pulls_the_actor_toward_high_advantage_actions():
    """AWR 项存在时，actor 朝高优势窗口的执行动作靠得比朝低优势窗口更近。"""
    torch.manual_seed(0)
    ac = _ac_cfg(action_std=0.0, ref_dropout=0.0, residual_scale=0.5, hidden_dim=64)
    cfg = _online_cfg(ac=ac, awr_weight=20.0, online_bc_weight=0.0, online_q_weight=0.0,
                      weight_ramp_updates=0, actor_update_period=1)
    agent = RLTAgent(cfg, device="cpu")
    _flatten_critic(agent)
    n = 16
    mc = torch.zeros(n)
    mc[: n // 2] = 1.0  # 前一半是成功的 episode
    batch = _batch(n, mc_return=mc, mc_valid=torch.ones(n))
    # 目标必须落在残差盒里，否则每一行都顶到边界，权重就看不出差别了
    batch["action"] = batch["ref"] + 0.1 * torch.randn(n, C, D)
    # 50 步：再多这个 MLP 就把 16 行全背下来了，两组误差一起归零，权重的作用也就
    # 看不出来了。AWR 影响的是收敛的先后，不是最终能不能拟合。
    for _ in range(50):
        agent.update_actor(batch, bc_weight=0.0, q_weight=0.0)
    with torch.no_grad():
        err = (agent.actor.mu(batch["x"], batch["ref"]) - batch["action"]).pow(2).mean(dim=(1, 2))
    assert err[: n // 2].mean() < err[n // 2 :].mean(), "高优势窗口应该被模仿得更准"


def test_online_weights_are_ramped_in_rather_than_switched():
    """warmup -> online 是线性过渡；一步换权重会把 actor 直接从 VLA 流形上甩出去。"""
    cfg = _online_cfg(
        actor_update_period=1,
        warmup_bc_weight=10.0, warmup_q_weight=0.1,
        online_bc_weight=0.0, online_q_weight=1.1,
        weight_ramp_updates=10,
    )
    agent = RLTAgent(cfg, device="cpu")
    batch = _batch(8)
    agent.update(batch, warmup=True)
    seen = [agent.update(batch, warmup=False)["bc_weight"] for _ in range(12)]
    assert seen[0] == pytest.approx(10.0)
    assert seen[-1] == pytest.approx(0.0)
    assert all(b <= a + 1e-6 for a, b in zip(seen, seen[1:], strict=False)), "权重必须单调下降"


def test_actor_keeps_training_during_warmup():
    """warmup 只换权重对，不冻结 actor——冻结会连带停掉 target 的 EMA。"""
    agent = RLTAgent(_online_cfg(actor_update_period=1), device="cpu")
    before = copy.deepcopy(agent.actor.state_dict())
    agent.update(_batch(8), warmup=True)
    assert any(not torch.equal(v, before[k]) for k, v in agent.actor.state_dict().items())


def test_agent_state_dict_roundtrips_both_targets():
    agent = RLTAgent(_online_cfg(), device="cpu")
    agent.update(_batch(8))
    restored = RLTAgent(_online_cfg(), device="cpu")
    restored.load_state_dict(agent.state_dict())
    for name in ("actor", "critic", "actor_target", "critic_target"):
        a, b = getattr(agent, name).state_dict(), getattr(restored, name).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
