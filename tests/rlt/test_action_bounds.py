"""动作限幅：按数据分位数逐维设界，而不是在归一化空间里砍一刀 ±1。

会咬人的两件事：一是 SmolVLA 用 MEAN_STD 归一化，±1 只是 1 个标准差，会削掉
1/4 的动作；二是饱和通道（LIBERO 的夹爪只有 ±1 两个值）的分位数正好落在数据上，
界必须留出余量，否则探索噪声每一步都被单向压回去。
"""

import numpy as np
import pytest
import torch

from lerobot.policies.rlt import ActionBounds, ChunkActor, fit_action_bounds
from lerobot.policies.rlt.action_bounds import BOUNDS_FILENAME, DEFAULT_MARGIN
from lerobot.policies.rlt.configuration_rlt import ActorCriticConfig

C, D = 4, 3


def _cfg(**kw):
    base = {"chunk_len": C, "action_dim": D, "proprio_dim": 2, "rl_token_dim": 6, "hidden_dim": 8}
    base.update(kw)
    return ActorCriticConfig(**base)


# ------------------------------------------------------------------ 数学
def test_quantiles_map_into_the_vla_normalized_space():
    """界的单位必须和 VLA 的归一化动作一致，否则和参考 chunk 不可比。"""
    mean, std = np.array([0.0, 1.0]), np.array([2.0, 4.0])
    b = ActionBounds.from_quantiles(
        q_low=np.array([-2.0, -3.0]), q_high=np.array([2.0, 5.0]), mean=mean, std=std, margin=1.0 + 1e-9
    )
    # (±2 - 0)/2 = ±1 ; (-3-1)/4 = -1, (5-1)/4 = 1
    assert torch.allclose(b.low, torch.tensor([-1.0, -1.0]), atol=1e-5)
    assert torch.allclose(b.high, torch.tensor([1.0, 1.0]), atol=1e-5)


def test_bounds_are_asymmetric_when_the_data_is():
    """动作分布通常不对称，对称界会在一侧浪费、另一侧砍掉真实指令。"""
    b = ActionBounds.from_quantiles(
        q_low=np.array([-0.1]), q_high=np.array([0.9]), mean=np.array([0.0]), std=np.array([1.0])
    )
    assert b.low.item() < 0 < b.high.item()
    assert abs(b.high.item()) > abs(b.low.item())


def test_margin_of_one_is_refused():
    with pytest.raises(ValueError, match="margin must exceed"):
        ActionBounds.from_quantiles(
            np.array([-1.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]), margin=1.0
        )


def test_saturated_channel_keeps_headroom():
    """夹爪只有 ±1 两个值，界压在数据上会把探索噪声整条单向削回去。"""
    grip = np.where(np.arange(1000) % 2 == 0, 1.0, -1.0).reshape(-1, 1)
    b = fit_action_bounds(grip, mean=np.array([0.0]), std=np.array([1.0]))
    assert b.low.item() < -1.0 and b.high.item() > 1.0
    normalized = torch.tensor(grip, dtype=torch.float32)
    assert torch.equal(b.clamp(normalized), normalized), "饱和值一个都不该被削"


def test_quantiles_ignore_outliers_that_min_max_would_not():
    data = np.concatenate([np.zeros((998, 1)), np.full((2, 1), 500.0)])
    b = fit_action_bounds(data, mean=np.array([0.0]), std=np.array([1.0]))
    assert b.high.item() < 10.0, "单帧离群点不该把界撑开"


def test_fit_rejects_a_non_2d_action_array():
    with pytest.raises(ValueError, match="expected"):
        fit_action_bounds(np.zeros((4, 5, 3)), np.zeros(3), np.ones(3))


def test_high_must_exceed_low():
    with pytest.raises(ValueError, match="must exceed its low"):
        ActionBounds(low=torch.ones(2), high=torch.zeros(2))


# ------------------------------------------------------------- 存取 / 配置
def test_save_load_roundtrip(tmp_path):
    b = ActionBounds(low=torch.tensor([-2.0, -0.5]), high=torch.tensor([1.5, 3.0]))
    b.save(tmp_path)
    assert (tmp_path / BOUNDS_FILENAME).is_file()
    back = ActionBounds.load(tmp_path)
    assert torch.allclose(back.low, b.low) and torch.allclose(back.high, b.high)


def test_load_returns_none_when_absent(tmp_path):
    assert ActionBounds.load(tmp_path) is None


def test_from_config_falls_back_to_the_scalar_clip():
    b = ActionBounds.from_config(_cfg(action_clip=2.0))
    assert torch.equal(b.low, torch.full((D,), -2.0))
    assert torch.equal(b.high, torch.full((D,), 2.0))


def test_from_config_prefers_fitted_bounds():
    cfg = _cfg(action_clip=1.0, action_bounds=[[-3.0, -1.0, -2.0], [4.0, 1.0, 2.0]])
    b = ActionBounds.from_config(cfg)
    assert torch.equal(b.high, torch.tensor([4.0, 1.0, 2.0]))


# --------------------------------------------------------------- 接入 actor
def _actor_inputs(batch=2):
    x = torch.zeros(batch, 6 + 2)
    ref = torch.zeros(batch, C, D)
    return x, ref


def test_actor_clamps_each_dimension_against_its_own_bound():
    cfg = _cfg(action_bounds=[[-3.0, -1.0, -0.2], [3.0, 1.0, 0.2]], max_residual=0.0)
    actor = ChunkActor(cfg)
    x, ref = _actor_inputs()
    ref = ref + torch.tensor([10.0, 10.0, 10.0])
    out = actor.mu(x, ref)
    assert torch.allclose(out[..., 0], torch.tensor(3.0))
    assert torch.allclose(out[..., 1], torch.tensor(1.0))
    assert torch.allclose(out[..., 2], torch.tensor(0.2))


def test_actor_leaves_in_range_references_untouched():
    """零初始化输出层 => 第一步 RL 策略必须逐位等于 base VLA。"""
    cfg = _cfg(action_bounds=[[-5.0] * D, [5.0] * D])
    actor = ChunkActor(cfg)
    x, _ = _actor_inputs()
    # 全部超出 ±1，但都在拟合出的界内——正是被旧的标量 clip 削掉的那部分
    ref = torch.linspace(-4.5, 4.5, 2 * C * D).reshape(2, C, D)
    assert ref.abs().max() > 1.0
    assert torch.equal(actor.mu(x, ref), ref)


def test_a_scalar_clip_would_have_truncated_that_reference():
    """对照组：没有这次改动时，同一条参考会被砍掉。"""
    actor = ChunkActor(_cfg(action_clip=1.0))
    x, _ = _actor_inputs()
    ref = torch.full((2, C, D), 2.5)
    assert torch.allclose(actor.mu(x, ref), torch.tensor(1.0))


def test_bounds_travel_in_the_state_dict():
    """ActorMirror 和 learner 子进程都是「裸 config 建 actor + load_state_dict」。"""
    cfg = _cfg(action_bounds=[[-3.0, -1.0, -0.2], [3.0, 1.0, 0.2]])
    source = ChunkActor(cfg)
    mirror = ChunkActor(_cfg())  # 回退到 ±1
    assert not torch.equal(mirror.action_high, source.action_high)
    mirror.load_state_dict(source.state_dict())
    assert torch.equal(mirror.action_high, source.action_high)
    assert torch.equal(mirror.action_low, source.action_low)


def test_actor_rejects_bounds_of_the_wrong_width():
    with pytest.raises(ValueError, match="acts in"):
        ChunkActor(_cfg(), bounds=ActionBounds.symmetric(D + 1))


# ------------------------------------------------------------ stage-2 入口
def test_train_online_refuses_to_start_without_bounds(tmp_path):
    """静默回退会让机械臂少走 1/4，不报错——所以宁可开跑前就停。"""
    from lerobot.rlt.train_online import load_action_bounds

    with pytest.raises(FileNotFoundError, match="lerobot-rlt-fit-action-bounds"):
        load_action_bounds(tmp_path, action_dim=D)


def test_train_online_rejects_bounds_of_the_wrong_width(tmp_path):
    ActionBounds.symmetric(D).save(tmp_path)
    from lerobot.rlt.train_online import load_action_bounds

    with pytest.raises(ValueError, match="re-fit"):
        load_action_bounds(tmp_path, action_dim=D + 2)


def test_default_margin_leaves_real_headroom():
    assert DEFAULT_MARGIN > 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
