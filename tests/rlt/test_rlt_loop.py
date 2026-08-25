"""L1 集成测试：不加载 SmolVLA，用桩控制器跑通完整的 rollout + 异步 learner 闭环。

覆盖 `train_online.py` 里真正容易出错、但 L0 单测碰不到的部分：
RolloutWorker 的 chunk 组装与中间状态补算、干预路径、learner 线程的 UTD 节奏与
参数发布、以及 critical-phase 交接。真机上这些环节出错的代价最高。
"""

import time

import pytest
import torch

from lerobot.policies.rlt import (
    ActorCriticConfig,
    OnlineRLConfig,
    RLTAgent,
    TransitionSource,
)
from lerobot.rlt.envs.mock import MockManipEnv
from lerobot.rlt.learner import ActorMirror, LearnerThread
from lerobot.rlt.replay_buffer import ChunkReplayBuffer
from lerobot.rlt.rollout import RolloutWorker
from lerobot.rlt.teleop.base import InterventionResult

C, D, PROPRIO, TOKEN = 4, 6, 6, 8
X_DIM = TOKEN + PROPRIO


def _cfg(residual_scale=0.0, **kw):
    ac = ActorCriticConfig(
        chunk_len=C,
        action_dim=D,
        proprio_dim=PROPRIO,
        rl_token_dim=TOKEN,
        hidden_dim=16,
        residual_scale=residual_scale,
    )
    base = {"ac": ac, "device": "cpu", "batch_size": 8, "subsample_stride": 2, "utd": 2}
    base.update(kw)
    return OnlineRLConfig(**base)


class StubController:
    """RLTController 的替身：RL 状态和参考 chunk 都由观测确定性地导出。

    保留真实控制器的接口（chunk_len / compute_x / plan_chunk / rl_token），
    这样 RolloutWorker 走的是和真机完全相同的代码路径。
    """

    def __init__(self, cfg, agent=None):
        self.chunk_len = cfg.ac.chunk_len
        self.action_dim = cfg.ac.action_dim
        self.agent = agent
        self.rl_token = torch.nn.Linear(1, 1)  # 只为 next(...).device 提供设备
        self.n_plans = 0
        self.n_compute_x = 0

    def _x(self, batch):
        state = batch["observation.state"]
        b = state.shape[0]
        return torch.cat([state[:, :1].expand(b, TOKEN), state[:, :PROPRIO]], dim=-1)

    def compute_x(self, batch, with_ref=False):
        self.n_compute_x += 1
        x = self._x(batch)
        if not with_ref:
            return x
        return x, self._ref_full(x.shape[0])

    def _ref_full(self, b):
        return torch.full((b, 2 * self.chunk_len, self.action_dim), 0.25)

    def plan_chunk(self, batch, use_actor=True, deterministic=False):
        self.n_plans += 1
        x = self._x(batch)
        b = x.shape[0]
        ref_full = self._ref_full(b)
        ref = ref_full[:, : self.chunk_len]
        action = (
            self.agent.act(x, ref, deterministic=deterministic)
            if (use_actor and self.agent is not None)
            else ref
        )
        return {"x": x, "ref_full": ref_full, "action_chunk": action}


class DriftingPlanController(StubController):
    """A controller whose plan is derived from the observation, like a real VLA.

    `StubController` hands back the same constant chunk forever, so a stale
    slice of an old plan and a freshly sampled one are numerically identical —
    which is precisely why the mislabelled-reference bug survived the suite.
    Here the plan is `seed(x) + [0, 1, 2, ...]`, so shifting an old plan forward
    by k gives `seed(x_0) + k` where a fresh sample gives `seed(x_k)`.
    """

    def _plan(self, x):
        steps = torch.arange(2 * self.chunk_len, dtype=torch.float32).view(1, -1, 1)
        return (x[:, :1].unsqueeze(-1) + steps).expand(-1, -1, self.action_dim)

    def compute_x(self, batch, with_ref=False):
        self.n_compute_x += 1
        x = self._x(batch)
        return (x, self._plan(x)) if with_ref else x

    def plan_chunk(self, batch, use_actor=True, deterministic=False):
        self.n_plans += 1
        x = self._x(batch)
        ref_full = self._plan(x)
        ref = ref_full[:, : self.chunk_len]
        action = (
            self.agent.act(x, ref, deterministic=deterministic)
            if (use_actor and self.agent is not None)
            else ref
        )
        return {"x": x, "ref_full": ref_full, "action_chunk": action}


def test_every_anchor_stores_the_reference_belonging_to_its_own_state():
    """存下来的参考必须是控制器在该 anchor 的状态上会给出的那一份。

    子采样出来的 anchor 曾经复用 chunk 边界的计划、只做平移，于是 80% 的
    transition 的参考是 VLA 的开环续写，而 `action` 里存的是下一次重新规划的结果
    ——在真实 buffer 上两者逐元素差 0.41（参考本身的 std 只有 0.9）。
    """
    cfg = _cfg()
    env = MockManipEnv(action_dim=D, max_episode_steps=40)
    buffer = ChunkReplayBuffer(
        capacity=256,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=cfg.discount,
        stride=cfg.subsample_stride,
        device="cpu",
    )
    worker = RolloutWorker(env, DriftingPlanController(cfg), buffer, cfg.subsample_stride)
    _run_episode(worker)

    n = len(buffer)
    assert n > 0
    # `_x` 把 state[:, 0] 放在 x 的第 0 维，计划的第一步就等于它。
    seed = buffer.x[:n][:, :1].expand(-1, D)
    assert torch.allclose(buffer.ref[:n][:, 0, :], seed)
    assert torch.allclose(buffer.ref_next[:n][:, 0, :], buffer.x_next[:n][:, :1].expand(-1, D))


def test_only_chunk_boundary_anchors_are_valid_imitation_targets():
    """基线 rollout 执行的就是参考本身，所以 aligned 行必须是恒等映射。"""
    cfg = _cfg()
    _c, _e, buffer, _ctrl, worker = _make(cfg=cfg)
    _run_episode(worker)

    n = len(buffer)
    aligned = buffer.aligned[:n] > 0
    assert aligned.float().mean() == pytest.approx(cfg.subsample_stride / C)
    assert torch.allclose(buffer.action[:n][aligned], buffer.ref[:n][aligned])


class ScriptedInterventionEnv(MockManipEnv):
    """第 n 个 chunk 由"人类"接管，用于验证干预路径。"""

    def __init__(self, intervene_on=(1,), **kw):
        super().__init__(**kw)
        self.intervene_on = set(intervene_on)
        self.chunk_idx = 0
        self.human_value = 0.75

    def run_intervention(self, chunk_len):
        idx, self.chunk_idx = self.chunk_idx, self.chunk_idx + 1
        if idx not in self.intervene_on:
            return None
        obs_list, rewards = [], []
        for _ in range(chunk_len):
            o, r, _done = self.step(torch.full((self.action_dim,), self.human_value))
            obs_list.append(o)
            rewards.append(r)
        return InterventionResult(
            action_chunk=torch.full((chunk_len, self.action_dim), self.human_value),
            obs_list=obs_list,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            n_steps=chunk_len,
        )


def _make(env=None, cfg=None):
    cfg = cfg or _cfg()
    env = env or MockManipEnv(action_dim=D, max_episode_steps=40)
    buffer = ChunkReplayBuffer(
        capacity=256,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=cfg.discount,
        stride=cfg.subsample_stride,
        device="cpu",
    )
    controller = StubController(cfg)
    worker = RolloutWorker(env, controller, buffer, cfg.subsample_stride)
    return cfg, env, buffer, controller, worker


def _run_episode(worker, max_chunks=64) -> int:
    """Run one episode to its end and wait for its records to land.

    Transitions are emitted when the episode ends, and the assembler thread is
    what emits them, so the drain is part of the contract — not test noise.
    """
    worker.reset()
    for i in range(max_chunks):
        _n, done, _succ, _interv = worker.run_chunk(use_actor=False)
        if done:
            worker.drain()
            return i + 1
    raise AssertionError("episode never ended")


def test_rollout_fills_buffer_with_aligned_transitions():
    cfg, _env, buffer, controller, worker = _make()
    chunks = _run_episode(worker)

    assert len(buffer) > 0
    # 每个 chunk 只规划一次；中间 offset 的状态是批量补算的，不占控制回路
    assert controller.n_plans == chunks
    n = len(buffer)
    assert torch.isfinite(buffer.x[:n]).all()
    assert torch.isfinite(buffer.x_next[:n]).all()
    # 非终止样本的 x_next 必须是真实的后继状态，不能自指
    non_terminal = buffer.done[:n] == 0
    if non_terminal.any():
        same = (buffer.x[:n][non_terminal] == buffer.x_next[:n][non_terminal]).all(dim=-1)
        assert not same.all()


def test_intervention_preserves_reference_and_records_executed_human_action():
    """接管不改变 actor 条件输入；human action 单独保存在行为 target 中。"""
    env = ScriptedInterventionEnv(intervene_on={0}, action_dim=D, max_episode_steps=40)
    _cfg_, _e, buffer, _c, worker = _make(env=env)
    _run_episode(worker)

    n = len(buffer)
    assert n > 0
    human = buffer.source_chunk[:n] == int(TransitionSource.HUMAN)
    assert human.any(), "被接管的步要打上 HUMAN 标签"
    # HUMAN 步的执行动作是人的命令，但条件 reference 仍是 VLA 的 0.25。
    assert torch.allclose(buffer.action[:n][human], torch.tensor(env.human_value))
    assert torch.allclose(buffer.ref[:n], torch.tensor(0.25))


class ProbingInterventionEnv(ScriptedInterventionEnv):
    """记录 rollout 一共问了几次"人要接管吗"，以及真的交出去几次。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_pending_polls = 0

    def intervention_pending(self):
        self.n_pending_polls += 1
        return True


def test_warmup_never_asks_for_or_accepts_a_takeover():
    """warmup 要让 critic 学冻结 VLA 的价值，人开的那一段不属于这个分布。"""
    env = ProbingInterventionEnv(intervene_on=set(range(8)), action_dim=D, max_episode_steps=40)
    _cfg, _e, _buf, _c, worker = _make(env=env)
    worker.allow_intervention = False
    worker.reset()
    for _ in range(4):
        _n, _done, _succ, intervened = worker.run_chunk(use_actor=False)
        assert not intervened

    assert env.n_pending_polls == 0, "warmup 期间连问都不该问"
    assert env.chunk_idx == 0, "run_intervention 一次都不该被调用"


def test_takeover_works_again_once_warmup_is_over():
    env = ProbingInterventionEnv(intervene_on={0}, action_dim=D, max_episode_steps=40)
    _cfg, _e, _buf, _c, worker = _make(env=env)
    worker.allow_intervention = False
    worker.reset()
    worker.run_chunk(use_actor=False)

    worker.allow_intervention = True
    _n, _done, _succ, intervened = worker.run_chunk(use_actor=False)
    assert intervened
    assert env.chunk_idx > 0, "run_intervention 必须被调用"


def test_critical_phase_defers_rl_until_handover():
    cfg, _env, _buf, _ctrl, worker = _make()
    worker.reset(critical_phase=True)
    assert not worker.rl_engaged  # base VLA 先跑
    worker.rl_engaged = True  # 操作员按下 `r`
    worker.reset(critical_phase=False)
    assert worker.rl_engaged


def test_learner_thread_updates_and_publishes_weights():
    cfg, _env, buffer, _ctrl, worker = _make()
    for _ in range(2):
        _run_episode(worker)
    assert len(buffer) >= cfg.batch_size

    agent = RLTAgent(cfg, device="cpu")
    learner = LearnerThread(agent, buffer, cfg, idle_sleep_s=0.001)
    mirror = ActorMirror(cfg, "cpu")
    mirror.sync(learner)
    v0 = mirror.version

    learner.warmup = False
    learner.start()
    try:
        deadline = time.time() + 10.0
        while learner.updates < 20 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        learner.stop()
        learner.join(timeout=5.0)

    assert learner.updates >= 20, f"learner only did {learner.updates} updates"
    m = learner.metrics()
    assert "critic_loss" in m and m["critic_loss"] >= 0
    # actor 更新过后，mirror 应拉到新版本的权重
    assert mirror.sync(learner)
    assert mirror.version > v0


def test_learner_paces_itself_by_the_update_to_data_ratio():
    """UTD 按真实入库的 transition 数计，而不是按名义 chunk 长度。"""
    cfg, _env, buffer, _ctrl, worker = _make(cfg=_cfg(utd=3))
    for _ in range(2):
        _run_episode(worker)

    agent = RLTAgent(cfg, device="cpu")
    learner = LearnerThread(agent, buffer, cfg, idle_sleep_s=0.001)
    expected = cfg.utd * buffer.total_added

    learner.start()
    try:
        deadline = time.time() + 15.0
        while learner.updates < expected and time.time() < deadline:
            time.sleep(0.05)
    finally:
        learner.stop()
        learner.join(timeout=5.0)

    # 不再有新数据后，更新次数应停在 utd * 已入库 transition 数
    assert learner.updates == expected, f"{learner.updates} != {expected}"


def test_warmup_grinds_past_the_utd_budget_and_then_stops():
    """UTD 会把 20000 步 warmup 拖成一场采集；参考实现是就地把已有数据磨够。"""
    target = 200
    cfg = _cfg(utd=1, warmup_post_collect_updates=target)
    cfg, _env, buffer, _ctrl, worker = _make(cfg=cfg)
    _run_episode(worker)
    assert cfg.utd * buffer.total_added < target, "UTD 额度本来就该不够，否则测不到东西"

    agent = RLTAgent(cfg, device="cpu")
    learner = LearnerThread(agent, buffer, cfg, idle_sleep_s=0.001)
    learner.warmup = True

    learner.start()
    try:
        deadline = time.time() + 20.0
        while learner.updates < target and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.3)  # 给它机会越过目标，确认它真的停住了
    finally:
        learner.stop()
        learner.join(timeout=5.0)

    assert learner.updates == target, f"{learner.updates} != {target}"
    # UTD 额度照常花掉，补足的那部分不额外记账——否则 online 阶段一开始就欠着
    assert learner._consumed == buffer.total_added


def test_warmup_trains_the_actor_on_the_warmup_weight_pair():
    """warmup 不冻结 actor，只换 BC/Q 权重对——冻结会连带停掉 target 的 EMA。

    `residual_scale=0` 才测得出来：残差参数化的 actor 出厂就等于 VLA 参考，而
    warmup 数据里执行的正是那份参考，所以 BC 项在初始点上恰好为 0，没有梯度可看。
    """
    cfg = _cfg(
        residual_scale=0.0, warmup_bc_weight=10.0, warmup_q_weight=0.0, actor_update_period=1
    )
    cfg, _env, buffer, _ctrl, worker = _make(cfg=cfg)
    for _ in range(2):
        _run_episode(worker)

    agent = RLTAgent(cfg, device="cpu")
    before = [p.detach().clone() for p in agent.actor.parameters()]
    learner = LearnerThread(agent, buffer, cfg, idle_sleep_s=0.001)
    learner.warmup = True

    learner.start()
    try:
        deadline = time.time() + 10.0
        while learner.updates < 10 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        learner.stop()
        learner.join(timeout=5.0)

    assert learner.updates >= 10
    after = list(agent.actor.parameters())
    assert any(
        not torch.allclose(b, a) for b, a in zip(before, after, strict=True)
    ), "warmup 期间 actor 仍然要训练"
    assert learner.metrics()["bc_penalty"] > 0


class FailureTerminatingEnv(MockManipEnv):
    """第 n 步由"操作员"按失败键结束 episode：done=True 但 reward=0。

    这是真机 PiperChunkEnv 的语义（`f` 键终止但不给奖励），mock/libero 里
    done 恒等于 success，所以只有这个桩环境能暴露"失败被记成成功"的 bug。
    """

    def __init__(self, fail_at=3, **kw):
        super().__init__(**kw)
        self.fail_at = fail_at
        self.n = 0

    def step(self, action):
        obs, _reward, _success = super().step(action)
        self.n += 1
        return obs, 0.0, self.n >= self.fail_at


def test_failure_key_terminates_without_counting_as_success():
    env = FailureTerminatingEnv(fail_at=6, action_dim=D, max_episode_steps=40)
    _cfg_, _e, buffer, _c, worker = _make(env=env)
    worker.reset()
    worker.run_chunk(use_actor=False)
    n_exec, ep_done, ep_success, _interv = worker.run_chunk(use_actor=False)

    assert n_exec == 2, "失败键应立即结束当前 chunk"
    assert ep_done, "失败同样结束 episode"
    assert not ep_success, "reward 为 0 的终止不能被算作成功"
    # 失败仍是终止态（后续无奖励可得），所以覆盖到它的窗口要屏蔽 bootstrap
    worker.drain()
    n = len(buffer)
    assert n > 0
    assert (buffer.done[:n] == 1).any()
    assert (buffer.rewards[:n] == 0).all()
    assert (buffer.success[:n] == 0).all()


def test_discard_episode_drops_the_whole_episode():
    _cfg_, _env, buffer, _c, worker = _make()
    _run_episode(worker)
    stored, added = len(buffer), buffer.total_added
    assert stored > 0

    worker.reset()
    for _ in range(4):
        worker.run_chunk(use_actor=False)
    worker.drain()
    dropped = buffer.discard_episode()
    assert dropped == 4 * C, "`←` 应丢掉整个 episode 的 step trace"
    assert len(buffer) == stored, "已经落库的上一集不受影响"
    assert buffer.total_added == added

    # 丢弃后仍能继续正常写入
    _run_episode(worker)
    assert len(buffer) > stored


def test_mock_env_can_be_closed():
    """train() 的 finally 无条件调 env.close()；缺这个方法会盖掉真正的异常。"""
    env = MockManipEnv(action_dim=D, max_episode_steps=40)
    env.reset()
    env.close()


# ------------------------------------------------------- 边界处的异步装配
class TakeoverEnv(MockManipEnv):
    """操作员从第一个 chunk 起就一直握着控制权。

    和 `ScriptedInterventionEnv` 的区别只有一个，但正是被测的那个：
    `intervention_pending()` 为真，所以 rollout 应当把这一段的 VLA 采样推迟到
    装配线程上，而不是让人在边界上干等。
    """

    def __init__(self, human_value=0.75, **kw):
        super().__init__(**kw)
        self.human_value = human_value
        self.pending = True

    def intervention_pending(self):
        return self.pending

    def run_intervention(self, chunk_len):
        if not self.pending:
            return None
        obs_list, rewards = [], []
        done = truncated = False
        for _ in range(chunk_len):
            o, r, done, truncated = self.apply_action(
                torch.full((self.action_dim,), self.human_value)
            )
            obs_list.append(o)
            rewards.append(r)
            self.intervention.notify_step(o)
            if done or truncated:
                break
        n = len(obs_list)
        chunk = torch.full((n, self.action_dim), self.human_value)
        if n < chunk_len:  # 和真实 manager 一样，尾部保持最后一条人类指令
            chunk = torch.cat([chunk, chunk[-1:].expand(chunk_len - n, -1)], dim=0)
        rew = torch.zeros(chunk_len)
        rew[:n] = torch.tensor(rewards, dtype=torch.float32)
        return InterventionResult(
            action_chunk=chunk,
            obs_list=obs_list,
            rewards=rew,
            n_steps=n,
            done=done,
            truncated=truncated,
        )


class RefusingTakeoverEnv(TakeoverEnv):
    """人按了接管，但安全门拒绝了（真机上主臂被挪得太远）。"""

    def run_intervention(self, chunk_len):
        return None


BUFFER_COLUMNS = (
    "x", "action", "ref", "rewards", "x_next", "ref_next",
    "done", "mc_return", "mc_valid", "source_chunk", "source", "aligned", "success",
)


def _episode_columns(env_factory, *, async_records: bool):
    cfg = _cfg()
    env = env_factory()
    buffer = ChunkReplayBuffer(
        capacity=256,
        x_dim=X_DIM,
        chunk_len=C,
        action_dim=D,
        discount=cfg.discount,
        stride=cfg.subsample_stride,
        device="cpu",
    )
    worker = RolloutWorker(
        env,
        DriftingPlanController(cfg),
        buffer,
        cfg.subsample_stride,
        async_records=async_records,
    )
    _run_episode(worker)
    worker.stop()
    n = len(buffer)
    return n, {name: getattr(buffer, name)[:n].clone() for name in BUFFER_COLUMNS}


@pytest.mark.parametrize(
    "env_factory",
    [
        lambda: MockManipEnv(action_dim=D, max_episode_steps=40),
        lambda: ScriptedInterventionEnv(intervene_on={1}, action_dim=D, max_episode_steps=40),
        lambda: TakeoverEnv(action_dim=D, max_episode_steps=40),
    ],
    ids=["policy", "scripted-takeover", "held-takeover"],
)
def test_deferring_assembly_does_not_change_a_single_stored_value(env_factory):
    """把装配挪到后台线程只改"什么时候算"，不改算出来的东西。

    这是整个改动的安全网：控制回路上省下来的时间如果是拿存进 buffer 的内容换的，
    那 actor 的 BC 目标就被悄悄改了，而这在真机上要几百个 episode 之后才看得出来。
    """
    n_async, cols_async = _episode_columns(env_factory, async_records=True)
    n_sync, cols_sync = _episode_columns(env_factory, async_records=False)

    assert n_async == n_sync > 0
    for name in BUFFER_COLUMNS:
        assert torch.equal(cols_async[name], cols_sync[name]), name


def test_a_held_takeover_never_plans_on_the_control_thread():
    """人已经握着主臂时，边界上不该再有一次 VLM 前向 + flow-matching 采样。"""
    cfg = _cfg()
    env = TakeoverEnv(action_dim=D, max_episode_steps=40)
    buffer = ChunkReplayBuffer(
        capacity=256, x_dim=X_DIM, chunk_len=C, action_dim=D,
        discount=cfg.discount, stride=cfg.subsample_stride, device="cpu",
    )
    controller = DriftingPlanController(cfg)
    worker = RolloutWorker(env, controller, buffer, cfg.subsample_stride)
    _run_episode(worker)
    worker.stop()

    assert controller.n_plans == 0, "接管期间 plan_chunk 一次都不该被调用"
    assert controller.n_compute_x > 0, "但参考仍然要被采样并存下来"

    # 边界 anchor 的参考仍是"在该 anchor 的状态上重新采样"的那一份 —— 推迟计算
    # 不等于允许复用上一次的计划。
    n = len(buffer)
    assert n > 0
    assert torch.allclose(buffer.ref[:n][:, 0, :], buffer.x[:n][:, :1].expand(-1, D))


def test_a_refused_takeover_still_gets_a_plan():
    """安全门拒绝接管后策略要接着开，这时候计划是同步需要的。"""
    cfg = _cfg()
    env = RefusingTakeoverEnv(action_dim=D, max_episode_steps=40)
    buffer = ChunkReplayBuffer(
        capacity=256, x_dim=X_DIM, chunk_len=C, action_dim=D,
        discount=cfg.discount, stride=cfg.subsample_stride, device="cpu",
    )
    controller = StubController(cfg)
    worker = RolloutWorker(env, controller, buffer, cfg.subsample_stride)
    chunks = _run_episode(worker)
    worker.stop()

    assert controller.n_plans == chunks
    n = len(buffer)
    assert n > 0
    assert (buffer.source_chunk[:n] == int(TransitionSource.BASE)).all()


def test_an_assembler_failure_surfaces_on_the_control_thread():
    """装配线程静默死掉就是静默丢数据，必须炸在主线程上。"""
    cfg = _cfg()

    class ExplodingController(StubController):
        def compute_x(self, batch, with_ref=False):
            raise RuntimeError("boom")

    env = MockManipEnv(action_dim=D, max_episode_steps=40)
    buffer = ChunkReplayBuffer(
        capacity=256, x_dim=X_DIM, chunk_len=C, action_dim=D,
        discount=cfg.discount, stride=cfg.subsample_stride, device="cpu",
    )
    worker = RolloutWorker(env, ExplodingController(cfg), buffer, cfg.subsample_stride)
    worker.reset()
    worker.run_chunk(use_actor=False)
    with pytest.raises(RuntimeError, match="boom"):
        worker.drain()
    worker.stop()
