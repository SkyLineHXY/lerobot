"""L1 集成测试：不加载 SmolVLA，用桩控制器跑通完整的 rollout + 异步 learner 闭环。

覆盖 `train_online.py` 里真正容易出错、但 L0 单测碰不到的部分：
RolloutWorker 的 chunk 组装与中间状态补算、干预路径、learner 线程的 UTD 节奏与
参数发布、以及 critical-phase 交接。真机上这些环节出错的代价最高。
"""

import time

import torch

from lerobot.policies.rlt import ActorCriticConfig, OnlineRLConfig, RLTAgent
from lerobot.rlt.envs.mock import MockManipEnv
from lerobot.rlt.learner import ActorMirror, LearnerThread
from lerobot.rlt.replay_buffer import ChunkReplayBuffer
from lerobot.rlt.rollout import RolloutWorker
from lerobot.rlt.teleop.base import InterventionResult

C, D, PROPRIO, TOKEN = 4, 6, 6, 8
X_DIM = TOKEN + PROPRIO


def _cfg(**kw):
    ac = ActorCriticConfig(
        chunk_len=C, action_dim=D, proprio_dim=PROPRIO, rl_token_dim=TOKEN, hidden_dim=16
    )
    base = dict(ac=ac, device="cpu", batch_size=8, subsample_stride=2, utd=2)
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

    def compute_x(self, batch):
        self.n_compute_x += 1
        return self._x(batch)

    def plan_chunk(self, batch, use_actor=True, deterministic=False):
        self.n_plans += 1
        x = self._x(batch)
        b = x.shape[0]
        ref_full = torch.full((b, 2 * self.chunk_len, self.action_dim), 0.25)
        ref = ref_full[:, : self.chunk_len]
        action = (
            self.agent.act(x, ref, deterministic=deterministic)
            if (use_actor and self.agent is not None)
            else ref
        )
        return {"x": x, "ref_full": ref_full, "action_chunk": action}


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


def test_rollout_fills_buffer_with_aligned_transitions():
    cfg, _env, buffer, controller, worker = _make()
    worker.reset()
    for _ in range(6):
        worker.run_chunk(use_actor=False)

    assert len(buffer) > 0
    # 每个 chunk 只规划一次；中间 offset 的状态是批量补算的，不占控制回路
    assert controller.n_plans == 6
    n = len(buffer)
    assert torch.isfinite(buffer.x[:n]).all()
    assert torch.isfinite(buffer.x_next[:n]).all()
    assert (buffer.actual_steps[:n] > 0).all()
    assert (buffer.actual_steps[:n] <= C).all()
    # 非终止样本的 x_next 必须是真实的后继状态，不能自指
    non_terminal = buffer.done[:n] == 0
    if non_terminal.any():
        same = (buffer.x[:n][non_terminal] == buffer.x_next[:n][non_terminal]).all(dim=-1)
        assert not same.all()


def test_intervention_replaces_both_action_and_reference():
    """人类修正必须同时替换执行动作与存储的参考，否则 BC 项会把 actor 拉回失败的 VLA 动作。"""
    env = ScriptedInterventionEnv(intervene_on={0}, action_dim=D, max_episode_steps=40)
    cfg, _e, buffer, _c, worker = _make(env=env)
    worker.reset()
    _n, _done, _succ, intervened = worker.run_chunk(use_actor=False)
    assert intervened
    worker.run_chunk(use_actor=False)  # 触发上一个 chunk 的 transition 落库

    assert len(buffer) > 0
    # 被接管的 chunk，其存储的参考动作整条都是人类动作 —— 这正是论文 Sec. V 的要求：
    # 干预替换的是 VLA 参考，从而让 BC 项把 actor 拉向人类修正。
    stored_ref = buffer.ref[: len(buffer)]
    assert torch.allclose(stored_ref, torch.full_like(stored_ref, env.human_value))
    # offset 0 的动作窗口完全落在被接管的 chunk 内；offset>0 的窗口按定义横跨
    # 下一个 chunk，因此是人类动作与策略动作的拼接，不应整条都是人类值。
    assert torch.allclose(buffer.action[0], torch.full_like(buffer.action[0], env.human_value))


def test_critical_phase_defers_rl_until_handover():
    cfg, _env, _buf, _ctrl, worker = _make()
    worker.reset(critical_phase=True)
    assert not worker.rl_engaged  # base VLA 先跑
    worker.rl_engaged = True  # 操作员按下 `r`
    worker.reset(critical_phase=False)
    assert worker.rl_engaged


def test_learner_thread_updates_and_publishes_weights():
    cfg, _env, buffer, _ctrl, worker = _make()
    worker.reset()
    for _ in range(8):
        worker.run_chunk(use_actor=False)
    assert len(buffer) >= cfg.batch_size

    agent = RLTAgent(cfg, device="cpu")
    learner = LearnerThread(agent, buffer, cfg, idle_sleep_s=0.001)
    mirror = ActorMirror(cfg, "cpu")
    mirror.sync(learner)
    v0 = mirror.version

    learner.allow_actor_updates = True
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
    worker.reset()
    for _ in range(8):
        worker.run_chunk(use_actor=False)

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


def test_warmup_holds_the_actor_at_the_vla_reference():
    cfg, _env, buffer, _ctrl, worker = _make()
    worker.reset()
    for _ in range(8):
        worker.run_chunk(use_actor=False)

    agent = RLTAgent(cfg, device="cpu")
    before = [p.detach().clone() for p in agent.actor.parameters()]
    learner = LearnerThread(agent, buffer, cfg, idle_sleep_s=0.001)
    learner.allow_actor_updates = False  # warmup 期：只训 critic

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
    for b, a in zip(before, after, strict=True):
        assert torch.allclose(b, a), "warmup 期间 actor 不应被更新"


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
    env = FailureTerminatingEnv(fail_at=3, action_dim=D, max_episode_steps=40)
    _cfg_, _e, buffer, _c, worker = _make(env=env)
    worker.reset()
    n_exec, ep_done, ep_success, _interv = worker.run_chunk(use_actor=False)

    assert n_exec == 3, "失败键应立即结束当前 chunk"
    assert ep_done, "失败同样结束 episode"
    assert not ep_success, "reward 为 0 的终止不能被算作成功"
    # 失败仍是终止态（后续无奖励可得），所以要屏蔽 bootstrap
    n = len(buffer)
    assert n > 0
    assert (buffer.done[:n] == 1).all()
    assert (buffer.reward_disc[:n] == 0).all()


def test_discard_episode_drops_the_whole_episode():
    _cfg_, _env, buffer, _c, worker = _make()
    worker.reset()
    for _ in range(4):
        worker.run_chunk(use_actor=False)
    assert len(buffer) > 0

    before_total = buffer.total_added
    dropped = buffer.discard_episode()
    assert dropped == before_total
    assert len(buffer) == 0, "`←` 应丢掉整个 episode，而不只是未配对的那个 chunk"
    assert buffer.total_added == 0

    # 丢弃后仍能继续正常写入
    worker.reset()
    for _ in range(4):
        worker.run_chunk(use_actor=False)
    assert len(buffer) > 0


def test_mock_env_can_be_closed():
    """train() 的 finally 无条件调 env.close()；缺这个方法会盖掉真正的异常。"""
    env = MockManipEnv(action_dim=D, max_episode_steps=40)
    env.reset()
    env.close()
