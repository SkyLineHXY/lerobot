"""Piper 采集脚本的无硬件单测。

采集脚本出错的代价是"人站在机械臂旁边、录了半天的数据不能用"，所以这里把三件事钉死：
按键状态机的语义、初始化的运动方向、以及写进 `add_frame` 的那一帧长什么样。
"""

import numpy as np
import pytest

from lerobot.rlt.piper_env import JOINT_ORDER
from lerobot.scripts import lerobot_record_piper as rp
from lerobot.scripts.lerobot_record_piper import (
    ArmPair,
    ArmSpec,
    CameraSpec,
    CollectionSpec,
    CollectKeys,
    PiperRecorder,
    RecordPiperConfig,
)


# ------------------------------------------------------------------ 按键状态机
class FakeBackend:
    name = "fake"

    def __init__(self):
        self.queued: list[str] = []

    def poll(self):
        out, self.queued = self.queued, []
        return out

    def stop(self):
        pass


def _keys(subtasks=("1", "2")):
    keys = CollectKeys.__new__(CollectKeys)
    keys._impl = FakeBackend()
    keys._subtask_keys = set(subtasks)
    keys.collecting = False
    keys.error_mode = False
    keys.quit = False
    keys._save = False
    keys._discard = False
    keys.subtask_key = None
    keys.messages = []
    return keys


def _press(keys, *presses):
    keys._impl.queued.extend(presses)
    keys.poll()


def test_c_starts_and_space_pauses():
    keys = _keys()
    _press(keys, "c")
    assert keys.collecting is True
    _press(keys, "space")
    assert keys.collecting is False


def test_pause_clears_error_mode():
    """错误模式不能跨暂停泄漏 —— 重新开始时操作员默认自己处于正常状态。"""
    keys = _keys()
    _press(keys, "c", "e")
    assert keys.error_mode is True
    _press(keys, "space")
    assert keys.error_mode is False


def test_error_mode_toggles():
    keys = _keys()
    _press(keys, "e")
    assert keys.error_mode is True
    _press(keys, "e")
    assert keys.error_mode is False


def test_subtask_only_switches_while_paused():
    """采集途中切子任务会让一集里混进两段标注却看不出边界。"""
    keys = _keys()
    _press(keys, "c", "2")
    assert keys.poll_subtask() is None
    assert any("暂停" in m for m in keys.messages)

    _press(keys, "space", "2")
    assert keys.poll_subtask() == "2"


def test_save_and_discard_are_one_shot():
    keys = _keys()
    _press(keys, "s", "r")
    assert keys.poll_save() is True
    assert keys.poll_save() is False
    assert keys.poll_discard() is True
    assert keys.poll_discard() is False


def test_quit_on_q_and_esc():
    for key in ("q", "esc"):
        keys = _keys()
        _press(keys, key)
        assert keys.quit is True


def test_unbound_keys_are_ignored():
    keys = _keys()
    _press(keys, "x", "f", "left")
    assert (keys.collecting, keys.error_mode, keys.quit) == (False, False, False)


# ------------------------------------------------------------------ 初始化
class FakeBus:
    """真实的 move_to_joint_smoothly 是同步阻塞的，返回时从臂已经到位。"""

    def __init__(self, robot):
        self.robot = robot
        self.moves: list[dict] = []

    def move_to_joint_smoothly(self, target, duration_s=None, hz=None, max_joint_step_rad=None):
        self.moves.append({"target": list(target), "duration_s": duration_s, "max_step": max_joint_step_rad})
        self.robot._joints = np.asarray(target, dtype=np.float32)


class FakeRobot:
    def __init__(self, joints):
        self._joints = np.asarray(joints, dtype=np.float32)
        self.bus = FakeBus(self)
        self.sent: list[np.ndarray] = []

    def get_observation(self):
        return dict(zip(JOINT_ORDER, self._joints.tolist(), strict=True))

    def send_action(self, action):
        self.sent.append(np.array([action[k] for k in JOINT_ORDER], dtype=np.float32))
        return action


class FakeLeader:
    def __init__(self, joints):
        self._joints = np.asarray(joints, dtype=np.float32)
        self.manual: list[bool] = []
        self.feedback: list[dict] = []

    def get_raw_action(self):
        keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
        return dict(zip(keys, self._joints.tolist(), strict=True))

    def set_manual_control(self, enabled):
        self.manual.append(bool(enabled))

    def send_feedback(self, feedback):
        self.feedback.append(feedback)
        # 主臂真的走过去了，这样 _wait_leader_settle 不会一直转到超时
        self._joints = np.array(
            [feedback[f"joint_{i}.pos"] for i in range(1, 7)] + [feedback["gripper.pos"]],
            dtype=np.float32,
        )


def _pair(leader_joints, follower_joints, **spec_kw):
    """绕开 __init__ 里的硬件构造，直接装配一个 ArmPair。"""
    spec = ArmSpec(**spec_kw)
    pair = ArmPair.__new__(ArmPair)
    pair.spec = spec
    pair.robot = FakeRobot(follower_joints)
    pair.leader = FakeLeader(leader_joints)
    pair.robot_cfg = type("Cfg", (), {"home_position": [0.0] * 7})()
    return pair


def test_follower_moves_to_the_leader_pose():
    """方向不能反：是从臂去找主臂，不是主臂去找从臂。"""
    leader = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07]
    pair = _pair(leader, [0.0] * 7)
    pair.initialize(CollectionSpec(init_mode="follower_to_leader", init_duration_s=1.0))

    assert len(pair.robot.bus.moves) == 1
    assert pair.robot.bus.moves[0]["target"] == pytest.approx(leader)
    # 主臂全程只被释放，从没被命令走位
    assert pair.leader.manual == [True]
    assert pair.leader.feedback == []


def test_home_mode_sends_both_arms_to_the_origin():
    pair = _pair([0.1] * 7, [0.2] * 7, max_takeover_delta_rad=0.0)
    pair.initialize(CollectionSpec(init_mode="home"))

    assert pair.robot.bus.moves[0]["target"] == pytest.approx([0.0] * 7)
    assert len(pair.leader.feedback) == 1, "主臂也要回原点"
    assert pair.leader.manual[-1] is True, "回到原点后必须松开主臂让人拖"


def test_none_mode_moves_nothing_but_still_releases_the_leader():
    pair = _pair([0.1] * 7, [0.2] * 7)
    pair.initialize(CollectionSpec(init_mode="none"))
    assert pair.robot.bus.moves == []
    assert pair.leader.manual == [True], "不释放主臂人根本拖不动它"


def test_initialize_uses_the_configured_step_limit():
    """单步上限是「手臂不猛地弹过去」的唯一保障，必须真的传下去。"""
    pair = _pair([0.5] * 7, [0.0] * 7, max_takeover_delta_rad=0.0)
    pair.initialize(
        CollectionSpec(init_mode="follower_to_leader", init_duration_s=3.0, init_max_step_rad=0.02)
    )
    move = pair.robot.bus.moves[0]
    assert move["duration_s"] == 3.0
    assert move["max_step"] == 0.02


def test_safety_gate_raises_when_the_arms_never_converge():
    """从臂没跟到位就进主循环，第一拍限速会把它猛地拽过去。"""

    class StuckBus(FakeBus):
        def move_to_joint_smoothly(self, target, **kw):
            self.moves.append({"target": list(target)})  # 记录了，但从臂并没有真的动

    pair = _pair([1.0] * 7, [0.0] * 7, max_takeover_delta_rad=0.15)
    pair.robot.bus = StuckBus(pair.robot)
    with pytest.raises(RuntimeError, match="初始化后主从仍相差"):
        pair.initialize(CollectionSpec(init_mode="follower_to_leader"))


def test_unknown_init_mode_is_rejected():
    pair = _pair([0.0] * 7, [0.0] * 7)
    with pytest.raises(ValueError, match="未知的 init_mode"):
        pair.initialize(CollectionSpec(init_mode="teleport"))


# ------------------------------------------------------------------ 帧构造
class FakeArmPair:
    """PiperRecorder 视角下的一条臂。"""

    def __init__(self, spec):
        self.spec = spec
        self.leader_joints = np.arange(7, dtype=np.float32) * 0.1
        self.follower_joints = np.zeros(7, dtype=np.float32)
        self.sent: list[np.ndarray] = []

    def read_leader(self):
        return dict(zip(JOINT_ORDER, self.leader_joints.tolist(), strict=True))

    def read_follower(self):
        return self.follower_joints.copy()

    def send(self, joints):
        self.sent.append(joints.copy())


def _recorder(monkeypatch, cfg):
    monkeypatch.setattr(rp, "ArmPair", FakeArmPair)
    rec = PiperRecorder(cfg)
    rec.keys = _keys()
    return rec


def _cfg(n_arms=1, cameras=(), subtasks=None, **collection_kw):
    return RecordPiperConfig(
        arms=[ArmSpec(name=f"a{i}") for i in range(n_arms)],
        dataset=rp.DatasetSpec(subtasks=subtasks if subtasks is not None else {"1": "grab"}),
        collection=CollectionSpec(**collection_kw),
        cameras=list(cameras),
    )


def test_frame_keys_match_the_schema_exactly(monkeypatch):
    """add_frame 要求键集合恰好等于 schema —— 多一个少一个都会被拒收。"""
    cams = [CameraSpec(name="cam_top", width=8, height=4)]
    rec = _recorder(monkeypatch, _cfg(cameras=cams))
    images = {"cam_top": np.zeros((4, 8, 3), dtype=np.uint8)}
    frame = rec.build_frame(np.zeros(7, np.float32), np.zeros(7, np.float32), images)

    expected = set(rec.build_features()) | {"task"}
    assert set(frame) == expected


def test_subtask_and_back_event_are_shape_1_int64(monkeypatch):
    rec = _recorder(monkeypatch, _cfg())
    rec.current_subtask = "grab"
    rec.keys.error_mode = True
    frame = rec.build_frame(np.zeros(7, np.float32), np.zeros(7, np.float32), {})

    for key in ("subtask_index", "back_event"):
        assert frame[key].dtype == np.int64
        assert frame[key].shape == (1,)
    assert frame["back_event"][0] == 1
    assert frame["subtask_index"][0] == 0


def test_no_subtasks_means_no_subtask_column(monkeypatch):
    rec = _recorder(monkeypatch, _cfg(subtasks={}))
    assert "subtask_index" not in rec.build_features()
    assert "subtask_index" not in rec.build_frame(np.zeros(7, np.float32), np.zeros(7, np.float32), {})


def test_images_are_passed_through_as_uint8(monkeypatch):
    """lerobot 的相机已经输出 RGB，不该再翻通道。"""
    cams = [CameraSpec(name="cam_top", width=3, height=2)]
    rec = _recorder(monkeypatch, _cfg(cameras=cams))
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    frame = rec.build_frame(np.zeros(7, np.float32), np.zeros(7, np.float32), {"cam_top": img})
    np.testing.assert_array_equal(frame["observation.images.cam_top"], img)


# ------------------------------------------------------------------ 双臂拼接
def test_dual_arm_joint_names_and_dims(monkeypatch):
    rec = _recorder(monkeypatch, _cfg(n_arms=2))
    assert rec.dof == 14
    names = rec.joint_names()
    assert names[0] == "joint_1.pos" and names[6] == "joint_7.pos"
    assert names[7] == "joint_8.pos" and names[13] == "joint_14.pos"
    assert rec.build_features()["observation.state"]["shape"] == (14,)


def test_dual_arm_state_keeps_left_before_right(monkeypatch):
    rec = _recorder(monkeypatch, _cfg(n_arms=2, max_joint_step_rad=10.0))
    rec.pairs[0].follower_joints = np.full(7, 1.0, dtype=np.float32)
    rec.pairs[1].follower_joints = np.full(7, 2.0, dtype=np.float32)

    state, _action = rec.step_arms()
    assert state.shape == (14,)
    assert state[:7] == pytest.approx([1.0] * 7)
    assert state[7:] == pytest.approx([2.0] * 7)


# ------------------------------------------------------------------ 限速
def test_action_records_the_rate_limited_target(monkeypatch):
    """记的必须是真正下发出去的目标，否则「动作 → 下一帧状态」在数据里对不上。"""
    rec = _recorder(monkeypatch, _cfg(max_joint_step_rad=0.01))
    rec.pairs[0].leader_joints = np.full(7, 1.0, dtype=np.float32)
    rec.pairs[0].follower_joints = np.zeros(7, dtype=np.float32)

    _state, action = rec.step_arms()
    assert action[:6] == pytest.approx([0.01] * 6)
    np.testing.assert_allclose(rec.pairs[0].sent[0], action)
    assert rec._saturated == [True]


def test_dry_run_sends_nothing(monkeypatch):
    rec = _recorder(monkeypatch, _cfg(dry_run=True, max_joint_step_rad=10.0))
    rec.step_arms()
    assert rec.pairs[0].sent == []


def test_saturation_is_not_flagged_when_within_the_limit(monkeypatch):
    rec = _recorder(monkeypatch, _cfg(max_joint_step_rad=10.0))
    rec.step_arms()
    assert rec._saturated == [False]


# ------------------------------------------------------------------ 配置校验
def test_empty_arms_is_rejected(monkeypatch):
    monkeypatch.setattr(rp, "ArmPair", FakeArmPair)
    with pytest.raises(ValueError, match="`arms` 不能为空"):
        PiperRecorder(RecordPiperConfig(arms=[]))


def test_duplicate_subtask_descriptions_are_rejected(monkeypatch):
    monkeypatch.setattr(rp, "ArmPair", FakeArmPair)
    cfg = _cfg(subtasks={"1": "same", "2": "same"})
    with pytest.raises(ValueError, match="子任务描述有重复"):
        PiperRecorder(cfg)
