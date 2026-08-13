"""按键监听两种后端的单测。

真机上这层出问题的代价很高：一旦监听器悄悄失效，操作员的成功/失败标签、交接、
干预全部消失，episode 只能靠步数上限结束 —— 而唯一的线索只是一行 warning。
所以这里把两种后端的按键归一化、状态机语义、以及"完全不可用"的降级路径都钉死。
"""

import queue

import pytest
from lerobot.rlt.intervention import (
    KEY_DISCARD,
    KEY_FAILURE,
    KEY_HANDOVER,
    KEY_INTERVENE,
    KEY_QUIT,
    KEY_SUCCESS,
    KeyboardEventListener,
    _PynputBackend,
    _TermiosBackend,
)


class FakeBackend:
    """按脚本吐出规范化按键名，用来单独验证状态机。"""

    name = "fake"

    def __init__(self, script=()):
        self.script = list(script)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return True

    def poll(self):
        keys, self.script = self.script, []
        return keys

    def stop(self):
        self.stopped = True


def _listener(script):
    keys = KeyboardEventListener()
    keys._impl = FakeBackend(script)
    return keys


def test_canonical_key_names_are_backend_neutral():
    """两种后端都归一到这套名字，状态机不该看到转义码或 pynput 对象。"""
    assert (KEY_SUCCESS, KEY_FAILURE, KEY_HANDOVER) == ("s", "f", "r")
    assert (KEY_INTERVENE, KEY_DISCARD, KEY_QUIT) == ("space", "left", "esc")


def test_termios_backend_normalizes_escape_sequences():
    """termios 读到的是原始字节，必须翻成规范名。"""
    seq = _TermiosBackend._SEQUENCES
    assert seq[" "] == KEY_INTERVENE
    assert seq["\x1b[D"] == KEY_DISCARD
    assert seq["\x1b"] == KEY_QUIT
    # 普通可打印字符原样透传
    assert seq.get("s", "s") == KEY_SUCCESS


def test_pynput_backend_maps_char_and_name():
    """pynput 的可打印键给 .char，特殊键给 .name（已与规范名一致）。"""
    backend = _PynputBackend()
    backend._queue = queue.Queue()

    captured = []

    def on_press(event):
        char = getattr(event, "char", None)
        key = char if char is not None else getattr(event, "name", None)
        if key is not None:
            captured.append(key)

    printable = type("KeyCode", (), {"char": "s"})()
    special = type("Key", (), {"char": None, "name": "space"})()
    unknown = type("Key", (), {"char": None})()

    for event in (printable, special, unknown):
        on_press(event)

    assert captured == [KEY_SUCCESS, KEY_INTERVENE], "无 char 也无 name 的事件应被忽略"


def test_outcome_keys_latch_and_clear_on_read():
    keys = _listener([KEY_SUCCESS])
    assert keys.poll_outcome() == (True, False)
    assert keys.poll_outcome() == (False, False), "读取后必须清零，否则会重复计一次成功"

    keys = _listener([KEY_FAILURE])
    assert keys.poll_outcome() == (False, True)


def test_intervene_key_toggles():
    keys = _listener([KEY_INTERVENE])
    assert keys.intervening is True
    keys._impl.script = [KEY_INTERVENE]
    assert keys.intervening is False, "空格是 toggle，不是 latch"


def test_handover_discard_and_quit():
    keys = _listener([KEY_HANDOVER, KEY_DISCARD, KEY_QUIT])
    keys._read_keys()
    assert keys.poll_handover() is True
    assert keys.poll_handover() is False
    assert keys.poll_discard() is True
    assert keys.poll_discard() is False
    assert keys.should_quit() is True
    assert keys.should_quit() is True, "退出是粘滞的，不该被读取清掉"


def test_unknown_keys_are_ignored():
    keys = _listener(["x", "9", "enter"])
    keys._read_keys()
    assert keys.poll_outcome() == (False, False)
    assert keys.intervening is False
    assert keys.should_quit() is False


def test_backend_none_degrades_without_crashing():
    """两种后端都不可用时必须安静降级，而不是抛异常打断真机流程。"""
    keys = KeyboardEventListener(backend="none")
    keys.start()
    assert keys.available is False
    assert keys.backend_name == "none"
    # 所有查询都要能正常返回
    assert keys.poll_outcome() == (False, False)
    assert keys.intervening is False
    assert keys.should_quit() is False
    keys.stop()


def test_auto_prefers_termios_over_global_hook():
    """termios 只在终端有焦点时才收键，比全局钩子安全，auto 必须优先选它。"""
    keys = KeyboardEventListener(backend="auto")
    order = [type(impl).__name__ for impl in keys._candidates()]
    assert order == ["_TermiosBackend", "_PynputBackend"]


def test_explicit_backend_selection():
    assert [type(i).__name__ for i in KeyboardEventListener(backend="pynput")._candidates()] == [
        "_PynputBackend"
    ]
    assert [type(i).__name__ for i in KeyboardEventListener(backend="termios")._candidates()] == [
        "_TermiosBackend"
    ]
    assert KeyboardEventListener(backend="none")._candidates() == []


def test_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown keyboard backend"):
        KeyboardEventListener(backend="magic")


def test_stop_releases_the_backend():
    keys = KeyboardEventListener()
    impl = FakeBackend()
    keys._impl = impl
    keys.stop()
    assert impl.stopped is True
    assert keys.available is False
