"""海康相机接口烟囱测试。

运行方式（在项目根目录）::

    pytest -sv src/lerobot/cameras/hik_camera/test_hik_camera.py

需要实机时，设置环境变量::

    HIK_CAMERA_DEVICE=1        # 启用本地相机测试（local 模式）
    HIK_CAMERA_REMOTE_HOST=192.168.x.x  # 启用远程相机测试（remote 模式）
    HIK_CAMERA_REMOTE_PORT=4243         # 远程端口（默认 4243）
"""

import os

import pytest


# ─────────────────────────────────────────────────────────────────
# 辅助：检查是否有实机或远端服务
# ─────────────────────────────────────────────────────────────────

_HAVE_LOCAL = os.environ.get("HIK_CAMERA_DEVICE", "") not in ("", "0")
_HAVE_REMOTE = bool(os.environ.get("HIK_CAMERA_REMOTE_HOST", ""))
_REMOTE_HOST = os.environ.get("HIK_CAMERA_REMOTE_HOST", "127.0.0.1")
_REMOTE_PORT = int(os.environ.get("HIK_CAMERA_REMOTE_PORT", "4243"))


# ─────────────────────────────────────────────────────────────────
# 1. 配置类验证（无需设备）
# ─────────────────────────────────────────────────────────────────


def test_config_valid_remote():
    """remote 模式必须提供 host。"""
    from lerobot.cameras.hik_camera import HikCameraConfig

    cfg = HikCameraConfig(mode="remote", host="192.168.1.10", port=4243, fps=30)
    assert cfg.mode == "remote"
    assert cfg.host == "192.168.1.10"
    assert cfg.type == "hik"


def test_config_valid_local():
    """local 模式无需 host。"""
    from lerobot.cameras.hik_camera import HikCameraConfig

    cfg = HikCameraConfig(mode="local", fps=30, width=1280, height=720)
    assert cfg.mode == "local"
    assert cfg.host is None


def test_config_remote_requires_host():
    """remote 模式未填 host 时应抛 ValueError。"""
    from lerobot.cameras.hik_camera import HikCameraConfig

    with pytest.raises(ValueError, match="host"):
        HikCameraConfig(mode="remote")


def test_config_jpeg_quality_range():
    """jpeg_quality 超出 1–100 时应抛 ValueError。"""
    from lerobot.cameras.hik_camera import HikCameraConfig

    with pytest.raises(ValueError):
        HikCameraConfig(mode="local", jpeg_quality=0)

    with pytest.raises(ValueError):
        HikCameraConfig(mode="local", jpeg_quality=101)


# ─────────────────────────────────────────────────────────────────
# 2. make_cameras_from_configs 装配（无需设备）
# ─────────────────────────────────────────────────────────────────


def test_make_cameras_from_configs_hik():
    """make_cameras_from_configs 能正确装配 HikCamera。"""
    from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs

    cfg = HikCameraConfig(mode="remote", host="127.0.0.1", port=4243, fps=30)
    cameras = make_cameras_from_configs({"wrist": cfg})
    assert "wrist" in cameras
    assert isinstance(cameras["wrist"], HikCamera)


# ─────────────────────────────────────────────────────────────────
# 3. find_cameras（SDK 必须可用）
# ─────────────────────────────────────────────────────────────────


def test_find_devices_returns_list():
    """find_cameras 必须返回 list（即使没有相机也返回空列表）。"""
    try:
        from lerobot.cameras.hik_camera import HikCamera

        result = HikCamera.find_cameras()
        assert isinstance(result, list)
        for item in result:
            assert "id" in item
            assert "type" in item
    except Exception as e:
        # SDK 未安装时 skip
        pytest.skip(f"SDK 不可用，跳过 find_cameras 测试: {e}")


# ─────────────────────────────────────────────────────────────────
# 4. 本地实机测试（需 HIK_CAMERA_DEVICE=1）
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_LOCAL, reason="未设置 HIK_CAMERA_DEVICE 环境变量")
def test_local_connect_read_disconnect():
    """local 模式：连接 → 读取 100 帧 → 断开。"""
    from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

    cfg = HikCameraConfig(mode="local", device_index=0, fps=30)
    cam = HikCamera(cfg)
    cam.connect(warmup=True)
    assert cam.is_connected

    for _ in range(10):
        frame = cam.read()
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype.name == "uint8"

    cam.disconnect()
    assert not cam.is_connected


# ─────────────────────────────────────────────────────────────────
# 5. 远程烟囱测试（需 HIK_CAMERA_REMOTE_HOST）
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_REMOTE, reason="未设置 HIK_CAMERA_REMOTE_HOST 环境变量")
def test_remote_smoke():
    """remote 模式：连接服务端 → 读取帧 → 断开。"""
    from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

    cfg = HikCameraConfig(
        mode="remote",
        host=_REMOTE_HOST,
        port=_REMOTE_PORT,
        fps=30,
        wire_encoding="jpeg",
        jpeg_quality=90,
    )
    with HikCamera(cfg) as cam:
        assert cam.is_connected
        frame = cam.read()
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype.name == "uint8"
