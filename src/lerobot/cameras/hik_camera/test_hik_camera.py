"""海康相机接口烟囱测试。

运行方式（在项目根目录）::

    pytest -sv src/lerobot/cameras/hik_camera/test_hik_camera.py

需要实机时，设置环境变量::

    HIK_CAMERA_DEVICE=1        # 启用本地相机测试（local 模式）
    HIK_CAMERA_REMOTE_HOST=192.168.x.x  # 启用远程相机测试（remote 模式）
    HIK_CAMERA_REMOTE_PORT=4243         # 远程端口（默认 4243）

远程测试前，需先在 NUC 启动 ZMQ PUB 服务端::

    python -m lerobot.cameras.hik_camera.hik_camera_server \\
        --bind 0.0.0.0 --port 4243 --topic hik \\
        --fps 30 --wire-encoding jpeg --jpeg-quality 90
"""

import os

import pytest


# ─────────────────────────────────────────────────────────────────
# 辅助：检查是否有实机或远端服务
# ─────────────────────────────────────────────────────────────────

_HAVE_LOCAL = os.environ.get("HIK_CAMERA_DEVICE", "") not in ("", "0")
_HAVE_REMOTE = bool(os.environ.get("HIK_CAMERA_REMOTE_HOST", ""))
_REMOTE_HOST = os.environ.get("HIK_CAMERA_REMOTE_HOST", "172.20.10.2")
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
    """local 模式：连接 → 实时预览（按 q 退出）→ 断开。"""
    import time

    import cv2

    from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

    cfg = HikCameraConfig(mode="local", device_index=0, fps=30,width=640,height=480,)
    cam = HikCamera(cfg)
    cam.connect(warmup=True)
    assert cam.is_connected

    win = "HIK Local - press Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_display = 0.0
    t_fps = time.perf_counter()

    while True:
        frame = cam.read()
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype.name == "uint8"

        frame_count += 1
        elapsed = time.perf_counter() - t_fps
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            t_fps = time.perf_counter()

        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            f"FPS: {fps_display:.1f}  {w}x{h}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyWindow(win)
    cam.disconnect()
    assert not cam.is_connected


# ─────────────────────────────────────────────────────────────────
# 5. 远程烟囱测试（需 HIK_CAMERA_REMOTE_HOST）
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_REMOTE, reason="未设置 HIK_CAMERA_REMOTE_HOST 环境变量")
def test_remote_smoke():
    """remote 模式：连接服务端 → 实时预览（按 q 退出）→ 断开。"""
    import time

    import cv2

    from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

    cfg = HikCameraConfig(
        mode="remote",
        host=_REMOTE_HOST,
        port=_REMOTE_PORT,
        fps=30,
        wire_encoding="jpeg",
        jpeg_quality=90,
        width=640,
        height=480,
    )
    with HikCamera(cfg) as cam:
        assert cam.is_connected
        win = f"HIK Remote {_REMOTE_HOST}:{_REMOTE_PORT} - press Q to quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        frame_count = 0
        fps_display = 0.0
        t_fps = time.perf_counter()

        while True:
            frame = cam.read()
            assert frame.ndim == 3
            assert frame.shape[2] == 3
            assert frame.dtype.name == "uint8"

            frame_count += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                t_fps = time.perf_counter()

            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}  {w}x{h}  {_REMOTE_HOST}:{_REMOTE_PORT}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

        cv2.destroyWindow(win)


if __name__ == "__main__":
    test_remote_smoke()

# from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig
#
# cfg = HikCameraConfig(
#     mode="remote", host="172.20.10.2", port=4243, topic="hik",
#     fps=30
# )
# with HikCamera(cfg) as cam:
#     for i in range(100):
#         frame = cam.read()   # (720, 1280, 3) uint8 BGR numpy array
