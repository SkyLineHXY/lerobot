"""OpenCV ZMQ 摄像头接口烟囱测试。

运行方式（在项目根目录）::

    pytest -sv src/lerobot/cameras/opencv_zmq/test_opencv_zmq.py

需要实机时，设置环境变量::

    OPENCV_ZMQ_DEVICE=2        # 启用本地相机测试（local 模式），值为摄像头索引
    OPENCV_ZMQ_REMOTE_HOST=192.168.x.x  # 启用远程相机测试（remote 模式）
    OPENCV_ZMQ_REMOTE_PORT=4244         # 远程端口（默认 4244）

远程测试前，需先在 NUC 启动 ZMQ PUB 服务端::

    python -m lerobot.cameras.opencv_zmq.opencv_zmq_server \\
        --device 2 --bind 0.0.0.0 --port 4244 --topic opencv \\
        --width 640 --height 480 --fps 30 --wire-encoding jpeg --jpeg-quality 90

本地直采 + 同时 PUB 广播测试::

    OPENCV_ZMQ_LOCAL_PUB=1 pytest -sv src/lerobot/cameras/opencv_zmq/test_opencv_zmq.py::test_local_publish
"""

import os

import pytest


# ─────────────────────────────────────────────────────────────────
# 辅助：检查是否有实机或远端服务
# ─────────────────────────────────────────────────────────────────

_HAVE_LOCAL = os.environ.get("OPENCV_ZMQ_DEVICE", "") not in ("", "0")
_LOCAL_DEVICE = int(os.environ.get("OPENCV_ZMQ_DEVICE", "2"))
_HAVE_REMOTE = bool(os.environ.get("OPENCV_ZMQ_REMOTE_HOST", ""))
_REMOTE_HOST = os.environ.get("OPENCV_ZMQ_REMOTE_HOST", "172.20.10.2")
_REMOTE_PORT = int(os.environ.get("OPENCV_ZMQ_REMOTE_PORT", "4244"))
_HAVE_LOCAL_PUB = os.environ.get("OPENCV_ZMQ_LOCAL_PUB", "") not in ("", "0")


# ─────────────────────────────────────────────────────────────────
# 1. 配置类验证（无需设备）
# ─────────────────────────────────────────────────────────────────


def test_config_valid_remote():
    """remote 模式必须提供 host。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCameraConfig

    cfg = OpenCVZmqCameraConfig(
        index_or_path=0, mode="remote", host="192.168.1.10", port=4244, fps=30
    )
    assert cfg.mode == "remote"
    assert cfg.host == "192.168.1.10"
    assert cfg.type == "opencv_zmq"


def test_config_valid_local():
    """local 模式无需 host，默认不发布。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCameraConfig

    cfg = OpenCVZmqCameraConfig(index_or_path=0, mode="local", fps=30, width=640, height=480)
    assert cfg.mode == "local"
    assert cfg.host is None
    assert cfg.publish is False


def test_config_remote_requires_host():
    """remote 模式未填 host 时应抛 ValueError。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCameraConfig

    with pytest.raises(ValueError, match="host"):
        OpenCVZmqCameraConfig(index_or_path=0, mode="remote")


def test_config_local_publish():
    """local + publish 模式配置验证。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCameraConfig

    cfg = OpenCVZmqCameraConfig(
        index_or_path=0,
        mode="local",
        publish=True,
        bind_addr="0.0.0.0",
        port=4244,
        topic="opencv",
        fps=30,
        width=640,
        height=480,
    )
    assert cfg.publish is True
    assert cfg.bind_addr == "0.0.0.0"
    assert cfg.port == 4244


def test_config_jpeg_quality_range():
    """jpeg_quality 超出 1–100 时应抛 ValueError。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCameraConfig

    with pytest.raises(ValueError):
        OpenCVZmqCameraConfig(index_or_path=0, mode="local", jpeg_quality=0)

    with pytest.raises(ValueError):
        OpenCVZmqCameraConfig(index_or_path=0, mode="local", jpeg_quality=101)


# ─────────────────────────────────────────────────────────────────
# 2. make_cameras_from_configs 装配（无需设备）
# ─────────────────────────────────────────────────────────────────


def test_make_cameras_from_configs_opencv_zmq():
    """make_cameras_from_configs 能正确装配 OpenCVZmqCamera。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCamera, OpenCVZmqCameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs

    cfg = OpenCVZmqCameraConfig(
        index_or_path=0, mode="remote", host="127.0.0.1", port=4244, fps=30
    )
    cameras = make_cameras_from_configs({"cam": cfg})
    assert "cam" in cameras
    assert isinstance(cameras["cam"], OpenCVZmqCamera)


# ─────────────────────────────────────────────────────────────────
# 3. find_cameras（无需设备，始终返回空列表）
# ─────────────────────────────────────────────────────────────────


def test_find_cameras_returns_list():
    """find_cameras 返回 list，每项包含 type。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCamera

    result = OpenCVZmqCamera.find_cameras()
    assert isinstance(result, list)
    for item in result:
        assert "type" in item
        assert "id" in item


# ─────────────────────────────────────────────────────────────────
# 4. 本地实机测试（需 OPENCV_ZMQ_DEVICE=<索引>）
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_LOCAL, reason="未设置 OPENCV_ZMQ_DEVICE 环境变量")
def test_local_connect_read_disconnect():
    """local 模式：连接 → 实时预览（按 q 退出）→ 断开。"""
    import time

    import cv2

    from lerobot.cameras.opencv_zmq import OpenCVZmqCamera, OpenCVZmqCameraConfig

    cfg = OpenCVZmqCameraConfig(
        index_or_path=_LOCAL_DEVICE,
        mode="local",
        fps=30,
        width=640,
        height=480,
    )
    cam = OpenCVZmqCamera(cfg)
    cam.connect(warmup=True)
    assert cam.is_connected

    win = "OpenCV ZMQ Local - press Q to quit"
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
# 5. local + publish 测试（需 OPENCV_ZMQ_LOCAL_PUB=1 + 本机摄像头）
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_LOCAL_PUB, reason="未设置 OPENCV_ZMQ_LOCAL_PUB 环境变量")
def test_local_publish():
    """local + publish 模式：摄像头直采同时 PUB 广播，验证 PUB/SUB 收发正常。

    在同一进程中启动发布端（local+publish），再由独立 SUB 客户端订阅验证。
    """
    import threading
    import time

    import cv2

    from lerobot.cameras._zmq_utils import ZMQFrameSubscriber
    from lerobot.cameras.opencv_zmq import OpenCVZmqCamera, OpenCVZmqCameraConfig

    pub_port = 14244
    pub_topic = "test_opencv_pub"

    # 启动 local+publish 摄像头
    cfg = OpenCVZmqCameraConfig(
        index_or_path=_LOCAL_DEVICE,
        mode="local",
        publish=True,
        bind_addr="0.0.0.0",
        port=pub_port,
        topic=pub_topic,
        fps=30,
        width=640,
        height=480,
    )
    cam = OpenCVZmqCamera(cfg)
    cam.connect(warmup=True)
    assert cam.is_connected

    # 启动独立 SUB 客户端，验证能收到帧
    received_frames = []
    stop_event = threading.Event()

    def sub_reader():
        sub = ZMQFrameSubscriber(
            host="127.0.0.1",
            port=pub_port,
            topic=pub_topic,
            recv_timeout_ms=2000,
            conflate=False,
        )
        try:
            while not stop_event.is_set():
                try:
                    frame = sub.get_frame()
                    received_frames.append(frame)
                except TimeoutError:
                    continue
        finally:
            sub.close()

    reader_thread = threading.Thread(target=sub_reader, daemon=True)
    reader_thread.start()

    time.sleep(1.0)  # 等待几帧
    stop_event.set()
    reader_thread.join(timeout=3.0)

    cam.disconnect()
    assert not cam.is_connected

    assert len(received_frames) > 0, "SUB 端未收到任何帧"
    for frame in received_frames:
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype.name == "uint8"


# ─────────────────────────────────────────────────────────────────
# 6. 远程烟囱测试（需 OPENCV_ZMQ_REMOTE_HOST）
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _HAVE_REMOTE, reason="未设置 OPENCV_ZMQ_REMOTE_HOST 环境变量")
def test_remote_smoke():
    """remote 模式：连接服务端 → 实时预览（按 q 退出）→ 断开。"""
    import time

    import cv2

    from lerobot.cameras.opencv_zmq import OpenCVZmqCamera, OpenCVZmqCameraConfig

    cfg = OpenCVZmqCameraConfig(
        index_or_path=0,
        mode="remote",
        host=_REMOTE_HOST,
        port=_REMOTE_PORT,
        topic="opencv",
        fps=30,
        wire_encoding="jpeg",
        jpeg_quality=90,
        width=640,
        height=480,
    )
    with OpenCVZmqCamera(cfg) as cam:
        assert cam.is_connected
        win = f"OpenCV ZMQ Remote {_REMOTE_HOST}:{_REMOTE_PORT} - press Q to quit"
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


# ─────────────────────────────────────────────────────────────────
# 7. read_latest 非阻塞读取测试（无需设备）
# ─────────────────────────────────────────────────────────────────


def test_config_sub_conflate_default():
    """sub_conflate 默认应为 False，避免 ZMQ CONFLATE 与 multipart 冲突。"""
    from lerobot.cameras.opencv_zmq import OpenCVZmqCameraConfig

    cfg = OpenCVZmqCameraConfig(
        index_or_path=0, mode="remote", host="127.0.0.1", port=4244
    )
    assert cfg.sub_conflate is False, (
        "sub_conflate 应默认为 False，ZMQ CONFLATE 与 multipart 消息不兼容"
    )


if __name__ == "__main__":
    import sys

    test_remote_smoke()
    # 默认只跑不需要硬件的配置类测试
    sys.exit(pytest.main(["-sv", __file__]))
