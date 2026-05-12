"""OpenCV ZeroMQ PUB 服务端，运行在 NUC（USB/V4L2 相机直连侧）。

将本地 OpenCV VideoCapture 的帧通过 ZMQ PUB 广播，供 PC 端 OpenCVZmqCamera
（mode="remote"）订阅。消息格式与 HikCameraPublisher 完全一致。

启动方式::

    python -m lerobot.cameras.opencv_zmq.opencv_zmq_server \\
        --device 0 --bind 0.0.0.0 --port 4244 --topic opencv \\
        --fps 30 --width 640 --height 480 \\
        --wire-encoding jpeg --jpeg-quality 90
"""

import argparse
import logging
import time

import cv2
from numpy.typing import NDArray

from .._zmq_utils import ZMQFramePublisher

logger = logging.getLogger(__name__)


class OpenCVZmqServer:
    """使用 ZeroMQ PUB socket 持续广播本地 OpenCV 摄像头帧。

    Args:
        device: 摄像头设备索引或路径（传 int 或 str）。
        width: 输出宽度（像素），None = 相机默认。
        height: 输出高度（像素），None = 相机默认。
        fps: 目标帧率。
        wire_encoding: 帧传输编码，``jpeg`` 节省带宽，``raw`` 无损。
        jpeg_quality: JPEG 压缩质量（1–100）。
    """

    def __init__(
        self,
        device: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        wire_encoding: str = "jpeg",
        jpeg_quality: int = 90,
    ) -> None:
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._wire_encoding = wire_encoding
        self._jpeg_quality = jpeg_quality
        self._cap: cv2.VideoCapture | None = None
        self._publisher: ZMQFramePublisher | None = None

    def open_camera(self) -> None:
        """打开摄像头并配置分辨率/帧率。"""
        self._cap = cv2.VideoCapture(self._device)
        if not self._cap.isOpened():
            raise ConnectionError(f"无法打开摄像头: {self._device}")

        if self._width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("摄像头已打开: %dx%d @ %.1f fps", actual_w, actual_h, actual_fps)

    def bind(self, host: str, port: int, topic: str) -> None:
        """创建 ZMQ PUB socket 并绑定地址。"""
        self._publisher = ZMQFramePublisher(
            host=host,
            port=port,
            topic=topic,
            wire_encoding=self._wire_encoding,
            jpeg_quality=self._jpeg_quality,
        )

    def run(self) -> None:
        """阻塞主循环：从摄像头取帧 → 编码 → PUB 广播。"""
        if self._cap is None:
            raise RuntimeError("请先调用 open_camera()")
        if self._publisher is None:
            raise RuntimeError("请先调用 bind()")

        target_w = self._width
        target_h = self._height
        need_resize = target_w is not None and target_h is not None

        logger.info("PUB 循环开始，按 Ctrl+C 退出")

        while True:
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("读取帧失败，跳过")
                time.sleep(0.01)
                continue

            if need_resize:
                h, w = frame.shape[:2]
                if w != target_w or h != target_h:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

            self._publisher.send(frame)

    def close(self) -> None:
        """释放摄像头和 ZMQ 资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None
        logger.info("OpenCVZmqServer 已关闭")


# ------------------------------------------------------------------
# 服务入口
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenCV ZMQ PUB 服务端（运行于 NUC）")
    p.add_argument("--device", default="0", help="摄像头设备索引或路径（默认 0）")
    p.add_argument("--bind", default="0.0.0.0", help="绑定地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=4244, help="PUB 端口（默认 4244）")
    p.add_argument("--topic", default="opencv", help="PUB topic 字符串（默认 opencv）")
    p.add_argument("--fps", type=float, default=30.0, help="目标帧率（默认 30）")
    p.add_argument("--width", type=int, default=640, help="输出宽度（默认 640）")
    p.add_argument("--height", type=int, default=480, help="输出高度（默认 480）")
    p.add_argument(
        "--wire-encoding",
        default="jpeg",
        choices=["jpeg", "raw"],
        dest="wire_encoding",
        help="帧传输编码（默认 jpeg）",
    )
    p.add_argument("--jpeg-quality", type=int, default=90, dest="jpeg_quality", help="JPEG 质量 1–100")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = _parse_args()

    device = int(args.device) if args.device.isdigit() else args.device
    server = OpenCVZmqServer(
        device=device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        wire_encoding=args.wire_encoding,
        jpeg_quality=args.jpeg_quality,
    )

    server.open_camera()
    server.bind(args.bind, args.port, args.topic)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        server.close()
