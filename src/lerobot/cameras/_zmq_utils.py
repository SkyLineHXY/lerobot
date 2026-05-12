"""ZeroMQ PUB/SUB 帧传输通用工具。

供 HikCamera 与 OpenCVZmqCamera 共用，消息格式统一：

    [topic_bytes, header_msgpack, payload_bytes]

    header = {"ts": float, "shape": [H,W,C], "dtype": "uint8", "encoding": "jpeg"|"raw"}
"""

import logging

import cv2
import msgpack
import numpy as np
import zmq

logger = logging.getLogger(__name__)


class ZMQFrameSubscriber:
    """ZeroMQ SUB 帧接收器，订阅远程 PUB 端广播的视频帧。

    Args:
        host: PUB 端 IP 地址。
        port: PUB 端口。
        topic: SUB 订阅的 topic 字符串。
        recv_timeout_ms: recv_multipart 超时（毫秒），超时抛 TimeoutError。
        conflate: 是否开启 CONFLATE（仅保留最新帧，丢弃旧帧）。
    """

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        recv_timeout_ms: int = 1000,
        conflate: bool = True,
    ) -> None:
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._sub.setsockopt(zmq.RCVHWM, 1)
        if conflate:
            self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._sub.connect(f"tcp://{host}:{port}")
        logger.info("ZMQFrameSubscriber 已连接 tcp://%s:%d，topic='%s'", host, port, topic)

    def get_frame(self) -> np.ndarray:
        """接收一帧图像，返回 BGR uint8 numpy 数组 (H, W, 3)。"""
        try:
            parts = self._sub.recv_multipart()
        except zmq.Again:
            raise TimeoutError("ZMQ SUB recv 超时，PUB 端可能未启动或网络不通")

        if len(parts) < 3:
            raise RuntimeError(f"收到异常消息，parts={len(parts)}")

        header = msgpack.unpackb(parts[1])
        payload: bytes = parts[2]

        encoding = header["encoding"]
        if encoding == "jpeg":
            buf = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("JPEG 解码失败，收到的数据可能已损坏")
            return frame
        else:
            shape = tuple(header["shape"])
            dtype = np.dtype(header["dtype"])
            return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()

    def close(self) -> None:
        """关闭 SUB socket 和 ZMQ context。"""
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.close()
        self._ctx.term()
        logger.info("ZMQFrameSubscriber 已关闭")


class ZMQFramePublisher:
    """ZeroMQ PUB 帧发送器，将 OpenCV 帧编码后广播。

    Args:
        host: bind 地址。
        port: bind 端口。
        topic: PUB topic 字符串。
        wire_encoding: ``jpeg`` 或 ``raw``。
        jpeg_quality: JPEG 质量（1–100）。
    """

    def __init__(
        self,
        host: str,
        port: int,
        topic: str,
        wire_encoding: str = "jpeg",
        jpeg_quality: int = 90,
    ) -> None:
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.bind(f"tcp://{host}:{port}")
        self._topic = topic.encode()
        self._wire_encoding = wire_encoding
        self._jpeg_quality = jpeg_quality
        logger.info("ZMQFramePublisher 已绑定 tcp://%s:%d，topic='%s'", host, port, topic)

    def send(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        """编码并发送一帧。

        Args:
            frame: BGR uint8 ndarray (H, W, 3)。
            timestamp: 可选时间戳，None 时自动生成。
        """
        import time as _time

        if timestamp is None:
            timestamp = _time.time()

        if self._wire_encoding == "jpeg":
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                raise RuntimeError("JPEG 编码失败")
            payload: bytes = buf.tobytes()
        else:
            payload = frame.tobytes()

        header = {
            "ts": timestamp,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "encoding": self._wire_encoding,
        }
        header_packed = msgpack.packb(header)

        self._pub.send_multipart([self._topic, header_packed, payload])

    def close(self) -> None:
        """关闭 PUB socket 和 ZMQ context。"""
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.close()
        self._ctx.term()
        logger.info("ZMQFramePublisher 已关闭")
