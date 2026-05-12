"""海康相机 ZeroMQ PUB 服务端，运行在 NUC（相机直连侧）。

通过 PUB/SUB 广播视频帧，替代旧版 zerorpc 同步阻塞 RPC。客户端无需请求，
直接订阅最新帧（CONFLATE=1），消除 RTT 延迟与心跳维护开销。

消息格式（multipart）::

    [topic_bytes, header_msgpack, payload_bytes]

    header = {"ts": float, "shape": [H,W,C], "dtype": "uint8", "encoding": "jpeg"|"raw"}

启动方式::

    python -m lerobot.cameras.hik_camera.hik_camera_server \\
        --bind 0.0.0.0 --port 4243 --topic hik \\
        --sensor-width 1920 --sensor-height 1200 \\
        --output-width 1280 --output-height 720 \\
        --fps 30 --wire-encoding jpeg --jpeg-quality 90

注意：需要在 NUC 上安装 pyzmq、msgpack 与海康 MVS SDK。
"""

import argparse
import logging
import time

import cv2
import msgpack
import numpy as np
import zmq

from .hik_camera_sdk import HikCameraSDK

logger = logging.getLogger(__name__)


class HikCameraPublisher:
    """使用 ZeroMQ PUB socket 持续广播海康相机帧。

    Args:
        device_index: 相机枚举索引。
        sensor_width: SDK AOI 宽度（像素），None = 相机默认。
        sensor_height: SDK AOI 高度（像素），None = 相机默认。
        output_width: resize 后的输出宽度，None = 不缩放。
        output_height: resize 后的输出高度，None = 不缩放。
        wire_encoding: 帧传输编码，``jpeg`` 节省带宽，``raw`` 无损。
        jpeg_quality: JPEG 压缩质量（1–100），仅 ``jpeg`` 时生效。
    """

    def __init__(
        self,
        device_index: int = 0,
        sensor_width: int | None = None,
        sensor_height: int | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
        wire_encoding: str = "jpeg",
        jpeg_quality: int = 90,
    ) -> None:
        self._sdk = HikCameraSDK(device_index)
        self._sensor_width = sensor_width
        self._sensor_height = sensor_height
        self._output_width = output_width
        self._output_height = output_height
        self._wire_encoding = wire_encoding
        self._jpeg_quality = jpeg_quality
        self._ctx: zmq.Context | None = None
        self._pub: zmq.Socket | None = None

    def bind(self, host: str, port: int, topic: str) -> None:
        """创建 ZMQ PUB socket 并绑定地址。"""
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.bind(f"tcp://{host}:{port}")
        self._topic = topic.encode()
        logger.info(f"ZMQ PUB 已绑定 tcp://{host}:{port}，topic='{topic}'")

    def connect_camera(self, fps: float | None = None) -> None:
        """打开相机并开始取流。"""
        self._sdk.connect(
            fps=fps,
            sensor_width=self._sensor_width,
            sensor_height=self._sensor_height,
        )
        logger.info(f"相机已连接: {self._sdk.get_status()}")

    def run(self) -> None:
        """阻塞主循环：持续采帧 → resize → 编码 → PUB 广播。"""
        if self._pub is None:
            raise RuntimeError("请先调用 bind()")

        need_resize = (
            self._output_width is not None
            and self._output_height is not None
        )

        encode_params = (
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            if self._wire_encoding == "jpeg"
            else None
        )

        logger.info(
            "PUB 循环开始: encoding=%s, output=%sx%s",
            self._wire_encoding,
            self._output_width or "-",
            self._output_height or "-",
        )

        while True:
            raw: np.ndarray = self._sdk.capture_image()  # BGR uint8 (H, W, 3)

            if need_resize:
                raw_h, raw_w = raw.shape[:2]
                if raw_w != self._output_width or raw_h != self._output_height:
                    raw = cv2.resize(
                        raw, (self._output_width, self._output_height),
                        interpolation=cv2.INTER_AREA,
                    )

            ts = time.time()

            if self._wire_encoding == "jpeg":
                ok, buf = cv2.imencode(".jpg", raw, encode_params)
                if not ok:
                    raise RuntimeError("JPEG 编码失败")
                payload: bytes = buf.tobytes()
            else:
                payload = raw.tobytes()

            header = {
                "ts": ts,
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                "encoding": self._wire_encoding,
            }
            header_packed = msgpack.packb(header)

            self._pub.send_multipart([self._topic, header_packed, payload])  # type: ignore[union-attr]

    def close(self) -> None:
        """停止取流并释放 ZMQ 资源。"""
        if self._sdk.is_connected:
            self._sdk.disconnect()
        if self._pub is not None:
            self._pub.close()
        if self._ctx is not None:
            self._ctx.term()
        logger.info("HikCameraPublisher 已关闭")


# ------------------------------------------------------------------
# 服务入口
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="海康相机 ZMQ PUB 服务端（运行于 NUC）")
    p.add_argument("--bind", default="0.0.0.0", help="绑定地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=4243, help="PUB 端口（默认 4243）")
    p.add_argument("--topic", default="hik", help="PUB topic 字符串（默认 hik）")
    p.add_argument("--device-index", type=int, default=0, dest="device_index", help="相机枚举索引")
    p.add_argument("--fps", type=float, default=None, help="相机采集帧率（默认使用相机硬件默认值）")
    p.add_argument("--sensor-width", type=int, default=None, dest="sensor_width", help="SDK AOI 宽度（像素）")
    p.add_argument("--sensor-height", type=int, default=None, dest="sensor_height", help="SDK AOI 高度（像素）")
    p.add_argument("--output-width", type=int, default=None, dest="output_width", help="resize 后的输出宽度")
    p.add_argument("--output-height", type=int, default=None, dest="output_height", help="resize 后的输出高度")
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

    publisher = HikCameraPublisher(
        device_index=args.device_index,
        sensor_width=args.sensor_width,
        sensor_height=args.sensor_height,
        output_width=args.output_width,
        output_height=args.output_height,
        wire_encoding=args.wire_encoding,
        jpeg_quality=args.jpeg_quality,
    )

    publisher.connect_camera(fps=args.fps)
    publisher.bind(args.bind, args.port, args.topic)

    try:
        publisher.run()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        publisher.close()
