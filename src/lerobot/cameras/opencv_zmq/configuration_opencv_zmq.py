"""OpenCVZmq 摄像头配置类，继承 OpenCVCameraConfig 并新增 ZMQ PUB/SUB 选项。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation
from ..opencv.configuration_opencv import OpenCVCameraConfig

__all__ = ["OpenCVZmqCameraConfig"]


@CameraConfig.register_subclass("opencv_zmq")
@dataclass
class OpenCVZmqCameraConfig(OpenCVCameraConfig):
    """OpenCV + ZeroMQ 摄像头配置。

    在标准 OpenCV 摄像头基础上增加了 ZMQ PUB/SUB 远程拉流能力：

    - ``local`` 模式：等同 OpenCVCamera，直接操作本机 V4L2/USB 相机。
      若 ``publish=True``，同时作为 PUB 端广播给下游消费者。
    - ``remote`` 模式：通过 ZMQ SUB 订阅远程 NUC 上的 opencv_zmq_server 输出，
      不打开本地 VideoCapture。

    示例（远程拉流）::

        OpenCVZmqCameraConfig(
            index_or_path=0,
            mode="remote",
            host="192.168.1.10",
            port=4244,
            topic="opencv",
            fps=30,
            width=640,
            height=480,
        )

    示例（本地直采 + 同时 PUB 广播）::

        OpenCVZmqCameraConfig(
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
    """

    mode: Literal["local", "remote"] = "local"
    """连接模式：local = 本机直采；remote = 通过 ZMQ SUB 拉流。"""

    host: str | None = None
    """remote 模式下 NUC 的 IP 地址（必填）。"""

    port: int = 4244
    """remote 模式下的 ZMQ PUB 端口，或 local+publish 模式下的 bind 端口。"""

    topic: str = "opencv"
    """ZMQ PUB/SUB 的 topic 字符串。"""

    publish: bool = False
    """local 模式下是否同时 PUB 广播给下游消费者。"""

    bind_addr: str = "0.0.0.0"
    """local+publish 模式下的 bind 地址。"""

    wire_encoding: Literal["jpeg", "raw"] = "jpeg"
    """图像在网络上的编码方式：jpeg 节省带宽；raw 无损。"""

    jpeg_quality: int = 90
    """JPEG 压缩质量（1–100），仅 wire_encoding="jpeg" 时生效。"""

    recv_timeout_ms: int = 1000
    """remote 模式下 SUB socket 的接收超时（毫秒）。"""

    sub_conflate: bool = True
    """remote 模式下是否开启 ZMQ CONFLATE（仅保留最新帧）。"""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode == "remote" and self.host is None:
            raise ValueError("OpenCVZmqCameraConfig: mode='remote' 时 host 不能为 None")
        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality 必须在 1–100 之间，当前值: {self.jpeg_quality}")
