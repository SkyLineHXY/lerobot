"""HikCamera 配置类，遵循 lerobot CameraConfig 约定。"""

from dataclasses import dataclass
from typing import Literal

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["HikCameraConfig"]


@CameraConfig.register_subclass("hik")
@dataclass
class HikCameraConfig(CameraConfig):
    """海康工业相机配置。

    支持两种部署模式：
    - ``local``：相机直连本机（NUC），适合在 NUC 上直接训练或调试。
    - ``remote``：相机插在 NUC，通过 ZeroMQ PUB/SUB 将视频流转发给远端数据集采集 PC。

    **分辨率说明**：
    - ``width/height`` 表示**软件输出**分辨率（即最终产出的图像尺寸）。
    - ``sensor_width/sensor_height`` 可选，控制 SDK 的 AOI（传感器读出窗口），
      None 表示使用相机默认全幅。
    - local 模式下，若 sensor_* 与输出尺寸不一致，会在 ``_postprocess_image`` 中
      通过 ``cv2.resize`` 将 SDK 帧缩放到输出尺寸（不会裁剪）。
    - remote 模式下，resize 在 server 端完成（减少网络字节数）。

    示例（远程模式，适合 ``lerobot-record``）::

        HikCameraConfig(
            mode="remote",
            host="192.168.1.10",
            port=4243,
            topic="hik",
            fps=30,
            width=1280,
            height=720,
        )

    示例（本地模式，自定义 AOI 并缩放输出）::

        HikCameraConfig(
            mode="local",
            device_index=0,
            fps=30,
            sensor_width=1920,
            sensor_height=1200,
            width=1280,
            height=720,
        )
    """

    mode: Literal["local", "remote"] = "remote"
    """连接模式：local = 本机直采；remote = 通过 ZMQ SUB 拉流。"""

    host: str | None = None
    """remote 模式下 NUC 的 IP 地址（必填）。"""

    port: int = 4243
    """remote 模式下 NUC 侧 hik_camera_server 的 ZMQ PUB 端口。"""

    topic: str = "hik"
    """ZMQ PUB/SUB 的 topic 字符串，用于区分多路相机流。"""

    device_index: int = 0
    """NUC 上相机的枚举索引（多台相机时使用）。"""

    sensor_width: int | None = None
    """SDK AOI 传感器读出宽度（像素），None 表示使用相机默认全幅。"""

    sensor_height: int | None = None
    """SDK AOI 传感器读出高度（像素），None 表示使用相机默认全幅。"""

    color_mode: ColorMode = ColorMode.BGR
    """输出图像的颜色空间：BGR（OpenCV 默认）或 RGB。"""

    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    """图像旋转角度。"""

    warmup_s: int = 1
    """connect() 时的预热时间（秒），等待相机自动曝光稳定。"""

    wire_encoding: Literal["jpeg", "raw"] = "jpeg"
    """remote 模式下图像在网络上的编码方式：jpeg 节省带宽；raw 无损。"""

    jpeg_quality: int = 90
    """JPEG 压缩质量（1–100），仅 wire_encoding="jpeg" 时生效。"""

    recv_timeout_ms: int = 1000
    """remote 模式下 SUB socket 的接收超时（毫秒）。"""

    sub_conflate: bool = False
    """remote 模式下是否开启 ZMQ CONFLATE（仅保留最新一帧，丢弃旧帧）。

    注意：ZMQ CONFLATE 与 multipart 消息冲突，会导致 ``Assertion failed: !_more``，
    因此默认关闭。通过 RCVHWM=1 已能限制缓冲区仅保留最新消息。
    """

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.mode == "remote" and self.host is None:
            raise ValueError("HikCameraConfig: mode='remote' 时 host 不能为 None")

        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality 必须在 1–100 之间，当前值: {self.jpeg_quality}")
