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
    - ``remote``：相机插在 NUC，通过 zerorpc 将视频流转发给远端数据集采集 PC。

    示例（远程模式，适合 ``lerobot-record``）::

        HikCameraConfig(
            mode="remote",
            host="192.168.1.10",
            port=4243,
            fps=30,
            width=1280,
            height=720,
        )

    示例（本地模式，直接在 NUC 上采集）::

        HikCameraConfig(mode="local", device_index=0, fps=30, width=1280, height=720)
    """

    mode: Literal["local", "remote"] = "remote"
    """连接模式：local = 本机直采；remote = 通过 zerorpc 拉流。"""

    host: str | None = None
    """remote 模式下 NUC 的 IP 地址（必填）。"""

    port: int = 4243
    """remote 模式下 NUC 侧 hik_camera_server 的监听端口。"""

    device_index: int = 0
    """NUC 上相机的枚举索引（多台相机时使用）。"""

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

    heartbeat_s: int = 30
    """zerorpc 客户端心跳间隔（秒）；需大于单次取帧最大延迟。"""

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if self.mode == "remote" and self.host is None:
            raise ValueError("HikCameraConfig: mode='remote' 时 host 不能为 None")

        if not (1 <= self.jpeg_quality <= 100):
            raise ValueError(f"jpeg_quality 必须在 1–100 之间，当前值: {self.jpeg_quality}")
