"""海康相机 zerorpc 服务端，运行在 NUC（相机直连侧）。

启动方式::

    python -m lerobot.cameras.hik_camera.hik_camera_server \\
        --host 0.0.0.0 --port 4243 \\
        --device-index 0 \\
        --wire-encoding jpeg --jpeg-quality 90

注意：需要在 NUC 上安装 zerorpc 与海康 MVS SDK。
"""

import argparse
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class HikCameraServer:
    """将 HikCameraSDK 通过 zerorpc 暴露为远程服务。

    所有 public 方法会被 zerorpc.Server 自动暴露，客户端通过代理直接调用。
    方法命名刻意回避了 zerorpc.Client 的内置方法名（如 connect / disconnect / close）。

    Args:
        device_index: 相机枚举索引。
        wire_encoding: 帧传输编码，``jpeg`` 节省带宽，``raw`` 无损。
        jpeg_quality: JPEG 压缩质量（1–100），仅 ``jpeg`` 时生效。
    """

    def __init__(
        self,
        device_index: int = 0,
        wire_encoding: str = "jpeg",
        jpeg_quality: int = 90,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        from .hik_camera_sdk import HikCameraSDK

        self._sdk = HikCameraSDK(device_index)
        self._wire_encoding = wire_encoding
        self._jpeg_quality = jpeg_quality
        # 服务端默认参数，客户端调用 connect_camera 时可覆盖
        self._default_fps = fps
        self._default_width = width
        self._default_height = height

    # ------------------------------------------------------------------
    # 相机控制（命名避免与 zerorpc 内置方法冲突）
    # ------------------------------------------------------------------

    def connect_camera(
        self,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """打开相机并开始取流。

        Args:
            fps: 目标帧率，None 时使用服务端启动时指定的默认值。
            width: 输出宽度，None 时使用服务端默认值。
            height: 输出高度，None 时使用服务端默认值。
        """
        self._sdk.connect(
            fps=fps if fps is not None else self._default_fps,
            width=width if width is not None else self._default_width,
            height=height if height is not None else self._default_height,
        )

    def disconnect_camera(self) -> None:
        """停止取流并释放相机句柄。"""
        self._sdk.disconnect()

    def camera_is_connected(self) -> bool:
        """返回相机当前是否已连接。"""
        return self._sdk.is_connected

    def get_status(self) -> dict[str, Any]:
        """返回相机状态信息（分辨率、重连次数等）。"""
        return self._sdk.get_status()

    def find_devices(self) -> list[dict[str, Any]]:
        """枚举本机可见的海康相机设备。"""
        return self._sdk.find_devices()

    # ------------------------------------------------------------------
    # 帧采集
    # ------------------------------------------------------------------

    def get_frame(self) -> dict[str, Any]:
        """采集一帧并以指定编码返回。

        Returns:
            dict 包含:
            - ``data``: bytes — 编码后的帧数据
            - ``shape``: list[int] — 原始 numpy 数组形状 [H, W, C]
            - ``dtype``: str — numpy dtype 字符串（如 ``uint8``）
            - ``encoding``: str — ``jpeg`` 或 ``raw``
            - ``timestamp``: float — 采集时间戳（time.time()）
        """
        import cv2

        frame = self._sdk.capture_image()  # BGR uint8 (H, W, 3)

        if self._wire_encoding == "jpeg":
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
            )
            if not ok:
                raise RuntimeError("JPEG 编码失败")
            payload: bytes = buf.tobytes()
        else:
            payload = frame.tobytes()

        return {
            "data": payload,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "encoding": self._wire_encoding,
            "timestamp": time.time(),
        }


# ------------------------------------------------------------------
# 服务入口
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="海康相机 zerorpc 服务端（运行于 NUC）")
    p.add_argument("--host", default="0.0.0.0", help="绑定地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=4243, help="监听端口（默认 4243）")
    p.add_argument("--device-index", type=int, default=0, dest="device_index", help="相机枚举索引")
    p.add_argument(
        "--wire-encoding",
        default="jpeg",
        choices=["jpeg", "raw"],
        dest="wire_encoding",
        help="帧传输编码（默认 jpeg）",
    )
    p.add_argument("--jpeg-quality", type=int, default=90, dest="jpeg_quality", help="JPEG 质量 1–100")
    p.add_argument("--fps", type=float, default=None, help="相机采集帧率（默认使用相机硬件默认值）")
    p.add_argument("--width", type=int, default=None, help="输出图像宽度（像素，默认使用相机默认值）")
    p.add_argument("--height", type=int, default=None, help="输出图像高度（像素，默认使用相机默认值）")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    args = _parse_args()

    server_obj = HikCameraServer(
        device_index=args.device_index,
        wire_encoding=args.wire_encoding,
        jpeg_quality=args.jpeg_quality,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    # 启动时自动连接相机
    logger.info("正在连接相机...")
    server_obj.connect_camera()
    logger.info(f"相机状态: {server_obj.get_status()}")

    import zerorpc

    srv = zerorpc.Server(server_obj)
    bind_addr = f"tcp://{args.host}:{args.port}"
    srv.bind(bind_addr)
    logger.info(f"HikCameraServer 已绑定到 {bind_addr}，等待连接...")
    srv.run()
