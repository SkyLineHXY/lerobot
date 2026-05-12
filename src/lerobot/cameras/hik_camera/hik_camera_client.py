"""海康相机 zerorpc 客户端，运行在数据集采集 PC 侧。"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HikCameraClient:
    """通过 zerorpc 连接到 NUC 侧 HikCameraServer 的客户端。

    由 ``HikCamera`` 在 ``mode="remote"`` 时内部使用，不建议直接调用。

    Args:
        host: NUC 的 IP 地址。
        port: NUC 侧服务端口（默认 4243）。
        heartbeat: zerorpc 心跳间隔（秒），需大于单次取帧最大延迟。
    """

    def __init__(self, host: str, port: int, heartbeat: int = 30) -> None:
        import zerorpc  # 懒加载，仅 remote 模式需要

        self._host = host
        self._port = port
        self._server = zerorpc.Client(heartbeat=heartbeat)
        try:
            self._server.connect(f"tcp://{host}:{port}")
            logger.info(f"已连接到 HikCameraServer @ {host}:{port}")
        except Exception as e:
            logger.warning(f"连接 HikCameraServer 失败: {e}")

    # ------------------------------------------------------------------
    # 相机控制（代理到服务端方法，改名避免与 zerorpc.Client 内置方法冲突）
    # ------------------------------------------------------------------

    def open_camera(
        self,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """在 NUC 侧打开相机（对应服务端 connect_camera）。

        Args:
            fps: 目标帧率，None 表示使用服务端默认值。
            width: 输出宽度，None 表示使用服务端默认值。
            height: 输出高度，None 表示使用服务端默认值。
        """
        self._server.connect_camera(fps, width, height)

    def close_camera(self) -> None:
        """在 NUC 侧关闭相机（对应服务端 disconnect_camera）。"""
        self._server.disconnect_camera()

    @property
    def is_connected(self) -> bool:
        """查询 NUC 侧相机是否已连接。"""
        try:
            return bool(self._server.camera_is_connected())
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        """获取 NUC 侧相机状态信息。"""
        return dict(self._server.get_status())

    def find_devices(self) -> list[dict[str, Any]]:
        """在 NUC 侧枚举可见相机设备。"""
        return list(self._server.find_devices())

    # ------------------------------------------------------------------
    # 帧采集
    # ------------------------------------------------------------------

    def get_frame(self) -> np.ndarray:
        """从 NUC 拉取一帧图像，返回 BGR uint8 numpy 数组。

        Returns:
            shape (H, W, 3) 的 BGR uint8 ndarray。

        Raises:
            RuntimeError: 取帧或解码失败。
        """
        import cv2

        d = self._server.get_frame()
        encoding = d["encoding"]
        raw: bytes = d["data"]

        if encoding == "jpeg":
            buf = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("JPEG 解码失败，收到的数据可能已损坏")
            return frame
        else:
            shape = tuple(d["shape"])
            dtype = np.dtype(d["dtype"])
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭到服务端的 zerorpc 连接。"""
        try:
            self._server.close()
        except Exception as e:
            logger.warning(f"关闭 zerorpc 客户端时异常: {e}")
