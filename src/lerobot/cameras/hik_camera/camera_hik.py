"""HikCamera：海康工业相机的 lerobot Camera 实现。

支持两种部署模式：
- ``local``：相机直连本机，直接调用 HikCameraSDK。
- ``remote``：相机在 NUC，通过 zerorpc 拉流（需先在 NUC 上启动 hik_camera_server.py）。
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_hik import ColorMode, HikCameraConfig

logger = logging.getLogger(__name__)


class HikCamera(Camera):
    """海康工业相机驱动，符合 lerobot ``Camera`` 抽象接口。

    通过 ``HikCameraConfig.mode`` 选择本地（NUC）或远程（PC）模式。

    Example（远程模式，在数据集采集 PC 上使用）::

        from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

        config = HikCameraConfig(
            mode="remote", host="192.168.1.10", port=4243,
            fps=30, width=1280, height=720,
        )
        with HikCamera(config) as cam:
            for _ in range(100):
                frame = cam.read()   # (720, 1280, 3) uint8 BGR

    Example（本地模式，在 NUC 上直接调试）::

        config = HikCameraConfig(mode="local", device_index=0, fps=30)
        with HikCamera(config) as cam:
            frame = cam.read()
    """

    def __init__(self, config: HikCameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.rotation = get_cv2_rotation(config.rotation)

        # 运行时 backend：local → HikCameraSDK；remote → HikCameraClient
        self._backend = None

        # 后台读取线程相关
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock: Lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None
        self._new_frame_event: Event = Event()

    def __str__(self) -> str:
        if self.config.mode == "remote":
            return f"HikCamera(remote@{self.config.host}:{self.config.port})"
        return f"HikCamera(local#{self.config.device_index})"

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """相机是否处于已连接状态（后台线程运行中）。"""
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """枚举本机可见的海康相机设备。

        注意：远程模式下仍枚举的是本机设备，不会通过网络查询 NUC。
        若需枚举 NUC 侧设备，请直接调用 HikCameraClient.find_devices()。

        Returns:
            每台设备一个 dict，包含 ``type``、``id``（device_index）、
            ``name`` 以及 ``default_stream_profile``（占位，无法在枚举阶段获取）。
        """
        from .hik_camera_sdk import HikCameraSDK

        raw = HikCameraSDK.find_devices()
        result = []
        for d in raw:
            result.append(
                {
                    "type": "hik",
                    "id": d["index"],
                    "name": f"HikCamera #{d['index']} ({d.get('type','?')}) {d.get('model','')}",
                    "serial": d.get("serial", ""),
                    "connection": d.get("ip") or d.get("user_id", ""),
                    "default_stream_profile": {},
                }
            )
        return result

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """连接相机并启动后台读取线程。

        Args:
            warmup: 若为 True，在返回前等待 ``warmup_s`` 秒直到拿到首帧。

        Raises:
            DeviceAlreadyConnectedError: 相机已连接。
            ConnectionError: 相机或 zerorpc 服务端不可达。
        """
        if self.config.mode == "local":
            from .hik_camera_sdk import HikCameraSDK

            sdk = HikCameraSDK(self.config.device_index)
            sdk.connect()
            self._backend = sdk
        else:
            from .hik_camera_client import HikCameraClient

            client = HikCameraClient(
                host=self.config.host,  # type: ignore[arg-type]
                port=self.config.port,
                heartbeat=self.config.heartbeat_s,
            )
            client.open_camera()
            self._backend = client

        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            deadline = time.time() + self.warmup_s
            while time.time() < deadline:
                try:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                except TimeoutError:
                    pass
                time.sleep(0.05)
            with self._frame_lock:
                if self._latest_frame is None:
                    raise ConnectionError(f"{self} 预热期间未获取到图像帧")

        logger.info(f"{self} 已连接")

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """同步采集一帧（等待下一帧到来）。

        Returns:
            shape (H, W, 3) uint8 numpy 数组，颜色空间由 ``color_mode`` 决定。

        Raises:
            DeviceNotConnectedError: 相机未连接。
            RuntimeError: 读取线程未运行。
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} 读取线程未运行")

        self._new_frame_event.clear()
        return self.async_read(timeout_ms=10_000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """返回最近一帧（若缓冲区无新帧则等待）。

        Args:
            timeout_ms: 等待新帧的最大毫秒数。

        Returns:
            shape (H, W, 3) uint8 numpy 数组。

        Raises:
            DeviceNotConnectedError: 相机未连接。
            TimeoutError: 超时未收到新帧。
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} 读取线程未运行")

        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"{self} 等待帧超时（{timeout_ms:.0f} ms），线程存活: {self._thread.is_alive()}"
            )

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} 内部错误：Event 已置位但 latest_frame 为 None")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """立即返回当前缓冲帧（非阻塞，帧可能有一定延迟）。

        Args:
            max_age_ms: 帧最大允许年龄（毫秒），超出则抛 TimeoutError。

        Raises:
            DeviceNotConnectedError: 相机未连接。
            TimeoutError: 缓冲帧过旧。
            RuntimeError: 尚未采集到任何帧。
        """
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError(f"{self} 读取线程未运行")

        with self._frame_lock:
            frame = self._latest_frame
            ts = self._latest_timestamp

        if frame is None or ts is None:
            raise RuntimeError(f"{self} 尚未采集到任何帧")

        age_ms = (time.perf_counter() - ts) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} 缓冲帧过旧: {age_ms:.1f} ms（允许 {max_age_ms} ms）")

        return frame

    def disconnect(self) -> None:
        """停止后台线程并关闭相机连接。

        Raises:
            DeviceNotConnectedError: 相机未连接。
        """
        if not self.is_connected and self._thread is None:
            raise DeviceNotConnectedError(f"{self} 未连接")

        self._stop_read_thread()

        if self._backend is not None:
            try:
                if self.config.mode == "local":
                    self._backend.disconnect()
                else:
                    self._backend.close_camera()
                    self._backend.close()
            except Exception as e:
                logger.warning(f"{self} 关闭 backend 异常: {e}")
            self._backend = None

        logger.info(f"{self} 已断开")

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _read_from_hardware(self) -> NDArray[Any]:
        """从 backend 拉取原始 BGR 帧。"""
        if self.config.mode == "local":
            return self._backend.capture_image()
        else:
            return self._backend.get_frame()

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """应用颜色转换与旋转。"""
        import cv2

        if self.color_mode == ColorMode.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_mode != ColorMode.BGR:
            raise ValueError(f"不支持的 color_mode: {self.color_mode}")

        if self.rotation is not None:
            image = cv2.rotate(image, self.rotation)

        return image

    def _read_loop(self) -> None:
        """后台线程主循环：持续采帧并更新缓冲区。"""
        if self._stop_event is None:
            raise RuntimeError(f"{self} stop_event 未初始化")

        failure_count = 0
        while not self._stop_event.is_set():
            try:
                raw = self._read_from_hardware()
                processed = self._postprocess_image(raw)
                ts = time.perf_counter()

                with self._frame_lock:
                    self._latest_frame = processed
                    self._latest_timestamp = ts
                self._new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning(f"{self} 后台读帧异常（第{failure_count}次）: {e}")
                else:
                    logger.error(f"{self} 连续读帧失败超过上限，退出读取循环")
                    break

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self._stop_event = Event()
        self._thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self._thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._stop_event = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None
            self._new_frame_event.clear()
