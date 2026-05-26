"""OpenCVZmqCamera：继承 OpenCVCamera，新增 ZMQ PUB/SUB 远程拉流能力。

- ``local`` 模式：等同 OpenCVCamera，直接操作本机摄像头。可选 publish=True 同时广播。
- ``remote`` 模式：通过 ZMQ SUB 订阅远程帧，不打开本地 VideoCapture。
"""

import logging
import time

import numpy as np
from numpy.typing import NDArray

from .._zmq_utils import ZMQFramePublisher, ZMQFrameSubscriber
from ..opencv.camera_opencv import OpenCVCamera
from .configuration_opencv_zmq import OpenCVZmqCameraConfig

logger = logging.getLogger(__name__)


class OpenCVZmqCamera(OpenCVCamera):
    """OpenCV 摄像头 + ZMQ PUB/SUB 远程拉流。

    local 模式完全复用 ``OpenCVCamera`` 的 connect / 取帧 / 缓冲逻辑；
    若 publish=True，会在每帧采集后额外 PUB 一份编码后的帧。

    remote 模式不操作本地 VideoCapture，改为通过 ZMQ SUB 从远程 PUB 端
    拉帧，其他行为（缓冲、read/async_read、颜色空间、旋转）与父类一致。
    """

    def __init__(self, config: OpenCVZmqCameraConfig):
        super().__init__(config)
        self._zmq_config = config
        self._subscriber: ZMQFrameSubscriber | None = None
        self._publisher: ZMQFramePublisher | None = None

    @property
    def is_connected(self) -> bool:
        """remote 模式检查 subscriber + 后台线程，local 模式委托父类。"""
        if self._zmq_config.mode == "remote":
            return self._subscriber is not None and self.thread is not None and self.thread.is_alive()
        return super().is_connected

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self, warmup: bool = True) -> None:
        """连接相机。

        local 模式：调用父类 connect（打开 VideoCapture + 启动后台线程）。
        remote 模式：创建 ZMQFrameSubscriber + 启动后台线程。
        """
        if self._zmq_config.mode == "local":
            super().connect(warmup=warmup)

            if self._zmq_config.publish:
                self._publisher = ZMQFramePublisher(
                    host=self._zmq_config.bind_addr,
                    port=self._zmq_config.port,
                    topic=self._zmq_config.topic,
                    wire_encoding=self._zmq_config.wire_encoding,
                    jpeg_quality=self._zmq_config.jpeg_quality,
                )
        else:
            if self.is_connected:
                return

            self._subscriber = ZMQFrameSubscriber(
                host=self._zmq_config.host,  # type: ignore[arg-type]
                port=self._zmq_config.port,
                topic=self._zmq_config.topic,
                recv_timeout_ms=self._zmq_config.recv_timeout_ms,
                conflate=self._zmq_config.sub_conflate,
            )

            self._start_read_thread()

            if warmup and self.warmup_s > 0:
                deadline = time.time() + self.warmup_s
                while time.time() < deadline:
                    try:
                        self.async_read(timeout_ms=self.warmup_s * 1000)
                    except TimeoutError:
                        pass
                    time.sleep(0.05)
                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} 预热期间未获取到图像帧")

            logger.info("%s 已连接 (remote)", self)

    def disconnect(self) -> None:
        """断开连接。"""
        if self._zmq_config.mode == "local":
            if self._publisher is not None:
                self._publisher.close()
                self._publisher = None
            super().disconnect()
        else:
            if not self.is_connected and self.thread is None:
                return

            self._stop_read_thread()

            if self._subscriber is not None:
                self._subscriber.close()
                self._subscriber = None

            logger.info("%s 已断开 (remote)", self)

    # ------------------------------------------------------------------
    # 帧读取
    # ------------------------------------------------------------------

    def _read_from_hardware(self) -> NDArray:
        """local 模式走父类 OpenCV，remote 模式走 ZMQ SUB。"""
        if self._zmq_config.mode == "local":
            return super()._read_from_hardware()
        else:
            if self._subscriber is None:
                raise RuntimeError(f"{self} subscriber 未初始化")
            return self._subscriber.get_frame()

    # ------------------------------------------------------------------
    # 后台线程（remote 模式复用父类 _read_loop，但多 PUB）
    # ------------------------------------------------------------------

    def _read_loop(self) -> None:
        """后台线程主循环。

        local+publish 模式：父类拉帧后多一次 PUB send。
        remote 模式：行为与父类一致（_read_from_hardware 已指向 subscriber）。
        """
        if self._zmq_config.mode == "local" and self._publisher is not None:
            if self.stop_event is None:
                raise RuntimeError(f"{self}: stop_event 未初始化")

            failure_count = 0
            while not self.stop_event.is_set():
                try:
                    raw_frame = self.videocapture.read()[1]
                    processed_frame = self._postprocess_image(raw_frame)
                    capture_time = time.perf_counter()

                    with self.frame_lock:
                        self.latest_frame = processed_frame
                        self.latest_timestamp = capture_time
                    self.new_frame_event.set()

                    # 额外 PUB 广播
                    try:
                        self._publisher.send(processed_frame, timestamp=time.time())
                    except Exception as e:
                        logger.warning("%s PUB 发送异常: %s", self, e)

                    failure_count = 0
                except Exception as e:
                    failure_count += 1
                    if failure_count <= 10:
                        logger.warning("%s 后台读帧异常（第%d次）: %s", self, failure_count, e)
                    else:
                        raise RuntimeError(f"{self} 连续读帧失败超过上限") from e
        else:
            super()._read_loop()
