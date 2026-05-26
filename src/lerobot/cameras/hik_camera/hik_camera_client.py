"""海康相机 ZeroMQ SUB 客户端，运行在数据集采集 PC 侧。

由 ``HikCamera`` 在 ``mode="remote"`` 时内部使用，不建议直接调用。
"""

from .._zmq_utils import ZMQFrameSubscriber

# 保持向后兼容的别名
HikCameraSubscriber = ZMQFrameSubscriber
