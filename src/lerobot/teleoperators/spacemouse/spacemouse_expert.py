import multiprocessing
import time
from typing import Tuple

import numpy as np

from . import pyspacemouse


class SpaceMouseExpert:
    """SpaceMouse 硬件读取接口。

    在后台 daemon 进程中持续轮询 HID 设备，
    通过 multiprocessing.Manager().dict() 与主进程共享最新状态。
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager 共享状态
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0, 0, 0]

        self.process = multiprocessing.Process(target=self._read_spacemouse, daemon=True)
        self.process.start()

    def _read_spacemouse(self) -> None:
        while True:
            state = pyspacemouse.read_all()

            # 防御性：设备短暂断开时 read_all 可能返回 None 或空列表
            if not state:
                time.sleep(0.005)
                continue

            action: list[float] = [0.0] * 6
            buttons: list[int] = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw,
                ]
                buttons = list(state[0].buttons) + list(state[1].buttons)
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                ]
                buttons = list(state[0].buttons)

            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

            # 限制轮询速率（约 200 Hz），避免占满 CPU
            time.sleep(0.005)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """返回最新动作向量与按钮状态。"""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action, dtype=np.float32), list(buttons)

    def close(self) -> None:
        """关闭后台进程并释放 HID 设备。"""
        self.process.join(timeout=0.5)
        if self.process.is_alive():
            self.process.terminate()
        try:
            pyspacemouse.close()
        except Exception:
            pass
        try:
            self.manager.shutdown()
        except Exception:
            pass