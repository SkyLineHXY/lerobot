#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""数据采集时给操作员看的实时视图：相机画面横向拼接 + 状态面板。

线程模型：OpenCV 的 HighGUI 只有从**主线程**才能可靠地建窗口和重绘（Anaconda 下尤其
明显，torch / pynput 各自会拉起自己的 Qt/X11，后台线程建的窗口会静默地不显示）。
所以 :meth:`start` 和 :meth:`render_once` 必须在主线程调，:meth:`update` 才是任意线程
安全的 —— 它只在锁里存下最新的 (images, status)。

缺 cv2 或没有显示环境时整个视图降级成 no-op，采集流程不会因为缺显示依赖而跑不起来。
"""

from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["StatusView"]


def _is_image(val) -> bool:
    shape = getattr(val, "shape", None)
    if shape is None:
        return False
    nd = len(shape)
    return nd == 2 or (nd == 3 and shape[2] in (1, 3, 4))


class StatusView:
    """相机画面 + 采集状态面板的 OpenCV 窗口。"""

    def __init__(
        self,
        camera_names: list | None = None,
        bgr: bool = False,
        max_width: int = 1280,
        window_name: str = "Piper Collect",
        panel_height: int = 300,
        tile_height: int = 360,
    ) -> None:
        # None = 有什么显示什么；给了列表就按它决定显示哪几路和顺序
        self.camera_names = camera_names
        # bgr=True 表示输入是 RGB、需要转成 BGR 给 cv2。lerobot 的相机默认输出 RGB。
        self.bgr = bgr
        self.max_width = max_width
        self.window_name = window_name
        self.panel_height = panel_height
        self.tile_height = tile_height

        self._lock = threading.Lock()
        self._latest_images: dict | None = None
        self._latest_status: dict | None = None
        self._enabled = False
        self._window_created = False
        self._render_errors = 0
        self._cv2 = None

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> StatusView:
        """建立 cv2 与窗口。必须在主线程调用。"""
        try:
            import cv2
        except Exception as exc:
            logger.warning("StatusView 已禁用：导入 OpenCV 失败 (%s)。", exc)
            return self

        # 没有显示环境就别碰 GUI 调用，否则会把 GUI 后端搞崩
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            logger.warning(
                "StatusView 已禁用：没有 DISPLAY/WAYLAND_DISPLAY。"
                "请在图形会话里运行（或 `ssh -X` 后设置 DISPLAY）。"
            )
            return self

        self._cv2 = cv2
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)  # 泵一次事件循环，窗口才会真的映射出来
        except Exception as exc:
            logger.warning("StatusView 已禁用：创建窗口 '%s' 失败 (%s)。", self.window_name, exc)
            self._cv2 = None
            return self

        self._window_created = True
        self._enabled = True
        logger.info("StatusView 已启动：窗口 '%s'（由主线程驱动渲染）", self.window_name)
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(self, images: dict | None, status: dict | None) -> None:
        """存下最新的图像与状态。非阻塞，任意线程可调。"""
        if not self._enabled:
            return
        with self._lock:
            self._latest_images = images
            self._latest_status = status

    def render_once(self) -> bool:
        """绘制并泵一次 GUI 事件循环，必须在主线程反复调用。

        操作员在窗口里按 q/ESC 时返回 False，其余情况（含视图被禁用）返回 True。
        """
        if not self._enabled:
            return True
        cv2 = self._cv2
        with self._lock:
            images = self._latest_images
            status = dict(self._latest_status) if self._latest_status else {}
        try:
            canvas = self._render(images, status)
            if canvas is not None:
                cv2.imshow(self.window_name, canvas)
            self._render_errors = 0
        except Exception as exc:
            # 渲染失败不能打断采集，也不能刷屏，只报前三次
            self._render_errors += 1
            if self._render_errors <= 3:
                logger.warning("StatusView 渲染出错：%s", exc)
        try:
            key = cv2.waitKey(1) & 0xFF
        except Exception:
            key = 0
        return key not in (ord("q"), 27)

    def stop(self) -> None:
        if not self._enabled and not self._window_created:
            return
        self._enabled = False
        try:
            if self._cv2 is not None:
                self._cv2.destroyWindow(self.window_name)
                self._cv2.waitKey(1)
        except Exception:
            logger.debug("关闭 StatusView 窗口失败", exc_info=True)
        self._window_created = False

    # ------------------------------------------------------------------ 绘制
    def _select_images(self, images: dict) -> list:
        out = []
        if self.camera_names:
            for cam in self.camera_names:
                if cam in images and _is_image(images[cam]):
                    out.append((cam, np.asarray(images[cam])))
        else:
            for key, val in images.items():
                if _is_image(val):
                    out.append((key, np.asarray(val)))
        return out

    def _tile(self, frames: list) -> np.ndarray | None:
        cv2 = self._cv2
        if not frames:
            return None
        tiles = []
        for name, img in frames:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif self.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = np.ascontiguousarray(img)
            h, w = img.shape[:2]
            scale = self.tile_height / h
            img = cv2.resize(img, (max(1, int(w * scale)), self.tile_height))
            cv2.putText(img, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(img)
        strip = cv2.hconcat(tiles) if len(tiles) > 1 else tiles[0]
        if strip.shape[1] > self.max_width:
            scale = self.max_width / strip.shape[1]
            strip = cv2.resize(strip, (self.max_width, int(strip.shape[0] * scale)))
        return strip

    def _render(self, images: dict | None, status: dict) -> np.ndarray | None:
        cv2 = self._cv2
        strip = self._tile(self._select_images(images)) if images else None
        width = strip.shape[1] if strip is not None else max(640, self.max_width // 2)

        panel = np.zeros((self.panel_height, width, 3), dtype=np.uint8)
        self._draw_panel(panel, status)

        if strip is None:
            return panel
        return cv2.vconcat([strip, panel])

    def _draw_panel(self, panel: np.ndarray, status: dict) -> None:
        cv2 = self._cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        white = (235, 235, 235)
        grey = (150, 150, 150)
        yellow = (0, 230, 230)
        green = (60, 230, 60)
        red = (60, 60, 235)

        def put(text, x, y, color=white, scale=0.5, thick=1):
            cv2.putText(panel, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

        def put_fit(text, x, y, color, scale, thick, max_w):
            """字号自动缩小到不超过 max_w 像素为止。"""
            s = scale
            while s > 0.4:
                (tw, _th), _ = cv2.getTextSize(text, font, s, thick)
                if tw <= max_w:
                    break
                s -= 0.1
            cv2.putText(panel, text, (x, y), font, s, color, thick, cv2.LINE_AA)

        y = 24

        task = status.get("task")
        if task:
            put(f"TASK: {task}", 10, y, green, 0.6, 1)
            y += 26

        meta = []
        if status.get("step") is not None:
            meta.append(f"step {status['step']}")
        if status.get("fps") is not None:
            meta.append(f"{status['fps']:.1f} fps")
        if status.get("elapsed") is not None:
            elapsed = status["elapsed"]
            meta.append(f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}")
        if meta:
            put(" | ".join(meta), 10, y, grey, 0.5)
            y += 24

        if status.get("recording") is not None:
            rec = bool(status.get("recording"))
            back = bool(status.get("back"))
            ep = status.get("episode_index")
            buffered = status.get("buffered_frames")
            if rec:
                extra = ""
                if ep is not None and buffered is not None:
                    extra = f"   episode #{ep}  ({buffered} frames)"
                put(f"REC{extra}", 10, y, red, 0.6, 2)
            else:
                put("idle", 10, y, grey, 0.6, 2)
            put(
                "BACK=1" if back else "back=0",
                panel.shape[1] - 130,
                y,
                red if back else grey,
                0.6,
                2,
            )
            y += 26

            saved_eps = status.get("saved_episodes")
            saved_frames = status.get("saved_frames")
            if saved_eps is not None and saved_frames is not None:
                put(f"dataset: {saved_eps} episodes | {saved_frames} steps saved", 10, y, green, 0.5)
                y += 24

        # 子任务故意画得比别的行大约三倍，操作员隔着工位一眼就能看清
        skill_text = status.get("skill_text")
        if skill_text:
            put("SUBTASK:", 10, y, yellow, 0.6, 1)
            y += 36
            put_fit(skill_text, 10, y + 12, yellow, 1, 3, max_w=panel.shape[1] - 20)
