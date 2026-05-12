"""海康工业相机 SDK 薄封装，仅负责设备连接与图像帧采集。"""

import logging
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof, string_at
from typing import Any

import numpy as np

from .MvImport.CameraParams_const import MV_GIGE_DEVICE, MV_USB_DEVICE
from .MvImport.CameraParams_header import (
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_PIXEL_CONVERT_PARAM,
    MVCC_INTVALUE,
    MvCamera,
)
from .MvImport.MvErrorDefine_const import (
    MV_E_ABNORMAL_IMAGE,
    MV_E_BUFOVER,
    MV_E_GC_TIMEOUT,
    MV_E_HANDLE,
    MV_E_NODATA,
    MV_E_PARAMETER,
    MV_E_RESOURCE,
)
from .MvImport.PixelType_header import (
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono16,
    PixelType_Gvsp_RGB8_Packed,
    PixelType_Gvsp_YUV422_Packed,
    PixelType_Gvsp_YUV422_YUYV_Packed,
)

# 需要 MV_TRIGGER_MODE_OFF / MV_GAIN_MODE_CONTINUOUS 的值，直接用数值避免额外 import
_MV_TRIGGER_MODE_OFF = 0
_MV_GAIN_MODE_CONTINUOUS = 2
_MV_ACCESS_EXCLUSIVE = 1
_MV_FRAME_OUT_INFO_EX = None  # 懒加载，运行时从 CameraParams_header 取

logger = logging.getLogger(__name__)


def _decode_bytes(buf) -> str:
    """将 c_ubyte 数组解码为字符串（去尾零）。"""
    return bytes(buf).rstrip(b"\x00").decode("utf-8", errors="ignore")


class HikCameraSDK:
    """海康工业相机 SDK 封装，仅暴露设备连接与帧采集接口。

    不包含图像检测、坐标变换等业务逻辑。
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self._is_connected: bool = False
        self.data_buf = None
        self.nDataSize: int = 0
        self.reconnect_attempts: int = 0
        self.max_reconnect_attempts: int = 3

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ------------------------------------------------------------------
    # 设备枚举
    # ------------------------------------------------------------------

    @staticmethod
    def find_devices() -> list[dict[str, Any]]:
        """枚举当前可见的海康相机设备。

        Returns:
            每台设备一个 dict，包含 index / type / model / serial，
            GigE 设备额外包含 ip，USB3 设备额外包含 user_id。
        """
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer, device_list)
        if ret != 0:
            logger.warning(f"枚举设备失败，错误码: {ret}")
            return []

        devices: list[dict[str, Any]] = []
        for i in range(device_list.nDeviceNum):
            info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            d: dict[str, Any] = {"index": i}
            if info.nTLayerType == MV_GIGE_DEVICE:
                gige = info.SpecialInfo.stGigEInfo
                raw_ip = gige.nCurrentIp
                ip = f"{(raw_ip >> 24) & 0xFF}.{(raw_ip >> 16) & 0xFF}.{(raw_ip >> 8) & 0xFF}.{raw_ip & 0xFF}"
                d.update(
                    {
                        "type": "GigE",
                        "ip": ip,
                        "model": _decode_bytes(gige.chModelName),
                        "serial": _decode_bytes(gige.chSerialNumber),
                        "user_id": _decode_bytes(gige.chUserDefinedName),
                    }
                )
            elif info.nTLayerType == MV_USB_DEVICE:
                usb = info.SpecialInfo.stUsb3VInfo
                d.update(
                    {
                        "type": "USB3",
                        "model": _decode_bytes(usb.chModelName),
                        "serial": _decode_bytes(usb.chSerialNumber),
                        "user_id": _decode_bytes(usb.chUserDefinedName),
                    }
                )
            else:
                d["type"] = f"Unknown(0x{info.nTLayerType:04X})"
            devices.append(d)

        return devices

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """连接指定索引的相机并开始取流。

        Raises:
            ConnectionError: 枚举失败、索引越界或设备打开失败。
        """
        if self._is_connected:
            self.disconnect()

        tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer, self.device_list)
        if ret != 0:
            raise ConnectionError(f"枚举设备失败，错误码: {ret}")

        if self.device_list.nDeviceNum == 0:
            raise ConnectionError("未发现任何海康相机设备")

        if self.device_index >= self.device_list.nDeviceNum:
            raise ConnectionError(
                f"相机索引 {self.device_index} 超出范围，共发现 {self.device_list.nDeviceNum} 台设备"
            )

        stDeviceInfo = cast(
            self.device_list.pDeviceInfo[self.device_index], POINTER(MV_CC_DEVICE_INFO)
        ).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceInfo)
        if ret != 0:
            raise ConnectionError(f"创建设备句柄失败，错误码: {ret}")

        ret = self.cam.MV_CC_OpenDevice(_MV_ACCESS_EXCLUSIVE, 0)
        if ret != 0:
            hex_code = f"0x{ret & 0xFFFFFFFF:08X}"
            if ret == 0x80000003:
                raise ConnectionError(
                    f"打开设备失败 (MV_E_CALLORDER {hex_code})，设备可能已被其他进程独占，请先关闭占用进程"
                )
            elif ret == 0x80000203:
                raise ConnectionError(f"打开设备失败，访问被拒绝 (MV_E_ACCESS_DENIED {hex_code})")
            else:
                raise ConnectionError(f"打开设备失败，错误码: {ret} ({hex_code})")

        # 设置触发模式为 off，增益为连续自动
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", _MV_TRIGGER_MODE_OFF)
        if ret != 0:
            logger.warning(f"设置触发模式失败，错误码: {ret}")

        ret = self.cam.MV_CC_SetEnumValue("GainAuto", _MV_GAIN_MODE_CONTINUOUS)
        if ret != 0:
            logger.warning(f"设置增益模式失败，错误码: {ret}")

        # 获取数据包大小并分配缓冲区
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise ConnectionError(f"获取数据包大小失败，错误码: {ret}")

        self.nDataSize = stParam.nCurValue
        self.data_buf = (c_ubyte * self.nDataSize)()

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise ConnectionError(f"开始取流失败，错误码: {ret}")

        self._is_connected = True
        self.reconnect_attempts = 0
        logger.info(f"海康相机 #{self.device_index} 连接成功")

    def disconnect(self) -> None:
        """停止取流并释放设备句柄。"""
        try:
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                logger.warning(f"停止取流失败，错误码: {ret}")
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                logger.warning(f"关闭设备失败，错误码: {ret}")
            ret = self.cam.MV_CC_DestroyHandle()
            if ret != 0:
                logger.warning(f"销毁句柄失败，错误码: {ret}")
        except Exception as e:
            logger.error(f"断开相机连接异常: {e}")
        finally:
            self._is_connected = False
            logger.info(f"海康相机 #{self.device_index} 已断开")

    def reconnect(self) -> bool:
        """重新连接相机（有限次数）。"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"重连次数已达上限 {self.max_reconnect_attempts}，停止重连")
            return False
        self.reconnect_attempts += 1
        logger.info(f"尝试重连相机，第 {self.reconnect_attempts} 次")
        self.disconnect()
        time.sleep(0.5)
        try:
            self.connect()
            logger.info("相机重连成功")
            return True
        except ConnectionError as e:
            logger.warning(f"相机重连失败: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """返回相机当前状态信息。"""
        status: dict[str, Any] = {
            "is_connected": self._is_connected,
            "device_index": self.device_index,
            "reconnect_attempts": self.reconnect_attempts,
            "max_reconnect_attempts": self.max_reconnect_attempts,
        }
        if self._is_connected:
            try:
                w = MVCC_INTVALUE()
                memset(byref(w), 0, sizeof(MVCC_INTVALUE))
                h = MVCC_INTVALUE()
                memset(byref(h), 0, sizeof(MVCC_INTVALUE))
                self.cam.MV_CC_GetIntValue("Width", w)
                self.cam.MV_CC_GetIntValue("Height", h)
                status["width"] = int(w.nCurValue)
                status["height"] = int(h.nCurValue)
            except Exception:
                pass
        return status

    # ------------------------------------------------------------------
    # 图像采集
    # ------------------------------------------------------------------

    def _get_bytes_per_pixel(self, pixel_type: int) -> int:
        """根据像素格式返回每像素字节数。"""
        mapping = {
            0: 1,
            PixelType_Gvsp_Mono8: 1,
            PixelType_Gvsp_Mono10: 2,
            PixelType_Gvsp_Mono12: 2,
            PixelType_Gvsp_Mono16: 2,
            PixelType_Gvsp_RGB8_Packed: 3,
            PixelType_Gvsp_BGR8_Packed: 3,
            PixelType_Gvsp_BayerGR8: 1,
            PixelType_Gvsp_BayerRG8: 1,
            PixelType_Gvsp_BayerGB8: 1,
            PixelType_Gvsp_BayerBG8: 1,
            PixelType_Gvsp_YUV422_Packed: 2,
            PixelType_Gvsp_YUV422_YUYV_Packed: 2,
        }
        bpp = mapping.get(pixel_type, 0)
        if bpp == 0:
            logger.warning(f"未知像素格式: {pixel_type}，假设每像素1字节")
            return 1
        return bpp

    def _get_pixel_format_name(self, pixel_type: int) -> str:
        """返回像素格式的可读名称（用于调试日志）。"""
        names = {
            0: "Unknown/Undefined",
            PixelType_Gvsp_Mono8: "Mono8",
            PixelType_Gvsp_RGB8_Packed: "RGB8_Packed",
            PixelType_Gvsp_BGR8_Packed: "BGR8_Packed",
            PixelType_Gvsp_BayerGR8: "BayerGR8",
            PixelType_Gvsp_BayerRG8: "BayerRG8",
            PixelType_Gvsp_BayerGB8: "BayerGB8",
            PixelType_Gvsp_BayerBG8: "BayerBG8",
        }
        return names.get(pixel_type, f"Unknown({pixel_type})")

    def capture_image(self) -> np.ndarray:
        """采集一帧图像。

        Returns:
            BGR uint8 numpy 数组，形状 (H, W, 3)。

        Raises:
            RuntimeError: 相机未连接或帧读取失败。
        """
        from .MvImport.CameraParams_header import MV_FRAME_OUT_INFO_EX

        if not self._is_connected:
            raise RuntimeError("相机未连接，无法采集图像")

        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.nDataSize, stFrameInfo, 1000)
        if ret != 0:
            hex_code = f"0x{ret & 0xFFFFFFFF:08X}"
            if ret == MV_E_NODATA:
                raise RuntimeError(f"相机无数据 (MV_E_NODATA {hex_code})，检查触发模式配置")
            elif ret == MV_E_BUFOVER:
                raise RuntimeError(f"缓冲区溢出 (MV_E_BUFOVER {hex_code})")
            elif ret == MV_E_GC_TIMEOUT:
                raise RuntimeError(f"获取图像超时 (MV_E_GC_TIMEOUT {hex_code})")
            elif ret == MV_E_HANDLE:
                self._is_connected = False
                raise RuntimeError(f"无效的设备句柄 (MV_E_HANDLE {hex_code})，相机可能已断开")
            elif ret == MV_E_PARAMETER:
                raise RuntimeError(f"参数错误 (MV_E_PARAMETER {hex_code})")
            elif ret == MV_E_RESOURCE:
                raise RuntimeError(f"资源不足 (MV_E_RESOURCE {hex_code})")
            elif ret == MV_E_ABNORMAL_IMAGE:
                raise RuntimeError(f"异常图像 (MV_E_ABNORMAL_IMAGE {hex_code})，可能丢包")
            else:
                self._is_connected = False
                raise RuntimeError(f"获取图像失败，错误码: {ret} ({hex_code})")

        fmt_name = self._get_pixel_format_name(stFrameInfo.enPixelType)
        logger.debug(
            f"获取帧: {stFrameInfo.nWidth}x{stFrameInfo.nHeight} {fmt_name} {stFrameInfo.nFrameLen}B"
        )

        return self._decode_frame(stFrameInfo)

    def _decode_frame(self, stFrameInfo) -> np.ndarray:
        """将原始帧缓冲区解码为 BGR numpy 数组。"""
        import cv2

        from .MvImport.CameraParams_header import MV_CC_PIXEL_CONVERT_PARAM

        w, h = stFrameInfo.nWidth, stFrameInfo.nHeight

        if stFrameInfo.enPixelType == PixelType_Gvsp_Mono8:
            data = np.frombuffer(self.data_buf, count=w * h, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)

        if stFrameInfo.enPixelType == PixelType_Gvsp_RGB8_Packed:
            data = np.frombuffer(self.data_buf, count=w * h * 3, dtype=np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)

        if stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed:
            return np.frombuffer(self.data_buf, count=w * h * 3, dtype=np.uint8).reshape(h, w, 3).copy()

        # 其他格式：尝试 SDK 格式转换为 BGR8
        fmt_name = self._get_pixel_format_name(stFrameInfo.enPixelType)
        logger.debug(f"使用 SDK 转换像素格式: {fmt_name}")

        n_dst = w * h * 3
        stConvert = MV_CC_PIXEL_CONVERT_PARAM()
        memset(byref(stConvert), 0, sizeof(stConvert))
        stConvert.nWidth = w
        stConvert.nHeight = h
        stConvert.pSrcData = cast(self.data_buf, POINTER(c_ubyte))
        stConvert.nSrcDataLen = stFrameInfo.nFrameLen
        stConvert.enSrcPixelType = stFrameInfo.enPixelType
        stConvert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        stConvert.pDstBuffer = (c_ubyte * n_dst)()
        stConvert.nDstBufferSize = n_dst

        ret = self.cam.MV_CC_ConvertPixelType(stConvert)
        if ret != 0:
            raise RuntimeError(
                f"像素格式转换失败 ({fmt_name} → BGR8)，错误码: {ret} (0x{ret & 0xFFFFFFFF:08X})"
            )

        buf = string_at(stConvert.pDstBuffer, n_dst)
        return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()

    def __del__(self) -> None:
        if self._is_connected:
            try:
                self.disconnect()
            except Exception:
                pass
