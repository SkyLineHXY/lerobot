# 海康工业相机驱动（lerobot Camera）

## 概述

本模块将海康工业相机适配为 lerobot 标准 `Camera` 接口，可直接用作 `lerobot-record` 的图像数据源，采集的帧将以 MP4 格式写入 `LeRobotDataset`。

支持两种部署模式：

| 模式 | 场景 | 说明 |
|------|------|------|
| `local` | 相机直连本机 | 在 NUC 上直接训练/调试时使用，无网络开销 |
| `remote` | 相机在 NUC，数据集采集在 PC | NUC 运行 `hik_camera_server.py`，PC 通过 zerorpc 拉流 |

## 文件结构

```
hik_camera/
├── __init__.py               # 导出 HikCamera, HikCameraConfig
├── configuration_hik.py      # HikCameraConfig（注册为 "hik"）
├── camera_hik.py             # HikCamera(Camera)：lerobot 接口主类
├── hik_camera_sdk.py         # HikCameraSDK：海康 SDK 薄封装（连接 + 取帧）
├── hik_camera_server.py      # zerorpc 服务端（运行于 NUC）
├── hik_camera_client.py      # zerorpc 客户端（HikCamera 内部使用）
├── MvImport/                 # 海康 MVS SDK Python 绑定（不修改）
├── requirements.txt          # 依赖列表
├── test_hik_camera.py        # 烟囱测试
└── README.md                 # 本文档
```

## lerobot Camera 接口

`HikCamera` 完整实现了 `lerobot.cameras.camera.Camera` 抽象：

| 方法 / 属性 | 说明 |
|-------------|------|
| `is_connected` (property) | 后台读取线程是否运行中 |
| `find_cameras()` (staticmethod) | 枚举本机可见的海康设备，返回 `list[dict]` |
| `connect(warmup=True)` | 建立连接并启动后台取帧线程；`warmup=True` 时等待曝光稳定 |
| `read()` | 同步阻塞，等待下一帧（适合精确时序控制循环） |
| `async_read(timeout_ms=200)` | 从缓冲区取最新未消费帧，超时抛 `TimeoutError` |
| `read_latest(max_age_ms=500)` | 立即返回当前缓冲帧（非阻塞，帧可能有延迟） |
| `disconnect()` | 停止线程并关闭设备 |

所有 `read*` 方法返回 `np.ndarray`，形状 `(H, W, 3)`，`dtype=uint8`，颜色空间由 `color_mode` 决定（默认 BGR）。

支持 Python 上下文管理器（`with HikCamera(...) as cam:`），退出时自动调用 `disconnect()`。

## 配置

```python
from lerobot.cameras.hik_camera import HikCameraConfig

# 远程模式（主要用法：PC 采集，相机在 NUC）
cfg = HikCameraConfig(
    mode="remote",
    host="192.168.1.10",   # NUC IP
    port=4243,
    device_index=0,
    fps=30,
    width=1280,
    height=720,
    color_mode="bgr",          # "bgr" 或 "rgb"
    rotation=0,                # 0 / 90 / 180 / -90
    wire_encoding="jpeg",      # "jpeg"（省带宽）或 "raw"（无损）
    jpeg_quality=90,
    heartbeat_s=30,
    warmup_s=1,
)

# 本地模式（NUC 上直接调试）
cfg_local = HikCameraConfig(mode="local", device_index=0, fps=30)
```

## 部署方式

### 架构图

```
[海康工业相机]
     │  USB / GigE
     ▼
[ NUC ] ─── hik_camera_server.py (zerorpc :4243)
              │
              │  TCP / LAN
              ▼
[ 数据集采集 PC ] ─── lerobot-record（HikCamera mode=remote）
```

### 步骤 1：NUC 侧启动服务

```bash
# 安装依赖（若尚未安装）
pip install zerorpc msgpack pyzmq opencv-python

# 启动服务（相机索引 0，JPEG 传输，质量 90）
python -m lerobot.cameras.hik_camera.hik_camera_server \
    --host 0.0.0.0 \
    --port 4243 \
    --device-index 0 \
    --wire-encoding jpeg \
    --jpeg-quality 90
```

服务启动后会自动连接相机并打印当前状态。

### 步骤 2：PC 侧采集数据集

```bash
lerobot-record \
    --robot.type=franka_gen_gripper \
    --robot.cameras='{"wrist": {"type": "hik", "mode": "remote", "host": "192.168.1.10", "port": 4243, "fps": 30, "width": 1280, "height": 720}}'
```

或在 Python 中直接使用：

```python
from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

cfg = HikCameraConfig(
    mode="remote", host="192.168.1.10", port=4243, fps=30, width=1280, height=720
)
with HikCamera(cfg) as cam:
    for i in range(100):
        frame = cam.read()   # (720, 1280, 3) uint8 BGR numpy array
```

### 本地模式（NUC 上直接调试）

```python
from lerobot.cameras.hik_camera import HikCamera, HikCameraConfig

with HikCamera(HikCameraConfig(mode="local", device_index=0, fps=30)) as cam:
    frame = cam.read()
    print(frame.shape)   # (H, W, 3)
```

## 注意事项

1. **SDK 依赖**：NUC 上必须安装海康 MVS SDK，`MvImport/` 目录中的 `.dll` / `.so` 需完整。
2. **zerorpc 心跳**：默认 `heartbeat_s=30`，若帧尺寸大或网络慢导致单帧耗时超过 30 s，需适当调大。
3. **带宽估算**：1280×720 JPEG90 约 100–200 KB/帧；30 fps 约 3–6 MB/s，千兆网绰绰有余。`raw` 模式约 2.8 MB/帧，需百兆以上。
4. **多相机**：当前服务端为单实例单相机；如需多路相机，启动多个服务端实例，每个分配不同端口。
5. **设备权限**：Linux 下若设备无法访问，检查 `/etc/udev/rules.d/` 中的 USB 权限规则。

## 故障排查

| 问题 | 可能原因 | 排查方法 |
|------|----------|----------|
| 相机连接失败 | 相机未上电 / IP 冲突 / 被其他进程独占 | 检查 MVS 客户端是否占用；用 `find_cameras()` 确认设备可见 |
| `JPEG 解码失败` | 网络丢包或 zerorpc 截断 | 降低帧率或换 `wire_encoding="raw"` 排查 |
| `TimeoutError` 频繁 | heartbeat 过小或 PC 处理太慢 | 增大 `heartbeat_s`；降低分辨率或帧率 |
| 图像色彩异常 | `color_mode` 与下游框架不匹配 | lerobot 内部用 BGR；若策略预期 RGB，设置 `color_mode="rgb"` |
| NUC 服务端无日志 | zerorpc 未安装 | `pip install zerorpc pyzmq msgpack` |
