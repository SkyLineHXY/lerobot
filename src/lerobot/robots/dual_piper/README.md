# Dual Piper Robot

这是一个双臂Piper机器人的实现，支持14个关节（每个手臂7个关节）和多个相机。

## 架构设计

### 关节命名
- **左臂**: joint_1 到 joint_7
- **右臂**: joint_8 到 joint_14

### 硬件接口
- **左臂CAN总线**: `can_port_left` (默认: 'can_left')
- **右臂CAN总线**: `can_port_right` (默认: 'can_right')

### 相机配置
- `cam_left_wrist`: 左手腕相机
- `cam_right_wrist`: 右手腕相机
- `cam_top`: 顶部相机

## 数据格式

根据 `hdf5_to_lerobotv3_dualpiper.py` 的数据集格式：

### 观测空间 (observation.state)
- **形状**: (14,)
- **内容**: 14个关节的位置 [joint_1.pos, ..., joint_14.pos]
- **类型**: float32

### 动作空间 (action)
- **形状**: (14,)
- **内容**: 14个关节的目标位置 [joint_1.pos, ..., joint_14.pos]
- **类型**: float32

### 图像观测 (observation.images)
- `observation.images.cam_left_wrist`: 左手腕相机图像
- `observation.images.cam_right_wrist`: 右手腕相机图像
- `observation.images.cam_top`: 顶部相机图像

## 使用示例

### 基本连接和控制

```python
from lerobot.robots.dual_piper import DualPiper, DualPIPERConfig

# 创建配置
config = DualPIPERConfig(
    can_port_left='can_left',
    can_port_right='can_right'
)

# 初始化机器人
robot = DualPiper(config)

# 连接并校准
robot.connect(calibrate=True)

# 获取观测
obs = robot.get_observation()
print(f"Left arm joint 1: {obs['joint_1.pos']}")
print(f"Right arm joint 8: {obs['joint_8.pos']}")

# 发送动作
action = {
    'joint_1.pos': 0.0,  # 左臂关节1
    'joint_2.pos': 0.0,  # 左臂关节2
    # ... 更多左臂关节
    'joint_8.pos': 0.0,  # 右臂关节1
    'joint_9.pos': 0.0,  # 右臂关节2
    # ... 更多右臂关节
}
robot.send_action(action)

# 断开连接
robot.disconnect()
```

### 自定义相机配置

```python
from lerobot.cameras.realsense import RealSenseCameraConfig

config = DualPIPERConfig(
    cameras={
        "cam_left_wrist": RealSenseCameraConfig(
            serial_number_or_name="148522072680",
            fps=60,
            width=640,
            height=480,
        ),
        "cam_right_wrist": RealSenseCameraConfig(
            serial_number_or_name="327122074756",
            fps=60,
            width=640,
            height=480,
        ),
        "cam_top": RealSenseCameraConfig(
            serial_number_or_name="327122074757",
            fps=30,
            width=1280,
            height=720,
        ),
    }
)
```

## 与单臂Piper的区别

| 特性 | 单臂Piper | 双臂Dual Piper |
|------|-----------|---------------|
| 关节数量 | 7 | 14 |
| 电机总线 | 1个 (bus) | 2个 (bus_left, bus_right) |
| CAN端口 | 1个 | 2个 |
| 状态维度 | (7,) | (14,) |
| 动作维度 | (7,) | (14,) |

## 与数据集转换工具的对应关系

`hdf5_to_lerobotv3_dualpiper.py` 中的配置参数：
- `state_dim`: 14 (对应14个关节)
- `action_dim`: 14 (对应14个关节动作)
- `camera_names`: ["cam_left_wrist", "cam_right_wrist", "cam_top"]
- `fps`: 机器人控制频率

## 注意事项

1. **CAN总线配置**: 确保系统中正确配置了 `can_left` 和 `can_right` 两个CAN接口
2. **相机序列号**: 需要根据实际硬件修改RealSense相机的序列号
3. **校准**: 首次使用或重新启动后，建议执行校准操作
4. **同步控制**: 两个手臂的动作是并行发送的，确保动作指令的一致性

## 文件结构

```
dual_piper/
├── __init__.py              # 模块导出
├── config_dual_piper.py     # 配置类定义
├── dual_piper.py           # 主机器人类实现
└── README.md               # 本文档
```

## 开发参考

- 基于 `lerobot.robots.piper` 的单臂实现
- 数据格式参考 `hdf5_to_lerobotv3_dualpiper.py`
- 遵循 LeRobot v3.0 数据集规范

