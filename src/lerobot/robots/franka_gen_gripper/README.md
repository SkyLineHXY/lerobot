# Franka + Gen Controller UMI Gripper

This robot module integrates a **Franka Emika 7-DoF robotic arm** with a **Gen Controller UMI gripper** into a single LeRobot-compatible `Robot` class.

## Architecture

```
FrankaGenGripper (Robot)
├── Franka (Robot)                    ← 7-DoF arm via zerorpc
│   └── FrankaInterfaceClient         ← RPC to franka_interface_server on NUC
└── UmiGripper (Robot)                ← 1-DoF gripper via serial + V4L2
    ├── DataBus                       ← Serial 921600 baud, encoder/tactile threads
    └── CameraCapture                 ← 3x V4L2 cameras, MJPEG, ~30 FPS
```

## Hardware Setup

### Franka Arm

1. The Franka arm is controlled via `franka_interface_server` running on a real-time NUC connected to the arm via Ethernet.
2. The user machine communicates with the NUC over zerorpc (default port 4242).
3. Ensure the NUC is reachable at the configured `robot_ip`.

### UMI Gripper

1. Connect the gripper via USB (USB 3.0 required for cameras).
2. Install udev rules to create stable device symlinks:

```bash
sudo cp config/99-usb-serial.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

3. After setup you should see:
   - Serial: `/dev/ttyDeviceLeft` or `/dev/ttyDeviceRight`
   - Cameras: `/dev/{left,right}_video_{0,1,2}_{main,sec}`

## Installation

```bash
# From the lerobot root:
pip install -e ".[dev]"

# Additional dependencies for UMI gripper:
pip install pyserial opencv-python-headless

# For Franka arm:
pip install zerorpc
```

## Usage

### CLI

```bash
# Teleoperate with default settings (left gripper)
lerobot-teleoperate --robot.type=franka_gen_gripper \
    --robot.robot_ip=192.168.1.104 \
    --robot.gripper_side=left

# Record a dataset
lerobot-record --robot.type=franka_gen_gripper \
    --robot.robot_ip=192.168.1.104 \
    --robot.gripper_side=left \
    --dataset.repo_id=your_user/franka_umi_dataset
```

### Python API

```python
from lerobot.robots.franka_gen_gripper import FrankaGenGripper, FrankaGenGripperConfig

config = FrankaGenGripperConfig(
    id="my_robot",
    robot_ip="192.168.1.104",
    gripper_side="left",
    gripper_camera_width=320,
    gripper_camera_height=240,
    gripper_enable_tactile=False,
)

robot = FrankaGenGripper(config)
robot.connect()

# Read observations
obs = robot.get_observation()
# obs contains:
#   "ee_pose.x", "ee_pose.y", "ee_pose.z",
#   "ee_pose.rx", "ee_pose.ry", "ee_pose.rz",  (Franka EE pose)
#   "gripper.pos",                                (gripper distance, meters)
#   "cam_0", "cam_1", "cam_2",                   (camera images, np.ndarray)

# Send actions
robot.send_action({
    "ee_pose.x": 0.4,
    "ee_pose.y": 0.0,
    "ee_pose.z": 0.3,
    "ee_pose.rx": 3.14,
    "ee_pose.ry": 0.0,
    "ee_pose.rz": 0.0,
    "gripper.pos": 0.05,   # 5cm opening
})

# Reset to home
robot.reset()

robot.disconnect()
```

## Observation & Action Space

### Observations

| Key | Type | Description |
|-----|------|-------------|
| `ee_pose.x` | `float` | End-effector X position (meters) |
| `ee_pose.y` | `float` | End-effector Y position (meters) |
| `ee_pose.z` | `float` | End-effector Z position (meters) |
| `ee_pose.rx` | `float` | End-effector rotation X (radians) |
| `ee_pose.ry` | `float` | End-effector rotation Y (radians) |
| `ee_pose.rz` | `float` | End-effector rotation Z (radians) |
| `gripper.pos` | `float` | Gripper opening distance (0.0 ~ 0.103 meters) |
| `cam_0` | `ndarray (H,W,3)` | Center camera image |
| `cam_1` | `ndarray (H,W,3)` | Left camera image |
| `cam_2` | `ndarray (H,W,3)` | Right camera image |
| `tactile_left` | `ndarray (500,)` | Left tactile pad (optional) |
| `tactile_right` | `ndarray (500,)` | Right tactile pad (optional) |

### Actions

| Key | Type | Description |
|-----|------|-------------|
| `ee_pose.x` | `float` | Target X position |
| `ee_pose.y` | `float` | Target Y position |
| `ee_pose.z` | `float` | Target Z position |
| `ee_pose.rx` | `float` | Target rotation X |
| `ee_pose.ry` | `float` | Target rotation Y |
| `ee_pose.rz` | `float` | Target rotation Z |
| `gripper.pos` | `float` | Target opening distance (clamped to [0.0, 0.103]) |

## Configuration

All config fields can be set via CLI flags or Python:

| Field | Default | Description |
|-------|---------|-------------|
| `robot_ip` | `192.168.1.104` | Franka NUC IP address |
| `control_mode` | `cartesian_impedance` | Franka control mode |
| `gripper_side` | `left` | Gripper side (`left` / `right`) |
| `gripper_serial_port` | auto | Serial port (auto from udev) |
| `gripper_camera_width` | `320` | Camera frame width |
| `gripper_camera_height` | `240` | Camera frame height |
| `gripper_camera_count` | `3` | Number of cameras |
| `gripper_enable_tactile` | `False` | Enable tactile sensor data |
| `gripper_encoder_freq` | `30.0` | Encoder polling rate (Hz) |

## Troubleshooting

**Serial port not found**: Ensure udev rules are installed and gripper USB cable is connected. Check `ls /dev/ttyDevice*`.

**Cameras not opening**: Check `v4l2-ctl --list-devices` and verify symlinks exist under `/dev/*_video_*`. USB 3.0 is required.

**Franka connection refused**: Ensure `franka_interface_server` is running on the NUC and the IP/port are correct.

**Permission denied on serial/video**: Run `sudo chmod 666 /dev/ttyDevice*` and `sudo chmod 666 /dev/*_video_*`, or add your user to the `dialout` and `video` groups.
