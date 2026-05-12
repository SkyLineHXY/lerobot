"""FrankaMultiCam 组合机器人测试脚本。

测试 Franka（EE 位姿 + 夹爪）+ 多路相机（Hik + OpenCVZmq）的组合行为。

用法::

    # 最小：仅 Franka（无相机）
    python -m lerobot.robots.franka_multicam.test_franka_multicam \\
        --robot-ip 192.168.172.252

    # 完整：Franka + Hik + OpenCVZmq
    python -m lerobot.robots.franka_multicam.test_franka_multicam \\
        --robot-ip 192.168.172.252 \\
        --hik-host 192.168.172.134 --hik-port 4243 \\
        --opencv-host 192.168.172.134 --opencv-port 4244

按 q 或 ESC 退出预览。
"""

import argparse
import logging
import time

import cv2
import numpy as np

from lerobot.cameras.hik_camera.configuration_hik import HikCameraConfig
from lerobot.cameras.opencv_zmq.configuration_opencv_zmq import OpenCVZmqCameraConfig
from .config_franka_multicam import FrankaMultiCamConfig
from .franka_multicam import FrankaMultiCam

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FrankaMultiCam 组合机器人测试")
    p.add_argument("--robot-ip", default="192.168.172.252", help="Franka NUC IP")
    p.add_argument("--robot-port", type=int, default=4242, help="Franka zerorpc 端口")
    p.add_argument("--no-gripper", action="store_true", help="禁用夹爪")
    p.add_argument("--hik-host", default=None, help="Hik camera NUC IP（不指定则跳过 Hik）")
    p.add_argument("--hik-port", type=int, default=4243, help="Hik ZMQ PUB 端口")
    p.add_argument("--opencv-host", default=None, help="OpenCVZmq camera NUC IP（不指定则跳过 OpenCVZmq）")
    p.add_argument("--opencv-port", type=int, default=4244, help="OpenCVZmq ZMQ PUB 端口")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cameras: dict = {}

    if args.hik_host:
        cameras["front"] = HikCameraConfig(
            mode="remote",
            host=args.hik_host,
            port=args.hik_port,
            topic="hik",
            fps=30,
            width=1280,
            height=720,
        )

    if args.opencv_host:
        cameras["wrist"] = OpenCVZmqCameraConfig(
            index_or_path=0,
            mode="remote",
            host=args.opencv_host,
            port=args.opencv_port,
            topic="opencv",
            fps=30,
            width=640,
            height=480,
        )

    config = FrankaMultiCamConfig(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        use_gripper=not args.no_gripper,
        cameras=cameras,
    )

    robot = FrankaMultiCam(config)

    print("\n" + "=" * 60)
    print("  FrankaMultiCam 测试")
    print("=" * 60)

    # ---- Static properties ----
    print("\n[1] observation_features:")
    for k, v in robot.observation_features.items():
        print(f"    {k}: {v}")
    print("\n[2] action_features:")
    for k, v in robot.action_features.items():
        print(f"    {k}: {v}")

    # ---- Connect ----
    print("\n[3] 连接中...")
    robot.connect()
    print(f"    已连接: {robot.is_connected}")

    # ---- Live loop ----
    print("\n[4] 实时预览（按 q/ESC 退出）...")
    frame_count = 0
    t0 = time.perf_counter()
    last_fps_print = t0

    window_names = list(cameras.keys())

    try:
        while True:
            obs = robot.get_observation()
            frame_count += 1

            # 打印状态
            now = time.perf_counter()
            if now - last_fps_print >= 1.0:
                fps = frame_count / (now - t0)
                pose_str = " ".join(
                    f"{obs.get(k, float('nan')):.4f}" for k in
                    ["ee_pose.x", "ee_pose.y", "ee_pose.z", "ee_pose.rx", "ee_pose.ry", "ee_pose.rz"]
                )
                gripper_str = f"  gripper={obs.get('gripper.width', 'N/A')}"
                print(f"\r  FPS={fps:.1f}  EE=[{pose_str}]{gripper_str}  ", end="")

                # 叠加 FPS 到帧上
                for cam_key in window_names:
                    frame = obs.get(cam_key)
                    if frame is not None:
                        disp = frame.copy()
                        cv2.putText(
                            disp, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                        )
                        cv2.imshow(cam_key, disp)

                last_fps_print = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                print("\n  用户退出")
                break

    except KeyboardInterrupt:
        print("\n  Ctrl+C 退出")
    finally:
        print("\n[5] 断开连接...")
        robot.disconnect()
        cv2.destroyAllWindows()
        print("  完成")


if __name__ == "__main__":
    main()
