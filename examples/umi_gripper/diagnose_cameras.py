#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Diagnose why UmiGripper.get_observation() returns all-zero camera frames.

Probes three layers independently so you can tell where data flow stops:

    layer 1: raw V4L2 / OpenCV       -> can the OS open the device at all?
    layer 2: gen_con_sdk CameraCapture -> does the SDK enumerate it correctly?
    layer 3: UmiGripper wrapper       -> does the wrapper's capture thread push
                                          frames into _latest_frames before
                                          get_observation() reads them?

Usage:
    python examples/umi_gripper/diagnose_cameras.py --side left
    python examples/umi_gripper/diagnose_cameras.py --side left --skip-raw
"""

import argparse
import os
import time
from pathlib import Path

import cv2

from lerobot.robots.gen_gripper import UmiGripper, UmiGripperConfig
from lerobot.robots.gen_gripper.config_umi_gripper import SIDE_DEFAULTS
from lerobot.robots.gen_gripper.gen_con_sdk_python_release.scripts.camera import (
    CameraCapture,
)


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def list_video_devices() -> list[str]:
    return sorted(p for p in os.listdir("/dev") if p.startswith("video"))


def layer1_raw_v4l2(devices: list[str], save_dir: Path) -> None:
    """Open each device with cv2.VideoCapture directly, no SDK involved."""
    banner("Layer 1: raw cv2.VideoCapture per device")
    for dev in devices:
        if not os.path.exists(dev):
            print(f"  {dev}: missing")
            continue
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  {dev}: cv2 failed to open")
            continue
        # Force MJPG (matches what the SDK does)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        # Warm up
        for _ in range(5):
            cap.grab()
            time.sleep(0.01)
        ret, frame = cap.read()
        if not ret or frame is None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  {dev}: opened (negotiated {w}x{h}) but read() returned no frame")
        else:
            nz = bool(frame.any())
            print(f"  {dev}: read() shape={frame.shape}, nonzero={nz}")
            if nz:
                out = save_dir / f"raw_{Path(dev).name}.png"
                cv2.imwrite(str(out), frame)
        cap.release()


def layer2_sdk(side: str, video_devices: list[str], width: int, height: int, save_dir: Path) -> None:
    """Run the SDK's CameraCapture init alone, then poke each capture handle."""
    banner("Layer 2: gen_con_sdk CameraCapture")
    print(f"  resolutions=[({width},{height})], video_devices={len(video_devices)} entries")
    try:
        cam = CameraCapture(
            serial_port="",
            camera_count=len(video_devices),
            resolutions=[(width, height)],
            show_preview=False,
            video_devices=video_devices,
        )
    except SystemExit as e:
        print(f"  CameraCapture aborted via sys.exit({e.code})")
        return
    except Exception as e:
        print(f"  CameraCapture raised: {e!r}")
        return

    print(f"  SDK opened {len(cam.cameras)} camera(s)")
    for entry in cam.cameras:
        cap = entry["cap"]
        ret, frame = cap.read()
        if not ret or frame is None:
            print(
                f"    cam_id={entry['id']} dev={entry['dev']} "
                f"({entry['width']}x{entry['height']}): cap.read() returned no frame"
            )
            continue
        nz = bool(frame.any())
        print(
            f"    cam_id={entry['id']} dev={entry['dev']} "
            f"({entry['width']}x{entry['height']}): shape={frame.shape}, nonzero={nz}"
        )
        if nz:
            out = save_dir / f"sdk_cam_{entry['id']}.png"
            cv2.imwrite(str(out), frame)

    cam._release_resources()


def layer3_wrapper(side: str, camera_count: int, observe_sec: float, save_dir: Path) -> None:
    """Connect via UmiGripper, then watch _latest_frames and frame_count over time."""
    banner("Layer 3: UmiGripper wrapper")
    cfg = UmiGripperConfig(side=side, camera_count=camera_count, enable_tactile=False)
    robot = UmiGripper(cfg)

    print(f"  connecting (side={side}, camera_count={camera_count}) ...")
    try:
        robot.connect()
    except SystemExit as e:
        print(f"  connect aborted via sys.exit({e.code})")
        return
    except Exception as e:
        print(f"  connect raised: {e!r}")
        return

    try:
        # Snapshot of state right after connect
        camera = robot._camera
        thread_alive = robot._capture_thread is not None and robot._capture_thread.is_alive()
        print(
            f"  post-connect: cameras_in_sdk={len(camera.cameras)}, "
            f"camera.running={camera.running}, capture_thread_alive={thread_alive}, "
            f"robot._connected={robot._connected}"
        )

        # Poll for `observe_sec` seconds, sampling every 0.5s
        print(f"  polling for {observe_sec:.1f}s ...")
        t0 = time.time()
        while time.time() - t0 < observe_sec:
            time.sleep(0.5)
            with robot._data_lock:
                buffered_ids = sorted(
                    cid for cid, frame in robot._latest_frames.items() if frame is not None
                )
            counts = [(c["id"], c["frame_count"]) for c in camera.cameras]
            print(
                f"    t+{time.time() - t0:4.1f}s  "
                f"sdk_frame_counts={counts}  "
                f"wrapper_buffered_ids={buffered_ids}  "
                f"thread_alive={robot._capture_thread.is_alive()}  "
                f"camera.running={camera.running}"
            )

        # Final get_observation
        obs = robot.get_observation()
        print("  final get_observation():")
        for i in range(camera_count):
            f = obs.get(f"cam_{i}")
            if f is None:
                print(f"    cam_{i}: MISSING")
            else:
                nz = bool(f.any())
                print(f"    cam_{i}: shape={f.shape}, nonzero={nz}")
                if nz:
                    out = save_dir / f"wrapper_cam_{i}.png"
                    cv2.imwrite(str(out), f)
    finally:
        robot.disconnect()


def verdict(save_dir: Path) -> None:
    banner("Where did frames stop?")
    print(
        "  - If layer 1 has nonzero frames but layer 2 doesn't:\n"
        "        SDK init / device-mapping problem (resolution, MJPG, udev path order).\n"
        "  - If layer 2 has nonzero frames but layer 3 doesn't:\n"
        "        UmiGripper._capture_loop is the culprit.\n"
        "        Most likely: race in connect() — `_capture_thread.start()` runs\n"
        "        BEFORE `self._connected = True`, so the loop sees False, exits,\n"
        "        and the finally clause releases the cameras.\n"
        "        Fix: set `self._connected = True` BEFORE starting the thread.\n"
        "  - If layer 3 sdk_frame_counts stay at 0 even though layer 2 worked:\n"
        "        Same race, OR cap.read() is being starved (USB bandwidth /\n"
        "        resolution mismatch). Compare the resolutions printed in layer 2.\n"
        f"\n  Saved snapshots (if any) are under: {save_dir}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=("left", "right"), default="left")
    p.add_argument("--camera-count", type=int, default=3, help="Wrapper-side camera_count")
    p.add_argument("--width", type=int, default=640, help="Resolution width passed to SDK")
    p.add_argument("--height", type=int, default=480, help="Resolution height passed to SDK")
    p.add_argument("--observe-sec", type=float, default=4.0, help="How long layer 3 polls")
    p.add_argument("--skip-raw", action="store_true", help="Skip layer 1 (raw cv2)")
    p.add_argument("--skip-sdk", action="store_true", help="Skip layer 2 (SDK alone)")
    p.add_argument("--skip-wrapper", action="store_true", help="Skip layer 3 (UmiGripper)")
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/home/zzq/lerobot/examples/umi_gripper/diagnose_out"),
    )
    args = p.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    banner("Environment")
    print(f"  cv2 version: {cv2.__version__}")
    print(f"  /dev/video* nodes: {list_video_devices()}")
    udev = SIDE_DEFAULTS[args.side]["video_devices"]
    present = [d for d in udev if os.path.exists(d)]
    missing = [d for d in udev if not os.path.exists(d)]
    print(f"  configured side={args.side} udev paths: {len(udev)} total")
    print(f"    present: {present}")
    if missing:
        print(f"    MISSING: {missing}")

    if not args.skip_raw:
        layer1_raw_v4l2(present, args.save_dir)

    if not args.skip_sdk:
        layer2_sdk(args.side, udev, args.width, args.height, args.save_dir)

    if not args.skip_wrapper:
        layer3_wrapper(args.side, args.camera_count, args.observe_sec, args.save_dir)

    verdict(args.save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
