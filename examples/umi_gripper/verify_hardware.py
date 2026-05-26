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

"""Real-hardware smoke test for the Gen Controller UMI gripper.

Run this on a machine with the gripper attached to verify the SDK plumbing:
encoder streaming, motor command, camera frame capture, and (optionally)
tactile sensors.

Usage:
    python examples/umi_gripper/verify_hardware.py --side left
    python examples/umi_gripper/verify_hardware.py --side right --enable-tactile --cycles 3
    python examples/umi_gripper/verify_hardware.py --side left --no-cameras --no-motion

The script never asserts — it prints a step-by-step report so you can eyeball
what works and what doesn't on this particular bench setup.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from lerobot.robots.gen_gripper import UmiGripper, UmiGripperConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", choices=("left", "right"), default="left", help="Gripper side")
    p.add_argument("--serial-port", default=None, help="Override serial port (default: side udev path)")
    p.add_argument("--camera-count", type=int, default=3, help="Number of cameras to open (0 to skip)")
    p.add_argument("--no-cameras", action="store_true", help="Shortcut for --camera-count 0")
    p.add_argument("--enable-tactile", action="store_true", help="Enable tactile sensor stream")
    p.add_argument("--no-motion", action="store_true", help="Skip the open/close cycles")
    p.add_argument("--cycles", type=int, default=2, help="Number of open/close cycles")
    p.add_argument(
        "--encoder-warmup-sec",
        type=float,
        default=2.0,
        help="How long to wait for the encoder stream to settle before reporting",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/home/zzq/lerobot/examples/umi_gripper/umi_gripper_verify"),
        help="Where to save one snapshot per camera",
    )
    p.add_argument("--calibrate", action="store_true", help="Send encoder calibration after connect")
    p.add_argument(
        "--stream-sec",
        type=float,
        default=10.0,
        help="Duration of the get_observation() throughput test (0 to skip)",
    )
    p.add_argument(
        "--stream-fps",
        type=float,
        default=30,
        help="Cap the streaming poll rate (0 = poll as fast as possible)",
    )
    p.add_argument(
        "--show-preview",
        default=True,
        action="store_true",
        help="During streaming, display each camera in a cv2 window (ESC to stop)",
    )
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def wait_for_encoder(robot: UmiGripper, timeout: float) -> bool:
    """Block until the encoder callback fires at least once, or timeout."""
    deadline = time.time() + timeout
    initial = robot._latest_encoder
    while time.time() < deadline:
        if robot._latest_encoder != initial or robot._latest_encoder != 0.0:
            return True
        time.sleep(0.05)
    return False


def wait_for_frames(robot: UmiGripper, expected: int, timeout: float = 5.0) -> int:
    """Wait until at least one frame is buffered for each expected camera."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with robot._data_lock:
            ready = sum(1 for v in robot._latest_frames.values() if v is not None)
        if ready >= expected:
            return ready
        time.sleep(0.1)
    with robot._data_lock:
        return sum(1 for v in robot._latest_frames.values() if v is not None)


def report_observation(obs: dict, enable_tactile: bool, camera_count: int) -> None:
    print(f"  gripper.pos = {obs['gripper.pos']:.4f} m")
    for i in range(camera_count):
        key = f"cam_{i}"
        frame = obs.get(key)
        if frame is None:
            print(f"  {key}: MISSING")
        else:
            nonzero = bool(frame.any())
            print(f"  {key}: shape={frame.shape}, dtype={frame.dtype}, nonzero={nonzero}")
    if enable_tactile:
        for side in ("tactile_left", "tactile_right"):
            v = obs.get(side)
            if v is None:
                print(f"  {side}: MISSING")
            else:
                print(
                    f"  {side}: len={len(v)}, min={int(v.min())}, max={int(v.max())}, mean={float(v.mean()):.1f}"
                )


def save_snapshots(robot: UmiGripper, camera_count: int, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    obs = robot.get_observation()
    saved = []
    for i in range(camera_count):
        frame = obs.get(f"cam_{i}")
        if frame is None or not frame.any():
            print(f"  cam_{i}: skipped (no data)")
            continue
        out = save_dir / f"{robot.config.side}_cam_{i}.png"
        cv2.imwrite(str(out), frame)
        saved.append(out)
    if saved:
        print(f"  saved {len(saved)} snapshot(s) to {save_dir}")
    else:
        print("  no snapshots saved")


def _frame_fingerprint(frame) -> bytes:
    """Cheap fingerprint to detect whether two frames have the same pixel content.

    A real-camera frame almost certainly differs in its first 256 bytes between
    consecutive captures, so this is a reliable "is this a new frame?" check
    without paying the cost of hashing the entire image at high frame rates.
    """
    return bytes(frame.flat[:256])


def stream_observations(
    robot: UmiGripper,
    duration_sec: float,
    target_fps: float,
    show_preview: bool,
) -> None:
    """Drive get_observation() in a tight loop, measuring real throughput.

    Reports three numbers per second of run:
      - poll rate          — how many get_observation() calls succeeded
      - encoder update rate — how often gripper.pos changed between polls
      - per-camera fresh-frame rate — how often each cam_i array's pixel
                                       contents changed between polls
    plus a final get_observation() latency distribution (mean/p50/p95/max).
    """
    camera_count = robot.config.camera_count
    interval = 1.0 / target_fps if target_fps > 0 else 0.0

    last_encoder = None
    encoder_updates = 0
    cam_fingerprints: dict[int, bytes] = {}
    cam_updates: dict[int, int] = {i: 0 for i in range(camera_count)}
    latencies_ms: list[float] = []
    poll_count = 0

    t_start = time.time()
    last_report = t_start

    if show_preview:
        for i in range(camera_count):
            cv2.namedWindow(f"cam_{i}", cv2.WINDOW_NORMAL)

    try:
        while time.time() - t_start < duration_sec:
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            latencies_ms.append((time.perf_counter() - loop_start) * 1000.0)
            poll_count += 1

            # Encoder freshness
            encoder = obs["gripper.pos"]
            if last_encoder is None or encoder != last_encoder:
                encoder_updates += 1
                last_encoder = encoder

            # Camera freshness
            for i in range(camera_count):
                frame = obs.get(f"cam_{i}")
                if frame is None:
                    continue
                fp = _frame_fingerprint(frame)
                if cam_fingerprints.get(i) != fp:
                    cam_updates[i] += 1
                    cam_fingerprints[i] = fp
                if show_preview:
                    cv2.imshow(f"cam_{i}", frame)

            if show_preview and cv2.waitKey(1) == 27:  # ESC
                print("  preview interrupted (ESC)")
                break

            # 1Hz progress line
            now = time.time()
            if now - last_report >= 1.0:
                elapsed = now - t_start
                cam_rates = " ".join(
                    f"cam_{i}={cam_updates[i] / elapsed:5.1f}" for i in range(camera_count)
                )
                print(
                    f"  t+{elapsed:5.1f}s  "
                    f"poll={poll_count / elapsed:6.1f} Hz  "
                    f"encoder_new={encoder_updates / elapsed:5.1f} Hz  "
                    f"{cam_rates}"
                )
                last_report = now

            # Optional rate cap
            if interval > 0:
                sleep = interval - (time.perf_counter() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        if show_preview:
            cv2.destroyAllWindows()

    elapsed = max(time.time() - t_start, 1e-6)
    print("\n  --- summary ---")
    print(
        f"  get_observation(): {poll_count} calls in {elapsed:.2f}s -> "
        f"{poll_count / elapsed:.1f} Hz"
    )
    pct = 100.0 * encoder_updates / poll_count if poll_count else 0.0
    print(
        f"  encoder fresh   : {encoder_updates} updates -> "
        f"{encoder_updates / elapsed:.1f} Hz ({pct:.0f}% of polls)"
    )
    for i in range(camera_count):
        u = cam_updates[i]
        pct = 100.0 * u / poll_count if poll_count else 0.0
        print(
            f"  cam_{i} fresh    : {u} new frames -> "
            f"{u / elapsed:.1f} Hz ({pct:.0f}% of polls)"
        )
    if latencies_ms:
        s = sorted(latencies_ms)
        n = len(s)
        mean = sum(s) / n
        p50 = s[n // 2]
        p95 = s[min(n - 1, int(n * 0.95))]
        print(
            f"  get_observation() latency: mean={mean:.2f} ms  "
            f"p50={p50:.2f} ms  p95={p95:.2f} ms  max={s[-1]:.2f} ms"
        )


def run_motion_cycle(robot: UmiGripper, cfg: UmiGripperConfig, cycles: int) -> None:
    targets = [cfg.gripper_max_distance, cfg.gripper_min_distance]
    for c in range(cycles):
        for tgt in targets:
            sent = robot.send_action({"gripper.pos": tgt})["gripper.pos"]
            # Give the actuator ~1s to move, polling encoder
            t_end = time.time() + 1.0
            last = None
            while time.time() < t_end:
                with robot._data_lock:
                    last = robot._latest_encoder
                time.sleep(0.05)
            err = abs((last or 0.0) - sent)
            print(f"  cycle {c + 1}: target={sent:.4f} m, encoder={last:.4f} m, |err|={err:.4f}")


def main() -> int:
    args = parse_args()
    if args.no_cameras:
        args.camera_count = 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = UmiGripperConfig(
        side=args.side,
        serial_port=args.serial_port,
        camera_count=args.camera_count,
        enable_tactile=args.enable_tactile,
    )
    print(f"Configuration: side={cfg.side}, serial_port={cfg.serial_port}, "
          f"camera_count={cfg.camera_count}, enable_tactile={cfg.enable_tactile}")

    robot = UmiGripper(cfg)

    section("Connect")
    try:
        robot.connect()
    except Exception as e:
        print(f"FAILED to connect: {e}")
        return 1
    print("connected")

    try:
        if args.calibrate:
            section("Calibrate")
            robot.calibrate()
            time.sleep(1.0)

        section("Encoder warmup")
        ok = wait_for_encoder(robot, args.encoder_warmup_sec)
        print(f"  encoder reading received: {ok} (latest={robot._latest_encoder:.4f} m)")

        if args.camera_count > 0:
            section("Camera warmup")
            ready = wait_for_frames(robot, args.camera_count, timeout=5.0)
            print(f"  cameras with frames: {ready}/{args.camera_count}")

        if args.enable_tactile:
            section("Tactile warmup")
            t_end = time.time() + 2.0
            while time.time() < t_end and robot._latest_tactile_left is None:
                time.sleep(0.05)
            got = robot._latest_tactile_left is not None
            print(f"  tactile reading received: {got}")

        section("First observation")
        obs = robot.get_observation()
        report_observation(obs, args.enable_tactile, args.camera_count)

        if args.camera_count > 0:
            section("Save snapshots")
            save_snapshots(robot, args.camera_count, args.save_dir)

        if not args.no_motion:
            section(f"Motion cycles ({args.cycles})")
            run_motion_cycle(robot, cfg, args.cycles)
            # Leave gripper closed at the end
            robot.send_action({"gripper.pos": cfg.gripper_min_distance})
            time.sleep(0.5)

        if args.stream_sec > 0:
            cap = "uncapped" if args.stream_fps <= 0 else f"capped at {args.stream_fps:g} Hz"
            preview = "with preview" if args.show_preview else "no preview"
            section(f"Streaming get_observation() for {args.stream_sec:.1f}s ({cap}, {preview})")
            stream_observations(robot, args.stream_sec, args.stream_fps, args.show_preview)

        section("Final observation")
        obs = robot.get_observation()
        report_observation(obs, args.enable_tactile, args.camera_count)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR during verification: {e}")
        return 1
    finally:
        section("Disconnect")
        try:
            robot.disconnect()
            print("disconnected cleanly")
        except Exception as e:
            print(f"disconnect error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
