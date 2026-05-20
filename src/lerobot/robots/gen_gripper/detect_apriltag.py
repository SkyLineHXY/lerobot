#!/usr/bin/env python3
"""
Real-time AprilTag detection for the UMI gripper cam0 (640x480).

Requirements:
    pip install pupil-apriltags

Usage:
    python detect_apriltag.py [options]

Examples:
    # Auto-detect camera device, tag36h11 family, no pose
    python detect_apriltag.py

    # Specify device explicitly
    python detect_apriltag.py --device /dev/left_video_0_main

    # Enable pose estimation (tag physical side length in meters)
    python detect_apriltag.py --tag-size 0.05

    # Undistort frames before detection for higher accuracy
    python detect_apriltag.py --tag-size 0.05 --undistort
"""

import argparse
import os
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Camera intrinsics for cam0  (640 x 480)
# ---------------------------------------------------------------------------
CAM_K = np.array(
    [
        [247.57278109441967, 0.0, 323.485709196671],
        [0.0, 229.30196629116588, 240.67747771696563],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# k1, k2, p1, p2  (radial + tangential, OpenCV convention)
CAM_D = np.array(
    [
        -0.023039507514895814,
        0.0031873965359944527,
        -0.002337065272196324,
        0.00010825496015046585,
    ],
    dtype=np.float64,
)

FX = CAM_K[0, 0]
FY = CAM_K[1, 1]
CX = CAM_K[0, 2]
CY = CAM_K[1, 2]

# ---------------------------------------------------------------------------
# Default camera device candidates (tried in order)
# ---------------------------------------------------------------------------
_DEFAULT_DEVICES = [
    "/dev/left_video_0_main",
    "/dev/right_video_0_main",
    "/dev/video0",
]

FRAME_W, FRAME_H = 640, 480


def open_camera(device: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(5):  # flush stale frames from driver buffer
        cap.grab()
    return cap


def build_undistort_maps(width: int, height: int):
    new_K, _ = cv2.getOptimalNewCameraMatrix(CAM_K, CAM_D, (width, height), alpha=1)
    map1, map2 = cv2.initUndistortRectifyMap(
        CAM_K, CAM_D, None, new_K, (width, height), cv2.CV_16SC2
    )
    return map1, map2, new_K


def draw_tag(
    img: np.ndarray,
    det,
    tag_size: float | None,
    draw_pose: bool,
    cam_k: np.ndarray,
) -> None:
    corners = det.corners.astype(int)

    # Filled semi-transparent overlay on the tag area
    overlay = img.copy()
    cv2.fillPoly(overlay, [corners.reshape(-1, 1, 2)], (0, 255, 0))
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Tag border
    cv2.polylines(img, [corners.reshape(-1, 1, 2)], isClosed=True, color=(0, 230, 0), thickness=2)

    # Corner circles + indices
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 60, 255), -1)
        cv2.putText(
            img, str(i), (int(cx) + 7, int(cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 255), 1, cv2.LINE_AA,
        )

    # Center dot
    cx, cy = det.center.astype(int)
    cv2.circle(img, (cx, cy), 4, (255, 60, 0), -1)

    # Tag-ID label with black background
    label = f"id={det.tag_id}"
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx, ty = cx - tw // 2, cy - 12
    cv2.rectangle(img, (tx - 2, ty - th - bl), (tx + tw + 2, ty + bl), (0, 0, 0), -1)
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 255), 2, cv2.LINE_AA)

    # Pose axes (X=red, Y=green, Z=blue) projected back onto the image
    if draw_pose and tag_size is not None and det.pose_R is not None and det.pose_t is not None:
        half = tag_size / 2.0
        axes_3d = np.float32([[0, 0, 0], [half, 0, 0], [0, half, 0], [0, 0, -half]])
        rvec, _ = cv2.Rodrigues(det.pose_R)
        tvec = det.pose_t.reshape(3, 1)
        pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam_k, np.zeros(4))
        pts = pts.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        cv2.arrowedLine(img, origin, tuple(pts[1]), (0, 0, 220), 2, tipLength=0.25)  # X red
        cv2.arrowedLine(img, origin, tuple(pts[2]), (0, 220, 0), 2, tipLength=0.25)  # Y green
        cv2.arrowedLine(img, origin, tuple(pts[3]), (220, 0, 0), 2, tipLength=0.25)  # Z blue

        dist_cm = float(np.linalg.norm(det.pose_t)) * 100.0
        dist_label = f"{dist_cm:.1f} cm"
        cv2.putText(
            img, dist_label, (cx - 25, cy + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-time AprilTag detection on UMI gripper cam0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default=None, help="Video device path (default: auto-detect)")
    parser.add_argument(
        "--tag-family",
        default="tag36h11",
        choices=["tag36h11", "tag25h9", "tag16h5", "tagStandard41h12", "tagStandard52h13"],
        help="AprilTag family (default: tag36h11)",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=None,
        metavar="METERS",
        help="Physical tag side length in metres — enables pose estimation",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable pose axis overlay even when --tag-size is given",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort frames before detection (slower, higher accuracy)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of detector threads (default: 2)",
    )
    args = parser.parse_args()

    try:
        from pupil_apriltags import Detector
    except ImportError:
        print("ERROR: pupil-apriltags is not installed.")
        print("       Run:  pip install pupil-apriltags")
        return 1

    # --- select camera device ---
    device = args.device
    if device is None:
        for d in _DEFAULT_DEVICES:
            if os.path.exists(d):
                device = d
                break
        else:
            device = "/dev/video0"
    print(f"[cam] opening {device}")

    try:
        cap = open_camera(device)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] resolution {width}x{height}")

    # --- optional undistort pre-processing ---
    undistort_maps = None
    effective_K = CAM_K
    if args.undistort:
        map1, map2, new_K = build_undistort_maps(width, height)
        undistort_maps = (map1, map2)
        effective_K = new_K
        print("[cam] undistort maps ready")

    # --- build detector ---
    estimate_pose = (args.tag_size is not None) and (not args.no_pose)
    detector = Detector(
        families=args.tag_family,
        nthreads=args.threads,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )
    fx, fy = effective_K[0, 0], effective_K[1, 1]
    cx_k, cy_k = effective_K[0, 2], effective_K[1, 2]
    print(
        f"[det] family={args.tag_family}  pose={'yes (tag_size=%.4fm)' % args.tag_size if estimate_pose else 'no'}"
    )
    print("[win] press  Q / ESC  to quit")

    win_name = "AprilTag Detection — cam0"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, width, height)

    fps_buf: list[float] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            if undistort_maps is not None:
                frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(
                gray,
                estimate_tag_pose=estimate_pose,
                camera_params=(fx, fy, cx_k, cy_k) if estimate_pose else None,
                tag_size=args.tag_size if estimate_pose else None,
            )

            for det in detections:
                draw_tag(frame, det, args.tag_size, estimate_pose, effective_K)

            # --- HUD ---
            now = time.monotonic()
            fps_buf.append(now)
            if len(fps_buf) > 60:
                fps_buf = fps_buf[-60:]
            fps = (len(fps_buf) - 1) / (fps_buf[-1] - fps_buf[0]) if len(fps_buf) >= 2 else 0.0

            hud1 = f"FPS: {fps:.1f}   Tags: {len(detections)}"
            hud2 = f"{args.tag_family}" + (f"   tag_size={args.tag_size:.3f}m" if estimate_pose else "")
            cv2.putText(frame, hud1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, hud2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
