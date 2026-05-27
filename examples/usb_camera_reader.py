"""读取 USB 摄像头数据的示例脚本。

支持：
- 实时预览画面
- 保存截图（按 's' 键）
- 显示帧率 (FPS)
- 指定摄像头设备编号
"""

import argparse
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="读取 USB 摄像头数据")
    parser.add_argument("--device", type=int, default=2, help="摄像头设备编号（默认 0）")
    parser.add_argument("--width", type=int, default=640, help="分辨率宽度（默认 640）")
    parser.add_argument("--height", type=int, default=480, help="分辨率高度（默认 480）")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率（默认 30）")
    parser.add_argument("--save-dir", type=str, default=".", help="截图保存目录（默认当前目录）")
    return parser.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头设备 {args.device}，请检查设备连接或编号")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开：设备={args.device}，分辨率={actual_w}x{actual_h}")
    print("按 's' 保存截图，按 'q' 或 ESC 退出")

    screenshot_idx = 0
    fps_counter = 0
    fps_display = 0.0
    t_fps = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败，摄像头可能已断开")
                break

            # 计算并显示 FPS
            fps_counter += 1
            elapsed = time.time() - t_fps
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                t_fps = time.time()

            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("USB Camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' 或 ESC
                break
            elif key == ord("s"):
                path = f"{args.save_dir}/screenshot_{screenshot_idx:04d}.png"
                cv2.imwrite(path, frame)
                print(f"截图已保存：{path}")
                screenshot_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头已关闭")


if __name__ == "__main__":
    main()
