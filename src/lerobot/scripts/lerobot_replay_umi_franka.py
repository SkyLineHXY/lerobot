"""Replay a UMI ee6d trajectory from a LeRobot v3 dataset on a real Franka.

The script is aligned with the :class:`FrankaGenGripper` robot interface and is
primarily intended to validate whether an ee6d dataset (8-D state =
``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]``) can drive
the physical Franka + UMI gripper without policy inference in the loop.


Examples
--------
Replay episode 0 with arm-only motion (no gripper commands) — recommended for
first-time end-effector verification::

    python -m lerobot.scripts.lerobot_replay_umi_franka \\
        --dataset-root /home/zzq/gendas_dataset/lerobotv3_data/pick_and_place_20260428 \\
        --episode 0 \\
        --robot-ip 192.168.1.104 \\
        --camera-extrinsic-yaml /home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml

Replay with the gripper trajectory enabled::

    python -m lerobot.scripts.lerobot_replay_umi_franka \\
        --dataset-root /home/zzq/gendas_dataset/lerobotv3_data/pick_and_place_20260428 \\
        --episode 0 \\
        --use-gripper

Sanity-check the computed targets without connecting to hardware::

    python -m lerobot.scripts.lerobot_replay_umi_franka \\
        --dataset-root /home/zzq/gendas_dataset/lerobotv3_data/pick_and_place_20260428 \\
        --episode 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.robots.franka_gen_gripper import (
    UMI_REFERENCE_MODES,
    FrankaGenGripper,
    FrankaGenGripperConfig,
    load_flange_to_camera_extrinsic,
    pose_xyz_quat_to_se3,
    pose_xyz_rotvec_to_se3,
    se3_to_xyz_rotvec,
)
from lerobot.robots.franka_gen_gripper.umi_transforms import (
    absolute_position_world_orientation_to_base,
    ee_relative_action_to_base,
    umi_camera_t0_action_to_base,
)

logger = logging.getLogger("replay_umi_franka")


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------

_ALL_REFERENCE_MODES = list(UMI_REFERENCE_MODES) + ["ee_at_t_now"]


@dataclass
class EpisodeData:
    """Per-episode trajectory used by the replay loop."""

    actions: np.ndarray  # (T, 7 or 8): 7D pos+rotvec+gripper (world_flange) or 8D quat
    states: np.ndarray   # (T, 1 or 8)
    fps: float
    action_dim: int = 8   # 7 = world_flange, 8 = legacy quat


def _read_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    return json.loads(info_path.read_text())


def _find_parquet_for_episode(dataset_root: Path, episode_index: int) -> Path:
    """Locate the parquet file holding ``episode_index``.

    Both the v3 single-file layout (``data/chunk-000/file-000.parquet``) and
    sharded layouts are supported by simple linear scan.
    """
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset_root / 'data'}")

    for path in parquet_files:
        # Inexpensive metadata read to check episode coverage.
        eps = pq.read_table(path, columns=["episode_index"]).to_pandas()["episode_index"]
        if (eps == episode_index).any():
            return path
    raise ValueError(f"Episode {episode_index} not found in any parquet under {dataset_root}")


def load_episode(dataset_root: Path, episode_index: int) -> EpisodeData:
    info = _read_dataset_info(dataset_root)
    fps = float(info.get("fps", 30))

    parquet_path = _find_parquet_for_episode(dataset_root, episode_index)
    df = pq.read_table(parquet_path).to_pandas()
    ep_df = df[df["episode_index"] == episode_index].sort_values("frame_index")
    if ep_df.empty:
        raise ValueError(f"Episode {episode_index} is empty")

    actions = np.stack(ep_df["action"].to_numpy()).astype(np.float64)
    states = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float64)
    action_dim = actions.shape[1]
    if action_dim not in (7, 8):
        raise ValueError(
            f"期望 7D（world_flange: pos+rotvec+gripper）或 8D（legacy: pos+quat+gripper）action，"
            f"实际维度 {action_dim}。"
        )
    return EpisodeData(actions=actions, states=states, fps=fps, action_dim=action_dim)


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------

def _row_to_se3_and_gripper(row: np.ndarray) -> tuple[np.ndarray, float]:
    """从 7D（rotvec）或 8D（quat）行向量解析 SE(3) 与夹爪宽度。"""
    if len(row) == 7:
        # world_flange 格式：[pos3, rotvec3, gripper]
        pos, rotvec, gripper = row[0:3], row[3:6], float(row[6])
        return pose_xyz_rotvec_to_se3(pos, rotvec), gripper
    else:
        # legacy 格式：[pos3, quat4, gripper]
        pos, quat_xyzw, gripper = row[0:3], row[3:7], float(row[7])
        return pose_xyz_quat_to_se3(pos, quat_xyzw), gripper


def _transform_dry(
    delta_k: np.ndarray,
    T_BE_t0: np.ndarray,
    T_FC: np.ndarray,
    reference: str,
) -> np.ndarray:
    if reference == "camera_t0":
        return umi_camera_t0_action_to_base(delta_k, T_BE_t0, T_FC)
    if reference in ("ee_at_t0", "ee_at_t_now"):
        return ee_relative_action_to_base(delta_k, T_BE_t0)
    if reference == "abs_pos_world_rot":
        return absolute_position_world_orientation_to_base(delta_k, T_BE_t0)
    raise ValueError(f"Unknown reference {reference!r}")


def run_dry(args: argparse.Namespace) -> int:
    episode = load_episode(Path(args.dataset_root), args.episode)
    logger.info(
        f"[dry-run] episode {args.episode}: {len(episode.actions)} frames @ {episode.fps} Hz"
    )

    T_FC = load_flange_to_camera_extrinsic(args.camera_extrinsic_yaml)
    logger.info(f"[dry-run] ᶠT_C translation = {T_FC[:3, 3].round(4).tolist()}")

    T_BE_t0 = np.eye(4)  # identity placeholder (no FK available)

    # world_flange 格式（7D 绝对位姿）：以首帧为 t₀ 预计算 episode-relative delta
    is_world_flange = (episode.action_dim == 7)
    if is_world_flange:
        W_T_F_t0_dry, _ = _row_to_se3_and_gripper(episode.actions[0])
        W_T_F_t0_inv_dry = np.linalg.inv(W_T_F_t0_dry)
        logger.info("[dry-run] world_flange 格式：以首帧为 t₀ 计算 episode-relative delta。")

    indices = sorted({0, len(episode.actions) // 2, len(episode.actions) - 1})
    for k in indices:
        abs_se3_k, grip = _row_to_se3_and_gripper(episode.actions[k])
        delta_k = W_T_F_t0_inv_dry @ abs_se3_k if is_world_flange else abs_se3_k
        T_target = _transform_dry(delta_k, T_BE_t0, T_FC, args.reference)
        pose6d = se3_to_xyz_rotvec(T_target)
        print(
            f"frame {k:>4d}: pos={pose6d[:3].round(4).tolist()} "
            f"rotvec={pose6d[3:].round(4).tolist()} grip={grip:.4f}"
        )
    return 0


def run_live(args: argparse.Namespace) -> int:
    episode = load_episode(Path(args.dataset_root), args.episode)

    end = args.end_frame if args.end_frame is not None else len(episode.actions)
    actions = episode.actions[args.start_frame:end]
    if len(actions) == 0:
        logger.error("No frames to replay after applying --start-frame / --end-frame.")
        return 2

    fps = float(args.fps) if args.fps else episode.fps
    dt = 1.0 / fps
    logger.info(
        f"Replay episode={args.episode}  frames={len(actions)}  fps={fps:.2f}  "
        f"reference={args.reference}  use_gripper={args.use_gripper}"
    )

    config = FrankaGenGripperConfig(
        id="replay_umi",
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        gripper_side=args.gripper_side,
        gripper_serial_port=args.gripper_serial_port,
        gripper_camera_count=args.gripper_camera_count,
        gripper_enable_tactile=False,
        camera_extrinsic_yaml_path=str(args.camera_extrinsic_yaml),
    )

    robot = FrankaGenGripper(config)
    robot.connect()
    controller_started = False
    try:
        # Touch the extrinsic up-front so a missing/invalid YAML fails fast,
        # before any motion is requested.
        _ = robot.umi_camera_extrinsic

        # Capture FK snapshot at t₀.
        time.sleep(0.3)
        T_BE_t0 = robot.get_ee_se3()
        logger.info(
            f"ᴮT_E(t₀): translation={T_BE_t0[:3, 3].round(4).tolist()}  "
            f"(this becomes the replay reference frame)"
        )

        if not args.yes:
            logger.warning("About to start motion. Make sure the workspace is clear.")
            try:
                input(">>> Press ENTER to begin replay (Ctrl-C to abort) ...")
            except (EOFError, KeyboardInterrupt):
                logger.info("Aborted before motion.")
                return 130

        # Start a Cartesian impedance controller on the NUC. Without an active
        # controller, polymetis raises:
        #   "Tried to perform a controller update with no controller running."
        Kx = np.array(args.Kx, dtype=float) if args.Kx else None
        Kxd = np.array(args.Kxd, dtype=float) if args.Kxd else None
        robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
        controller_started = True

        # world_flange 格式（7D 绝对位姿）：以首帧为 t₀ 预计算 episode-relative delta
        is_world_flange = (episode.action_dim == 7)
        W_T_F_t0_inv = None
        if is_world_flange:
            W_T_F_t0_action, _ = _row_to_se3_and_gripper(actions[0])
            W_T_F_t0_inv = np.linalg.inv(W_T_F_t0_action)
            if args.reference == "ee_at_t_now":
                logger.warning(
                    "world_flange 绝对位姿格式回放时 ee_at_t_now 无效（无 VIO 状态），"
                    "已自动退化为 ee_at_t0（固定 episode 起点）。"
                )

        # 仅对非 world_flange 的旧格式支持 ee_at_t_now（每帧刷新机器人 FK）
        per_step_t0 = (not is_world_flange) and (args.reference == "ee_at_t_now")
        if per_step_t0:
            logger.info("reference=ee_at_t_now：每帧实时刷新 T_BE_t₀。")

        # ------------------------------------------------------------------
        # Replay loop — fixed-rate scheduling without long-term drift.
        # ------------------------------------------------------------------
        next_t = time.perf_counter()
        late_frames = 0
        for k, row in enumerate(actions):
            # 每帧解析位姿与夹爪宽度
            abs_se3_k, grip = _row_to_se3_and_gripper(row)

            if is_world_flange:
                # 绝对 W_T_F(k) → episode-relative delta = inv(W_T_F(t₀)) @ W_T_F(k)
                delta_k = W_T_F_t0_inv @ abs_se3_k
                current_T_BE_t0 = T_BE_t0   # 固定 episode 起点
                effective_reference = "ee_at_t0"
            else:
                delta_k = abs_se3_k
                if per_step_t0:
                    current_T_BE_t0 = robot.get_ee_se3()
                else:
                    current_T_BE_t0 = T_BE_t0
                effective_reference = "ee_at_t0" if per_step_t0 else args.reference

            grip_cmd = grip if args.use_gripper else None

            try:
                robot.send_umi_action(
                    delta_k=delta_k,
                    T_BE_t0=current_T_BE_t0,
                    reference=effective_reference,
                    gripper_width=grip_cmd,
                )
            except Exception as e:
                logger.exception(f"send_umi_action failed at frame {k}: {e}")
                return 1

            next_t += dt
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                late_frames += 1
                # Reset schedule to avoid runaway lag when behind.
                next_t = time.perf_counter()

            if args.verbose and k % max(1, int(fps)) == 0:
                logger.debug(f"  frame {k}/{len(actions)}")

        if late_frames:
            logger.warning(
                f"{late_frames}/{len(actions)} frames missed the target period; "
                "consider lowering --fps or reducing serial-camera load."
            )
        logger.info("Replay finished cleanly.")
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted — stopping motion and disconnecting.")
        return 130
    finally:
        if controller_started:
            try:
                robot.terminate_policy()
            except Exception as e:
                logger.warning(f"terminate_policy failed: {e}")
        try:
            robot.disconnect()
        except Exception:
            pass


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay a UMI ee6d trajectory on a Franka + UMI gripper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument(
        "--dataset-root", type=Path, required=True,
        help="LeRobot v3 dataset root (must contain meta/info.json and data/).",
    )
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    p.add_argument("--start-frame", type=int, default=0, help="First frame to play (inclusive).")
    p.add_argument("--end-frame", type=int, default=None, help="Last frame to play (exclusive).")
    p.add_argument("--fps", type=float, default=20, help="Override dataset FPS.")

    # Robot connection
    p.add_argument("--robot-ip", default="192.168.110.17")
    p.add_argument("--robot-port", type=int, default=4242)
    p.add_argument("--gripper-side", default="left", choices=["left", "right"])
    p.add_argument("--gripper-serial-port", default=None,
                   help="Override auto-detected gripper serial port.")
    p.add_argument("--gripper-camera-count", type=int, default=0,
                   help="Number of UMI gripper cameras to open (0 = none, replay-only).")

    # UMI ee6d transform
    p.add_argument(
        "--camera-extrinsic-yaml",
        default="/home/zzq/franka_ws/src/franka_easy_handeye/cfg/camera_transform.yaml",
        help="Hand-eye calibration YAML (parent=panda_link8, child=camera).",
    )
    p.add_argument(
        "--reference", default="ee_at_t_now", choices=_ALL_REFERENCE_MODES,
        help=(
            "ee6d 坐标系约定。"
            "'ee_at_t_now'（推荐，world_flange 格式）每帧实时读取 FK 作为 t₀；"
            "'ee_at_t0' 固定 episode 起点；"
            "'camera_t0' 使用相机参考（legacy）。"
        ),
    )

    # Gripper trajectory
    p.add_argument(
        "--use-gripper", action="store_true",
        help="Send dataset gripper widths along with arm motion (default off).",
    )

    # Cartesian impedance gains (defaults match Franka.DEFAULT_CARTESIAN_KX/KXD)
    p.add_argument(
        "--Kx", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Override Cartesian impedance stiffness (6 floats).",
    )
    p.add_argument(
        "--Kxd", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Override Cartesian impedance damping (6 floats).",
    )

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Compute targets and print samples; do not connect to robot.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the pre-motion confirmation prompt.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.dry_run:
        return run_dry(args)
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
