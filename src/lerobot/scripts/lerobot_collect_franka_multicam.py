"""Interactive FrankaMultiCam dataset collection.

This script uses the same FrankaMultiCam robot interface and LeRobot v3 dataset
schema as ``lerobot_record_franka_multicam.py``, but exposes a manual collection
workflow:
    c      start / resume collection
    space  pause collection
    s      save the current episode buffer
    r      clear the current episode buffer
    q      finalize and exit
Usage:

    python -m lerobot.scripts.lerobot_collect_franka_multicam \
        --config_path configs/collect_franka_multicam.yaml
"""

import logging
import queue
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.franka.franka import EE_POSE_KEYS
from lerobot.robots.franka_gen_gripper.umi_transforms import (
    pose_xyz_rotvec_to_se3,
    se3_to_xyz_rotvec,
)
from lerobot.robots.franka_multicam.config_franka_multicam import FrankaMultiCamConfig  # noqa: F401
from lerobot.cameras.hik_camera.configuration_hik import HikCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.opencv_zmq.configuration_opencv_zmq import OpenCVZmqCameraConfig  # noqa: F401
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_record import DatasetRecordConfig, make_default_processors
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.control_utils import sanity_check_dataset_name
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class InteractiveCollectionConfig:
    """Interactive collection behavior."""

    max_frames_per_episode: int | None = None
    """Pause collection automatically once this many frames are buffered."""

    min_frames_before_save: int = 1
    """Minimum buffered frames required before accepting ``s`` save."""

    print_every_n_frames: int = 30
    """Log progress every N collected frames. Set to 0 to disable."""


@dataclass
class CollectFrankaMultiCamConfig:
    """FrankaMultiCam interactive data collection config."""

    robot: FrankaMultiCamConfig
    dataset: DatasetRecordConfig
    collection: InteractiveCollectionConfig = field(default_factory=InteractiveCollectionConfig)
    teleop: TeleoperatorConfig | None = None
    """可选遥操作器配置（如 SpaceMouseTeleopConfig）。
    当不为 None 时，由遥操作器闭环驱动机械臂并写入动作；
    为 None 时保持原有"obs→action 复制"行为（机械臂由外部驱动）。"""
    gripper_width_min_m: float = 0.0
    """夹爪全合时对应的宽度（米），用于将 gripper 信号映射到 gripper.width。"""
    gripper_width_max_m: float = 0.08
    """夹爪全开时对应的宽度（米）。"""
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    play_sounds: bool = True
    resume: bool = False


def _get_effective_root(cfg_dataset: DatasetRecordConfig) -> Path:
    """返回数据集的实际存储路径（与 LeRobotDataset 逻辑保持一致）。"""
    if cfg_dataset.root is not None:
        return Path(cfg_dataset.root)
    return HF_LEROBOT_HOME / cfg_dataset.repo_id


def _dataset_exists(root: Path) -> bool:
    """判断指定路径下是否已存在合法的 LeRobot 数据集（含 meta/info.json）。"""
    return (root / "meta" / "info.json").exists()


def _prompt_dataset_conflict(root: Path, repo_id: str) -> str:
    """检测到已存在数据集时，交互式询问用户的处理策略。

    Returns:
        ``"overwrite"``：删除并重新创建数据集。
        ``"resume"``：在已有数据集基础上继续扩充样本。
    """
    sep = "═" * 50
    logger.info(
        "\n%s\n"
        "  检测到已存在的数据集：\n"
        "    repo_id : %s\n"
        "    路径    : %s\n"
        "%s\n"
        "  请选择处理方式：\n"
        "    [1] 覆盖原数据集（删除并重新创建）\n"
        "    [2] 在原有数据集基础上继续扩充样本（resume）\n"
        "    [q] 退出程序\n"
        "%s",
        sep,
        repo_id,
        root,
        sep,
        sep,
    )
    while True:
        try:
            choice = input("请输入选项 [1/2/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n操作已取消。")
        if choice == "1":
            logger.warning("将删除数据集目录：%s", root)
            confirm = input("确认删除？此操作不可恢复 [yes/N]: ").strip().lower()
            if confirm == "yes":
                return "overwrite"
            logger.info("已取消删除，请重新选择。")
        elif choice == "2":
            return "resume"
        elif choice == "q":
            raise SystemExit("用户取消操作。")
        else:
            logger.warning("无效输入 '%s'，请输入 1、2 或 q。", choice)


def _episode_buffer_size(dataset: LeRobotDataset | None) -> int:
    if dataset is None or dataset.episode_buffer is None:
        return 0
    return int(dataset.episode_buffer.get("size", 0))


def _init_interactive_keyboard() -> tuple[Any, queue.Queue[str]]:
    """Start a pynput keyboard listener and return its event queue."""
    try:
        from pynput import keyboard
    except Exception as e:
        raise RuntimeError("Interactive collection requires pynput keyboard support.") from e

    key_queue: queue.Queue[str] = queue.Queue()

    def on_press(key: Any) -> None:
        try:
            if key == keyboard.Key.space:
                key_queue.put("space")
            elif hasattr(key, "char") and key.char in {"c", "s", "r", "q"}:
                key_queue.put(key.char)
        except Exception as e:  # pragma: no cover - defensive listener guard
            logger.warning("Keyboard event handling failed: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    try:
        listener.start()
    except Exception as e:
        raise RuntimeError("Failed to start keyboard listener for interactive collection.") from e

    return listener, key_queue


def _drain_keys(key_queue: queue.Queue[str]) -> list[str]:
    keys: list[str] = []
    while True:
        try:
            keys.append(key_queue.get_nowait())
        except queue.Empty:
            return keys


def _print_usage() -> None:
    logger.info(
        "\n"
        "═══════════════════════════════════════════\n"
        "  FrankaMultiCam Interactive Collection\n"
        "═══════════════════════════════════════════\n"
        "  c      → Start / resume collecting\n"
        "  space  → Pause\n"
        "  s      → Save current episode\n"
        "  r      → Clear current buffer\n"
        "  q      → Finalize dataset & exit\n"
        "═══════════════════════════════════════════"
    )


def _go_home_and_restart_cartesian(robot: Any) -> None:
    """终止当前 Cartesian 控制器 → go_home → 重启 Cartesian 阻抗控制器。

    在 spacemouse 采集模式下，每次 episode 开始前调用，确保机械臂从固定 home 位出发。
    """
    # 先终止当前 Cartesian 阻抗控制器（容错：首次启动时可能尚未启动）
    try:
        if hasattr(robot, "terminate_policy"):
            robot.terminate_policy()
        elif hasattr(robot, "_franka") and hasattr(robot._franka, "terminate_policy"):
            robot._franka.terminate_policy()
    except Exception as exc:
        logger.debug("terminate_policy 跳过（控制器可能尚未启动）: %s", exc)

    # 回到 home 位
    logger.info("机械臂正在回到 home 位，请勿干扰...")
    try:
        if hasattr(robot, "reset"):
            robot.reset()
        elif hasattr(robot, "_franka") and hasattr(robot._franka, "reset"):
            robot._franka.reset()
        else:
            logger.warning("未找到 reset() 方法，跳过 go_home。")
            return
    except Exception as exc:
        logger.warning("go_home 失败（可忽略）: %s", exc)
        return

    logger.info("机械臂已到达 home 位，正在重启 Cartesian 阻抗控制器...")
    # 重启 Cartesian 阻抗控制器
    if hasattr(robot, "start_cartesian_impedance"):
        robot.start_cartesian_impedance()
    elif hasattr(robot, "_franka") and hasattr(robot._franka, "start_cartesian_impedance"):
        robot._franka.start_cartesian_impedance()
    logger.info("Cartesian 阻抗控制器已就绪，可以开始采集。")


def _make_dataset(
    cfg: CollectFrankaMultiCamConfig,
    robot: Any,
    dataset_features: dict[str, Any],
    overwrite: bool = False,
) -> LeRobotDataset:
    num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            encoder_threads=cfg.dataset.encoder_threads,
        )
        if num_cameras > 0 and not cfg.dataset.streaming_encoding:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            )
        return dataset

    if overwrite:
        root = _get_effective_root(cfg.dataset)
        if root.exists():
            logger.info("正在删除已有数据集目录：%s", root)
            shutil.rmtree(root)

    sanity_check_dataset_name(cfg.dataset.repo_id, None)
    return LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=(
            cfg.dataset.num_image_writer_threads_per_camera * num_cameras if num_cameras > 0 else 0
        ),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        vcodec=cfg.dataset.vcodec,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        encoder_threads=cfg.dataset.encoder_threads,
    )


@parser.wrap()
def collect(cfg: CollectFrankaMultiCamConfig) -> LeRobotDataset:
    init_logging()
    logger.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="franka_multicam_collection", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)

    # 可选遥操作器（spacemouse 等）
    teleop_device = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    _, robot_action_proc, robot_obs_proc = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_proc,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_obs_proc,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    logger.info("dataset features:\n%s", pformat(dataset_features))

    dataset: LeRobotDataset | None = None
    listener = None
    collecting = False
    stop_requested = False
    saved_episodes = 0
    overwrite = False

    # 在非 resume 模式下，若目标路径已存在合法数据集，则询问用户处理策略
    if not cfg.resume:
        effective_root = _get_effective_root(cfg.dataset)
        if _dataset_exists(effective_root):
            conflict_choice = _prompt_dataset_conflict(effective_root, cfg.dataset.repo_id)
            if conflict_choice == "resume":
                cfg.resume = True
                logger.info("已切换为 resume 模式，将在已有数据集基础上继续扩充样本。")
            else:
                overwrite = True
                logger.info("已选择覆盖模式，将删除并重新创建数据集。")

    try:
        dataset = _make_dataset(cfg, robot, dataset_features, overwrite=overwrite)

        robot.connect()

        if teleop_device is not None:
            teleop_device.connect()
            # SpaceMouse 采集模式：先 go_home 再启动 Cartesian 阻抗控制器
            _go_home_and_restart_cartesian(robot)

        listener, key_queue = _init_interactive_keyboard()
        _print_usage()

        fps = cfg.dataset.fps
        frame_period_s = 1.0 / fps
        max_frames = cfg.collection.max_frames_per_episode

        # spacemouse 模式下的夹爪宽度跟踪（STAY 时保持上一帧值）
        last_gripper_width = cfg.gripper_width_max_m

        with VideoEncodingManager(dataset):
            while saved_episodes < cfg.dataset.num_episodes and not stop_requested:
                loop_t = time.perf_counter()

                for key in _drain_keys(key_queue):
                    buffer_size = _episode_buffer_size(dataset)

                    if key == "c":
                        if not collecting:
                            collecting = True
                            log_say("Start collecting", cfg.play_sounds)
                            logger.info("Collecting episode %d from frame %d", dataset.num_episodes, buffer_size)
                    elif key == "space":
                        if collecting:
                            collecting = False
                            log_say("Pause collecting", cfg.play_sounds)
                            logger.info("Paused at %d buffered frame(s)", buffer_size)
                    elif key == "s":
                        if collecting:
                            logger.warning("Pause with space before saving the current episode.")
                            continue
                        if buffer_size < cfg.collection.min_frames_before_save:
                            logger.warning(
                                "Not saving: buffer has %d frame(s), minimum is %d.",
                                buffer_size,
                                cfg.collection.min_frames_before_save,
                            )
                            continue
                        log_say("Save episode", cfg.play_sounds)
                        dataset.save_episode()
                        saved_episodes += 1
                        logger.info(
                            "Saved episode %d/%d | total_episodes=%d total_frames=%d",
                            saved_episodes,
                            cfg.dataset.num_episodes,
                            dataset.meta.total_episodes,
                            dataset.meta.total_frames,
                        )
                        # 保存完成后自动回 home，为下一个 episode 做准备
                        if teleop_device is not None and saved_episodes < cfg.dataset.num_episodes:
                            _go_home_and_restart_cartesian(robot)
                    elif key == "r":
                        collecting = False
                        dataset.clear_episode_buffer()
                        log_say("Clear episode buffer", cfg.play_sounds)
                        logger.info("Current episode buffer cleared.")
                    elif key == "q":
                        stop_requested = True
                        collecting = False
                        logger.info("Exit requested.")

                if stop_requested:
                    break

                # 处理 spacemouse 按键事件（与键盘热键并行生效）
                # 注意：TERMINATE_EPISODE / SUCCESS 事件不在此处理，避免与夹爪按键冲突
                # （右键 = gripper_close_button = terminate_button，两者共用同一物理按键）
                # episode 的保存/退出统一由键盘 s/q 控制。
                if teleop_device is not None and hasattr(teleop_device, "get_teleop_events"):
                    teleop_events = teleop_device.get_teleop_events()
                    if teleop_events.get(TeleopEvents.RERECORD_EPISODE, False):
                        collecting = False
                        dataset.clear_episode_buffer()
                        log_say("Clear episode buffer (spacemouse)", cfg.play_sounds)
                        logger.info("SpaceMouse 触发重录：已清空当前 episode 缓冲。")

                if not collecting:
                    precise_sleep(0.05)
                    continue

                start_frame_t = time.perf_counter()
                obs = robot.get_observation()
                obs_processed = robot_obs_proc(obs)

                if teleop_device is None:
                    # 原有行为：机械臂由外部驱动，将观测复制为动作
                    action_dict = {k: obs_processed[k] for k in robot.action_features if k in obs_processed}
                else:
                    # SpaceMouse 闭环驱动：delta EE → absolute EE pose
                    delta = teleop_device.get_action()

                    # 从当前观测构造 SE(3) 参考位姿
                    pos = np.array([obs_processed[k] for k in ("ee_pose.x", "ee_pose.y", "ee_pose.z")])
                    rvec = np.array([obs_processed[k] for k in ("ee_pose.rx", "ee_pose.ry", "ee_pose.rz")])
                    T_curr = pose_xyz_rotvec_to_se3(pos, rvec)

                    # SpaceMouse 输出的 6-DoF 增量（已缩放，单位：米/弧度）
                    dx = float(delta.get("delta_x", 0.0))
                    dy = float(delta.get("delta_y", 0.0))
                    dz = float(delta.get("delta_z", 0.0))
                    dr = float(delta.get("delta_roll", 0.0))
                    dp = float(delta.get("delta_pitch", 0.0))
                    dyw = float(delta.get("delta_yaw", 0.0))

                    # 积分到目标绝对位姿：
                    #   平移 —— 直接在基坐标系中叠加，保证"向上拨杆=臂向上运动"
                    #   旋转 —— 在末端坐标系（body frame）中右乘，保证绕夹爪自身轴旋转
                    R_curr = T_curr[:3, :3]
                    R_delta = R.from_rotvec([dr, dp, dyw]).as_matrix()
                    T_target = np.eye(4)
                    T_target[:3, :3] = R_curr @ R_delta  # 末端坐标系旋转增量
                    T_target[:3, 3] = T_curr[:3, 3] + np.array([dx, dy, dz])  # 基坐标系平移增量
                    pose6 = se3_to_xyz_rotvec(T_target)

                    action_dict = {key: float(pose6[i]) for i, key in enumerate(EE_POSE_KEYS)}

                    # 夹爪宽度映射（0=合/1=保持/2=开）
                    if "gripper.width" in robot.action_features:
                        g = int(delta.get("gripper", 1))
                        if g == 2:  # OPEN
                            last_gripper_width = cfg.gripper_width_max_m
                        elif g == 0:  # CLOSE
                            last_gripper_width = cfg.gripper_width_min_m
                        # g == 1 (STAY): last_gripper_width 保持不变
                        action_dict["gripper.width"] = last_gripper_width

                    robot.send_action(action_dict)

                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                dataset.add_frame(
                    {
                        **observation_frame,
                        **action_frame,
                        "task": cfg.dataset.single_task,
                    }
                )

                buffer_size = _episode_buffer_size(dataset)
                if cfg.display_data:
                    log_rerun_data(
                        observation=obs_processed,
                        action=action_dict,
                        compress_images=False,
                    )

                if cfg.collection.print_every_n_frames > 0 and buffer_size % cfg.collection.print_every_n_frames == 0:
                    logger.info("Buffered %d frame(s) in current episode.", buffer_size)

                if max_frames is not None and buffer_size >= max_frames:
                    collecting = False
                    logger.info("Reached max_frames_per_episode=%d; paused.", max_frames)

                dt_s = time.perf_counter() - start_frame_t
                precise_sleep(max(frame_period_s - dt_s, 0.0))

                loop_dt_s = time.perf_counter() - loop_t
                if loop_dt_s > frame_period_s * 1.5:
                    logger.debug("Collection loop slower than target FPS: %.1f Hz", 1.0 / loop_dt_s)

            if saved_episodes >= cfg.dataset.num_episodes:
                logger.info("Reached requested num_episodes=%d.", cfg.dataset.num_episodes)

            unsaved_frames = _episode_buffer_size(dataset)
            if unsaved_frames > 0:
                logger.warning(
                    "Discarding %d unsaved buffered frame(s). Press 's' before 'q' to save them.",
                    unsaved_frames,
                )
                dataset.clear_episode_buffer()

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset is not None:
            dataset.finalize()

        # 关闭 Cartesian 控制器后再断开连接（polymetis 要求）
        if teleop_device is not None and robot.is_connected:
            if hasattr(robot, "terminate_policy"):
                try:
                    robot.terminate_policy()
                except Exception as exc:
                    logger.warning("terminate_policy 失败（可忽略）: %s", exc)
            elif hasattr(robot, "_franka") and hasattr(robot._franka, "terminate_policy"):
                try:
                    robot._franka.terminate_policy()
                except Exception as exc:
                    logger.warning("terminate_policy 失败（可忽略）: %s", exc)

        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception as exc:
                logger.warning("teleop disconnect 失败（可忽略）: %s", exc)

        if robot.is_connected:
            robot.disconnect()

        if listener is not None:
            listener.stop()

        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


if __name__ == "__main__":
    collect()
