"""FrankaMultiCam 机器人数据采集脚本。

专为 FrankaMultiCam（Franka EE + 多路相机）设计。机械臂由外部系统驱动，
本脚本只读取 ``robot.get_observation()``，将 EE pose（含夹爪）同时作为
observation 和 action 写入 LeRobot v3 数据集。

用法::

    python -m lerobot.scripts.lerobot_record_franka_multicam \\
        --robot.type=franka_multicam \\
        --robot.robot_ip=172.20.10.2 \\
        --robot.robot_port=4242 \\
        --robot.cameras='{
            "cam_top":{"type":"opencv_zmq","mode":"remote","host":"172.20.10.2","port":4244,
                       "topic":"cam_top","fps":30,"width":640,"height":480,"index_or_path":0},
            "cam_left":{"type":"opencv_zmq","mode":"remote","host":"172.20.10.2","port":4245,
                        "topic":"cam_left","fps":30,"width":640,"height":480,"index_or_path":0},
            "cam_right":{"type":"opencv_zmq","mode":"remote","host":"172.20.10.2","port":4246,
                         "topic":"cam_right","fps":30,"width":640,"height":480,"index_or_path":0}
        }' \\
        --dataset.repo_id=zzq/franka_multicam_demo \\
        --dataset.root=/home/zzq/lerobot/datasets/franka_multicam_demo \\
        --dataset.single_task="franka multicam demo" \\
        --dataset.fps=30 \\
        --dataset.num_episodes=2 \\
        --dataset.episode_time_s=10 \\
        --dataset.push_to_hub=false

键盘控制：
  - 右箭头：结束当前 episode（提前）
  - 左箭头：丢弃当前 episode 并重录
  - ESC：停止采集
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    combine_feature_dicts,
    create_initial_features,
)
from lerobot.datasets.utils import build_dataset_frame
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.franka_multicam.config_franka_multicam import FrankaMultiCamConfig  # noqa: F401
from lerobot.cameras.hik_camera.configuration_hik import HikCameraConfig  # noqa: F401 — 触发 draccus 注册
from lerobot.cameras.opencv_zmq.configuration_opencv_zmq import OpenCVZmqCameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots.utils import make_robot_from_config
from lerobot.scripts.lerobot_record import (  # 复用录制脚本的配置与工具
    DatasetRecordConfig,
    make_default_processors,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

logger = logging.getLogger(__name__)


@dataclass
class RecordFrankaMultiCamConfig:
    """FrankaMultiCam 数据采集配置。"""

    robot: FrankaMultiCamConfig
    """FrankaMultiCam 机器人配置（含 cameras 字典）。"""

    dataset: DatasetRecordConfig
    """LeRobot v3 数据集录制参数。"""

    display_data: bool = False
    """是否启用 rerun 可视化。"""

    display_ip: str | None = None
    """Rerun 可视化 IP。"""

    display_port: int | None = None
    """Rerun 可视化端口。"""

    play_sounds: bool = True
    """是否播放录制提示音。"""

    resume: bool = False
    """是否续录已有数据集。"""


@parser.wrap()
def record(cfg: RecordFrankaMultiCamConfig) -> LeRobotDataset:
    init_logging()
    logger.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)

    # 1. 实例化 robot
    robot = make_robot_from_config(cfg.robot)

    # 2. 创建默认 processor pipeline
    _, robot_action_proc, robot_obs_proc = make_default_processors()

    # 3. 组装 dataset features（action 与 observation 共享 EE pose + gripper key）
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

    # 4. 创建或续录数据集
    dataset = None
    listener = None

    try:
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
            _num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            if _num_cameras > 0 and not cfg.dataset.streaming_encoding:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * _num_cameras,
                )
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, None)
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                    if num_cameras > 0
                    else 0
                ),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        # 5. 连接
        robot.connect()
        listener, events = init_keyboard_listener()

        fps = cfg.dataset.fps
        control_time_s = cfg.dataset.episode_time_s
        single_task = cfg.dataset.single_task

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)

                timestamp = 0.0
                start_episode_t = time.perf_counter()
                while timestamp < control_time_s:
                    start_loop_t = time.perf_counter()

                    if events["exit_early"]:
                        events["exit_early"] = False
                        break

                    # 获取观测
                    obs = robot.get_observation()
                    obs_processed = robot_obs_proc(obs)

                    # 构造 action：从 observation 中复制 EE pose + gripper 同名 key
                    action_dict = {}
                    for k in robot.action_features:
                        if k in obs_processed:
                            action_dict[k] = obs_processed[k]

                    # 写入数据集
                    observation_frame = build_dataset_frame(
                        dataset.features, obs_processed, prefix=OBS_STR
                    )
                    action_frame = build_dataset_frame(
                        dataset.features, action_dict, prefix=ACTION
                    )
                    dataset.add_frame({
                        **observation_frame,
                        **action_frame,
                        "task": single_task,
                    })
                    if cfg.display_data:
                        log_rerun_data(
                            observation=obs_processed, action=action_dict, compress_images=False
                        )

                    # 维持帧率
                    dt_s = time.perf_counter() - start_loop_t
                    precise_sleep(max(1.0 / fps - dt_s, 0.0))

                    timestamp = time.perf_counter() - start_episode_t

                # 处理 episode 结束
                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    return dataset


if __name__ == "__main__":
    record()
