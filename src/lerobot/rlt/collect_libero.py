"""用键盘/手柄在 LIBERO 里人工示教，采集成 LeRobotDataset。

阶段 1 的 `train_rl_token` 需要任务演示数据。官方 `lerobot/libero_*` 数据集只覆盖
固定任务集，自建任务（或换了初始摆放、换了相机）就没有对应的演示可用，这个脚本
补的是这一段。

数据集 schema、遥操作、算子循环都在 `collect_sim.py`；这里只剩 LIBERO 自己的
suite/task 选择。列名与用法见那个模块的文档。

    python -m lerobot.rlt.collect_libero --config_path examples/rlt/libero/1_collect.yaml
    python -m lerobot.rlt.collect_libero --config_path examples/rlt/libero/1_collect.yaml \\
        --teleop.device=gamepad --dataset.repo_id=me/libero_pick_bowl
"""

# 这里不能加 `from __future__ import annotations`：parser.wrap() 直接读
# inspect.getfullargspec(fn).annotations，注解变成字符串后 draccus 取不到配置类。
from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.rlt.collect_sim import (
    ACTION_NAMES,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    GRIPPER_STAY,
    ROTATION_CHANNELS,
    STATE_DIM,
    CollectDatasetConfig,
    CollectSimConfig,
    CollectViewConfig,
    SimTeleopCollector,
    build_features,
    collect,
)

__all__ = [
    "ACTION_NAMES",
    "GRIPPER_CLOSE",
    "GRIPPER_OPEN",
    "GRIPPER_STAY",
    "ROTATION_CHANNELS",
    "STATE_DIM",
    "CollectDatasetConfig",
    "CollectLiberoConfig",
    "CollectViewConfig",
    "LiberoTeleopCollector",
    "build_features",
    "main",
]


@dataclass
class CollectLiberoConfig(CollectSimConfig):
    suite: str = "libero_10"
    task_id: int = 0


class LiberoTeleopCollector(SimTeleopCollector):
    robot_type = "libero_panda"
    window_name = "LIBERO Collect"

    def setup_env(self) -> None:
        from lerobot.envs.libero import LiberoEnv, _get_suite, _parse_camera_names

        cameras = _parse_camera_names(self.cfg.camera_name)
        keys = self.check_camera_keys(cameras)
        self.env = LiberoEnv(
            task_suite=_get_suite(self.cfg.suite),
            task_id=self.cfg.task_id,
            task_suite_name=self.cfg.suite,
            obs_type="pixels_agent_pos",
            observation_width=self.cfg.observation_size,
            observation_height=self.cfg.observation_size,
            camera_name=self.cfg.camera_name,
            camera_name_mapping=dict(zip(cameras, keys, strict=True)),
            control_mode="relative",
            episode_length=self.cfg.max_episode_steps,
        )

    def episode_task(self) -> str:
        # `LiberoEnv.task` is the bddl identifier ("open_the_middle_drawer..."),
        # `task_description` the instruction ("open the middle drawer ..."). The
        # lerobot/libero_* datasets store the latter, and a policy trained on this
        # one has to see the same string, so the underscored name must never reach
        # the `task` column.
        return self.env.task_description


@parser.wrap()
def main(cfg: CollectLiberoConfig):
    collect(LiberoTeleopCollector(cfg))


if __name__ == "__main__":
    main()
