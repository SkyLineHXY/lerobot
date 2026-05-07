"""UMI ee6d 坐标变换工具。

实现 ``docs/umi/UMI ee6d 位姿变换推理.md`` 中的数学推导（E ≡ F = panda_link8）。

## 两种 ee6d 约定与对应公式

### 模式 A — EE-at-t₀（推荐，数据集标准格式）

    δ_k = Δ_k = ᶠ⁽ᵗ⁰⁾T_F(t_k) = inv(ᵂT_F(t₀)) · ᵂT_F(t_k)  ∈ SE(3)

推理（doc §4.3 Eq.3）：

    ᴮT_E(t_k) = ᴮT_E(t₀) · Δ_k

调用：``ee_relative_action_to_base(delta_k, T_BE_t0)``

数据集预处理（doc §9.4）：

    ᵂT_F(t) = ᵂT_C(t) · ᶜT_F = ᵂT_C(t) · inv(ᶠT_C)
    Δ_k     = inv(ᵂT_F(t₀)) · ᵂT_F(t_k)

调用：``compute_ee_at_t0_deltas(W_T_C_list, T_FC)``

### 模式 B — Camera-at-t₀（δ_k 是原始相机系相对运动）

    δ_k = ΔC_k = inv(ᵂT_C(t₀)) · ᵂT_C(t_k)  ∈ SE(3)

推理（doc §3.2 共轭变换）：

    ᴮT_E(t_k) = ᴮT_E(t₀) · ᶠT_C · ΔC_k · inv(ᶠT_C)

调用：``umi_camera_t0_action_to_base(delta_k, T_BE_t0, T_FC)``

两种模式在数学上等价（Δ_k = ᶠT_C · ΔC_k · inv(ᶠT_C)），
但**输入 delta_k 的含义不同**，混用会导致系统性旋转误差。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


# ----------------------------------------------------------------------
# Calibration loading
# ----------------------------------------------------------------------

def load_flange_to_camera_extrinsic(yaml_path: str | Path) -> np.ndarray:
    """Load ᶠT_C (panda_link8 → camera_color_optical_frame) from a YAML file.

    The expected layout matches franka_easy_handeye output::

        camera_transform:
          translation: {x, y, z}
          rotation:    {x, y, z, w}
          parent_frame: panda_link8
          child_frame:  camera_color_optical_frame

    Returns:
        4×4 SE(3) homogeneous transform ᶠT_C (camera frame expressed in the
        flange frame) — i.e. p_F = ᶠT_C · p_C.
    """
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as f:
        data = yaml.safe_load(f)

    if "camera_transform" not in data:
        raise KeyError(f"'camera_transform' key not found in {yaml_path}")
    payload = data["camera_transform"]

    parent = payload.get("parent_frame", "panda_link8")
    if parent != "panda_link8":
        raise ValueError(
            f"Unexpected parent_frame {parent!r} in {yaml_path} — expected 'panda_link8'. "
            "If your hand-eye calibration uses a different flange link, edit this check."
        )

    t = payload["translation"]
    q = payload["rotation"]
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [t["x"], t["y"], t["z"]]
    return T


# ----------------------------------------------------------------------
# SE(3) <-> 6D pose helpers
# ----------------------------------------------------------------------

def pose_xyz_quat_to_se3(pos: Sequence[float], quat_xyzw: Sequence[float]) -> np.ndarray:
    """Build 4×4 SE(3) from (xyz position, xyzw quaternion)."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    return T


def pose_xyz_rotvec_to_se3(pos: Sequence[float], rotvec: Sequence[float]) -> np.ndarray:
    """Build 4×4 SE(3) from (xyz position, axis-angle rotvec)."""
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = pos
    return T


def se3_to_xyz_rotvec(T: np.ndarray) -> np.ndarray:
    """Decompose 4×4 SE(3) into [x, y, z, rx, ry, rz] using axis-angle rotvec.

    Matches the 6-D pose convention expected by Franka's
    ``robot_update_desired_ee_pose`` / ``ee_pose.{x..rz}`` action keys.
    """
    pos = T[:3, 3]
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


def se3_to_xyz_quat(T: np.ndarray) -> np.ndarray:
    """Decompose 4×4 SE(3) into [x, y, z, qx, qy, qz, qw]."""
    pos = T[:3, 3]
    quat = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return np.concatenate([pos, quat])


# ----------------------------------------------------------------------
# UMI ee6d → robot base frame
# ----------------------------------------------------------------------

def umi_camera_t0_action_to_base(
    delta_k: np.ndarray,
    T_BE_t0: np.ndarray,
    T_FC: np.ndarray,
) -> np.ndarray:
    """Camera-at-t₀ 原始相机运动 → 基坐标系目标法兰位姿。

    **输入约定**：``delta_k`` 必须是相机系世界相对运动
    ``ΔC_k = inv(ᵂT_C(t₀)) · ᵂT_C(t_k)``，即原始 SLAM 输出的归一化相机轨迹，
    **未经** ᶜT_F 共轭。若数据集已预处理为 EE-at-t₀ 法兰格式（Δ_k），
    请改用 ``ee_relative_action_to_base``。

    推导（doc §3.2）：

        Δ_k = ᶠT_C · ΔC_k · inv(ᶠT_C)           # 共轭到法兰系
        ᴮT_E(t_k) = ᴮT_E(t₀) · Δ_k
                   = ᴮT_E(t₀) · ᶠT_C · ΔC_k · inv(ᶠT_C)

    Args:
        delta_k:  (4, 4) 原始相机相对运动 ΔC_k ∈ SE(3)（Camera-at-t₀ 参考）。
        T_BE_t0:  (4, 4) t₀ 时刻 FK 快照 ᴮT_E。
        T_FC:     (4, 4) 法兰→相机外参 ᶠT_C（hand-eye 标定结果）。
    """
    T_CF = np.linalg.inv(T_FC)
    return T_BE_t0 @ T_FC @ delta_k @ T_CF


def ee_relative_action_to_base(
    delta_k: np.ndarray,
    T_BE_t0: np.ndarray,
) -> np.ndarray:
    """EE-at-t₀ 法兰相对位姿 → 基坐标系目标法兰位姿：ᴮT_E(t_k) = ᴮT_E(t₀) · Δ_k。

    ``delta_k`` 应为经 ``compute_ee_at_t0_deltas`` 预处理的 Δ_k（EE-at-t₀ 法兰格式）。
    """
    return T_BE_t0 @ delta_k


def compute_ee_at_t0_deltas(
    W_T_C_list: list[np.ndarray],
    T_FC: np.ndarray,
) -> list[np.ndarray]:
    """将相机轨迹转换为 EE-at-t₀ 法兰相对位姿序列（数据集预处理，doc §9.4）。

    实现以下流水线（E ≡ F = panda_link8）：

        T_CF       = inv(ᶠT_C)
        ᵂT_F(t)    = ᵂT_C(t) · T_CF
        Δ_k        = inv(ᵂT_F(t₀)) · ᵂT_F(t_k)

    满足 Δ_0 = I（单位矩阵），即 t₀ 时刻法兰位姿即为参考原点。

    Args:
        W_T_C_list: 相机轨迹，每帧 ᵂT_C(t) ∈ SE(3) 的 (4,4) 数组列表，
                    以 t₀ 时刻为起点（index 0）。
        T_FC:       (4, 4) 法兰→相机外参 ᶠT_C（hand-eye 标定，由
                    ``load_flange_to_camera_extrinsic`` 加载）。

    Returns:
        与 ``W_T_C_list`` 等长的 Δ_k 列表，Δ_0 = I。
    """
    T_CF = np.linalg.inv(T_FC)
    W_T_F_list = [W_T_C @ T_CF for W_T_C in W_T_C_list]
    W_T_F_t0_inv = np.linalg.inv(W_T_F_list[0])
    return [W_T_F_t0_inv @ W_T_F for W_T_F in W_T_F_list]


def absolute_position_world_orientation_to_base(
    delta_k: np.ndarray,
    T_BE_t0: np.ndarray,
) -> np.ndarray:
    """Hybrid: position is t₀-relative (in base frame), orientation is absolute.

    Useful when the dataset stores ``pos_t = p_t - p_t0`` together with the
    EE orientation directly in the base / world frame (a common artefact of
    "zeroed-position + absolute-quaternion" preprocessing pipelines).
    """
    T = np.eye(4)
    T[:3, :3] = delta_k[:3, :3]
    T[:3, 3] = T_BE_t0[:3, 3] + delta_k[:3, 3]
    return T
