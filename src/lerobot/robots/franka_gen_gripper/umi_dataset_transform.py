"""UMI world_flange 数据集的 sample-relative 训练变换。

使用场景
--------
训练 ACT / Diffusion 等策略时，LeRobotDataset 返回的 batch 中包含：

- ``action``            : (B, chunk, 7) 绝对法兰位姿 [pos3_W + rotvec3_W + gripper1]
- ``observation.umi_pose`` : (B, [n_obs,] 6) 当前观测帧的绝对法兰位姿 [pos3_W + rotvec3_W]

本模块将 ``action`` 原地转换为相对于 ``observation.umi_pose`` 最后一帧的 sample-relative
表示，使策略行为与 UMI 论文（§III-B "t₀ = 推理时刻"）对齐。

用法示例（在 DataLoader collate 或 Dataset.__getitem__ 后调用）::

    from lerobot.robots.franka_gen_gripper.umi_dataset_transform import (
        apply_umi_sample_relative_transform,
    )

    for batch in dataloader:
        batch = apply_umi_sample_relative_transform(batch)
        loss = policy.forward(batch)
"""

from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation as R


# -------------------------------------------------------------------------
# SE(3) 辅助（纯 torch，无 scipy 依赖，支持批量）
# -------------------------------------------------------------------------

def _rotvec_to_rotmat_batch(rotvec: torch.Tensor) -> torch.Tensor:
    """将 (..., 3) axis-angle 转为 (..., 3, 3) 旋转矩阵（scipy 批量版）。"""
    flat = rotvec.reshape(-1, 3).cpu().numpy()
    mats = R.from_rotvec(flat).as_matrix()
    return torch.as_tensor(mats, dtype=rotvec.dtype, device=rotvec.device).reshape(
        *rotvec.shape[:-1], 3, 3
    )


def _rotmat_to_rotvec_batch(rotmat: torch.Tensor) -> torch.Tensor:
    """将 (..., 3, 3) 旋转矩阵转为 (..., 3) axis-angle。"""
    flat = rotmat.reshape(-1, 3, 3).cpu().numpy()
    rvecs = R.from_matrix(flat).as_rotvec()
    return torch.as_tensor(rvecs, dtype=rotmat.dtype, device=rotmat.device).reshape(
        *rotmat.shape[:-2], 3
    )


def _build_se3_batch(pos: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
    """从 (..., 3) pos 和 (..., 3) rotvec 构造 (..., 4, 4) SE(3)。"""
    *batch, _ = pos.shape
    T = torch.eye(4, dtype=pos.dtype, device=pos.device).expand(*batch, 4, 4).clone()
    T[..., :3, :3] = _rotvec_to_rotmat_batch(rotvec)
    T[..., :3, 3] = pos
    return T


def _invert_se3_batch(T: torch.Tensor) -> torch.Tensor:
    """批量求逆 SE(3)：inv(T) = [R^T, -R^T @ t; 0, 1]。"""
    R_T = T[..., :3, :3].transpose(-1, -2)
    t_inv = -(R_T @ T[..., :3, 3:4]).squeeze(-1)
    inv = torch.eye(4, dtype=T.dtype, device=T.device).expand_as(T).clone()
    inv[..., :3, :3] = R_T
    inv[..., :3, 3] = t_inv
    return inv


def _decompose_se3_to_pose6(T: torch.Tensor) -> torch.Tensor:
    """从 (..., 4, 4) SE(3) 拆解为 (..., 6) [pos3, rotvec3]。"""
    pos = T[..., :3, 3]
    rotvec = _rotmat_to_rotvec_batch(T[..., :3, :3])
    return torch.cat([pos, rotvec], dim=-1)


# -------------------------------------------------------------------------
# 核心变换
# -------------------------------------------------------------------------

def apply_umi_sample_relative_transform(
    batch: dict[str, torch.Tensor],
    *,
    remove_umi_pose: bool = True,
) -> dict[str, torch.Tensor]:
    """
    将 world_flange 绝对位姿 action 转为 sample-relative，就地修改 batch。
    约定
    ----
    - ``batch["action"]``          : shape ``(B, chunk_size, 7)``，最后一维
      ``[pos_x_W, pos_y_W, pos_z_W, rotvec_x_W, rotvec_y_W, rotvec_z_W, gripper]``。
    - ``batch["observation.umi_pose"]`` : shape ``(B, 6)`` 或 ``(B, n_obs, 6)``，
      当前（最后一帧）观测的绝对法兰位姿。
    算法
    ----
    设 base = ``umi_pose[..., -1, :]``（最后一帧），对 action chunk 中每步 k::
        delta_k = inv(SE3(base)) @ SE3(action_k_pose6)
        action_k_new[:6] = decompose_to_pose6(delta_k)   # pos+rotvec
        action_k_new[6]  = gripper（不变）
    Returns
    -------
    修改后的 batch（in-place）。若 ``remove_umi_pose=True``，
    删除 ``observation.umi_pose`` 键（策略不需要它）。
    Raises
    ------
    KeyError
        若 batch 中缺少 ``action`` 或 ``observation.umi_pose``。
    ValueError
        若 action 维度不是 7 或 umi_pose 维度不是 6。
    """
    if "observation.umi_pose" not in batch:
        raise KeyError(
            "'observation.umi_pose' 不在 batch 中。"
            "请确认数据集使用 world_flange 格式（--pose-format=world_flange）。"
        )
    if "action" not in batch:
        raise KeyError("'action' 不在 batch 中。")

    action = batch["action"]       # (B, chunk, 7) 或 (chunk, 7)
    umi_pose = batch["observation.umi_pose"]  # (B, 6) 或 (B, n_obs, 6)

    squeezed = action.ndim == 2
    if squeezed:
        action = action.unsqueeze(0)
        umi_pose = umi_pose.unsqueeze(0) if umi_pose.ndim == 1 else umi_pose

    if action.shape[-1] != 7:
        raise ValueError(f"action 最后一维应为 7，实际为 {action.shape[-1]}。")
    if umi_pose.shape[-1] != 6:
        raise ValueError(f"umi_pose 最后一维应为 6，实际为 {umi_pose.shape[-1]}。")

    # 取最后一帧 obs 作为 sample t₀ 参考
    if umi_pose.ndim == 3:
        base_pose6 = umi_pose[:, -1, :]   # (B, 6)
    else:
        base_pose6 = umi_pose              # (B, 6)

    B, chunk_size, _ = action.shape
    pos_base = base_pose6[:, :3]          # (B, 3)
    rotvec_base = base_pose6[:, 3:6]      # (B, 3)
    T_base = _build_se3_batch(pos_base, rotvec_base)          # (B, 4, 4)
    T_base_inv = _invert_se3_batch(T_base)                    # (B, 4, 4)

    pos_k = action[:, :, :3]       # (B, chunk, 3)
    rotvec_k = action[:, :, 3:6]   # (B, chunk, 3)
    gripper_k = action[:, :, 6:7]  # (B, chunk, 1)

    T_k = _build_se3_batch(
        pos_k.reshape(B * chunk_size, 3),
        rotvec_k.reshape(B * chunk_size, 3),
    ).reshape(B, chunk_size, 4, 4)  # (B, chunk, 4, 4)

    # delta_k = inv(T_base) @ T_k，广播：(B,1,4,4) @ (B,chunk,4,4)
    T_delta = T_base_inv.unsqueeze(1) @ T_k  # (B, chunk, 4, 4)

    pose6_rel = _decompose_se3_to_pose6(T_delta)  # (B, chunk, 6)
    new_action = torch.cat([pose6_rel, gripper_k], dim=-1)  # (B, chunk, 7)

    if squeezed:
        new_action = new_action.squeeze(0)

    batch["action"] = new_action

    if remove_umi_pose:
        del batch["observation.umi_pose"]

    return batch
