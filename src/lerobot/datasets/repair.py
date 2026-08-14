#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""把被硬杀中断的 LeRobot v3 数据集修回自洽状态，让断点续采能起来。

数据和元数据的耐久性不一样：episode 数据走一个常驻的 `pq.ParquetWriter`，footer 只有
`close()`（即 `finalize()`）时才写；而元数据是每集 flush 一次的。进程被 `kill -9` 后，
要么元数据引用了不可读的数据，要么数据没有任何元数据引用它。

两种情况下 `LeRobotDataset.__init__` 都会认定本地缓存不完整，转去访问 HuggingFace Hub；
对纯本地的 repo_id 那个请求会一直挂着，现象就是"续采卡死"。

采集只往后追加，损坏一定在尾部，所以这里保留最长的合法前缀（数据与元数据在每一集长度
上都对得上）并改写计数器。被摘掉的东西都移进 `_quarantine_*`，绝不删除。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["repair_dataset_consistency"]


def _parse_video_path(rel_parts: tuple, stem: str) -> tuple[str, int, int]:
    """('observation.images.cam_top', 'chunk-000', 'file-001.mp4') → (key, 0, 1)。"""
    key = rel_parts[0]
    chunk_idx = int(rel_parts[1].split("-")[1])
    file_idx = int(stem.split("-")[1])
    return key, chunk_idx, file_idx


def _quarantine(path: str | Path, quarantine_dir: str | Path) -> Path:
    """把 *path* 移进 *quarantine_dir*，文件名里保留原目录名以便事后追溯。"""
    path = Path(path)
    dst_dir = Path(quarantine_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{path.parent.name}__{path.name}"
    shutil.move(str(path), str(dst))
    return dst


def _truncate_parquet_by_episode(paths, keep_before: int, quarantine_dir: str | Path) -> None:
    """只保留 `episode_index < keep_before` 的行；被清空的文件整体隔离。

    在 pyarrow 层重写以原样保留 schema —— 绕一圈 pandas 会重新推断类型，list 列很容易变掉。
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for p in paths:
        table = pq.read_table(p)
        if "episode_index" not in table.schema.names:
            continue
        kept = table.filter(pc.less(table.column("episode_index"), keep_before))
        if kept.num_rows == table.num_rows:
            continue
        if kept.num_rows == 0:
            _quarantine(p, quarantine_dir)
            continue
        tmp = f"{p}.tmp"
        pq.write_table(kept, tmp, compression="snappy", use_dictionary=True)
        os.replace(tmp, p)


def repair_dataset_consistency(target_dir: str | Path) -> int:
    """把 *target_dir* 下的数据集截断到最后一集完整的 episode，返回保留的集数。

    **必须在构造 `LeRobotDataset` 之前调用** —— 一旦进了 `__init__`，不一致就会触发
    那个会挂死的 Hub 请求。
    """
    import pandas as pd
    import pyarrow.parquet as pq

    root = Path(target_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return 0

    qdir = root / f"_quarantine_{time.strftime('%Y%m%d_%H%M%S')}"

    # 1. 摘掉崩溃留下的、没有 footer 的 parquet
    corrupt: list[str] = []
    for p in sorted(root.rglob("*.parquet")):
        if qdir in p.parents:
            continue
        try:
            pq.read_metadata(p)
        except Exception:
            corrupt.append(str(p.relative_to(root)))
            _quarantine(p, qdir)
    if corrupt:
        logger.warning(
            "隔离了 %d 个上次崩溃留下的不可读 parquet：\n  %s",
            len(corrupt),
            "\n  ".join(corrupt),
        )

    ep_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not ep_files or not data_files:
        logger.warning("没有找到 episode 元数据或数据文件，无从修复。")
        return 0

    # 2. 元数据声明的集数 vs 磁盘上真实存在的行
    eps = pd.concat([pd.read_parquet(f) for f in ep_files])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    actual = pd.concat([pd.read_parquet(f, columns=["episode_index"]) for f in data_files])
    counts = actual["episode_index"].value_counts()

    # 3. 找出数据完整的最长前缀
    keep = 0
    for i, row in eps.iterrows():
        idx = int(row["episode_index"])
        if idx != i:  # 编号断档
            break
        if int(counts.get(idx, 0)) != int(row["length"]):
            break
        keep = i + 1

    declared_frames = int(eps["length"].sum())
    orphan_rows = int((actual["episode_index"] >= keep).sum())
    if keep == len(eps) and orphan_rows == 0 and int(counts.sum()) == declared_frames:
        logger.info("数据集自洽：%d 集 / %d 帧。", keep, declared_frames)
        return keep

    kept_frames = int(eps.iloc[:keep]["length"].sum()) if keep else 0
    logger.warning(
        "检测到数据集不一致，截断到最后一集完整的 episode：\n"
        "  集数：声明 %d → 保留 %d（丢弃 %d）\n"
        "  帧数：声明 %d → 保留 %d\n"
        "  没有元数据的孤儿数据行：%d",
        len(eps),
        keep,
        len(eps) - keep,
        declared_frames,
        kept_frames,
        orphan_rows,
    )

    # 4. 仍被幸存 episode 引用的视频。同时被保留集和丢弃集用到的文件予以保留：
    #    多出来的尾部帧没人引用，而 episode 按时间戳区间索引视频，不会读到那一段。
    referenced: set[tuple[str, int, int]] = set()
    kept_rows = eps.iloc[:keep]
    for col in [c for c in eps.columns if c.startswith("videos/") and c.endswith("/chunk_index")]:
        key = col[len("videos/") : -len("/chunk_index")]
        fcol = f"videos/{key}/file_index"
        if fcol not in eps.columns:
            continue
        for ci, fi in zip(kept_rows[col], kept_rows[fcol], strict=True):
            referenced.add((key, int(ci), int(fi)))

    # 5. 把元数据和数据都截到幸存前缀
    _truncate_parquet_by_episode(ep_files, keep, qdir)
    _truncate_parquet_by_episode(data_files, keep, qdir)

    # 6. 隔离不再承载任何幸存 episode 的视频
    videos_root = root / "videos"
    if videos_root.is_dir():
        for vf in sorted(videos_root.rglob("*.mp4")):
            rel = vf.relative_to(videos_root)
            try:
                ident = _parse_video_path(rel.parts, vf.stem)
            except (IndexError, ValueError):
                continue
            if ident not in referenced:
                logger.warning("  隔离无人引用的视频 %s", rel)
                _quarantine(vf, qdir)

    # 7. 流式编码器留下的临时目录
    for tmp in sorted(root.glob("tmp*")):
        if tmp.is_dir():
            qdir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(qdir / tmp.name))

    # 8. 改写计数器
    info = json.loads(info_path.read_text())
    info["total_episodes"] = keep
    info["total_frames"] = kept_frames
    info["splits"] = {"train": f"0:{keep}"}
    info_path.write_text(json.dumps(info, indent=2))

    logger.warning(
        "已修复数据集：total_episodes=%d, total_frames=%d。摘掉的文件都在 %s/ 里"
        "（确认没问题后可以自行删除）。\n"
        "  注意：meta/stats.json 里仍然含被丢弃那几集的统计。它只用于归一化，"
        "这是一点小偏差，不会导致加载失败。",
        keep,
        kept_frames,
        qdir.name,
    )
    return keep
