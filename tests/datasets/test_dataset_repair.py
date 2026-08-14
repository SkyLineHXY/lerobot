"""崩溃修复的单测。

这段代码会**移动文件**，跑错了就等于把好数据搬走，所以边界要钉死：自洽的数据集必须
一动不动，不自洽的只截掉尾巴，而且任何情况下都只隔离、不删除。
"""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.repair import repair_dataset_consistency


def _build(root, ep_lengths, data_lengths=None, videos=None):
    """造一个最小可信的 v3 数据集骨架。

    `ep_lengths` 是元数据声明的每集长度，`data_lengths` 是数据文件里实际写了多少行 ——
    两者不一致就是崩溃现场。
    """
    data_lengths = ep_lengths if data_lengths is None else data_lengths
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    eps = {"episode_index": list(range(len(ep_lengths))), "length": list(ep_lengths)}
    if videos:
        eps["videos/observation.images.cam/chunk_index"] = [0] * len(ep_lengths)
        eps["videos/observation.images.cam/file_index"] = list(videos)
    pq.write_table(pa.table(eps), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    rows = []
    for idx, n in enumerate(data_lengths):
        rows.extend([idx] * n)
    pq.write_table(
        pa.table({"episode_index": rows, "value": list(range(len(rows)))}),
        root / "data" / "chunk-000" / "file-000.parquet",
    )

    (root / "meta").mkdir(exist_ok=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": len(ep_lengths),
                "total_frames": sum(ep_lengths),
                "splits": {"train": f"0:{len(ep_lengths)}"},
            }
        )
    )
    return root


def _info(root):
    return json.loads((root / "meta" / "info.json").read_text())


def _quarantines(root):
    return [p for p in root.glob("_quarantine_*") if p.is_dir()]


def test_missing_dataset_is_a_noop(tmp_path):
    assert repair_dataset_consistency(tmp_path) == 0


def test_consistent_dataset_is_left_untouched(tmp_path):
    root = _build(tmp_path, [5, 5, 5])
    before = sorted(p.name for p in root.rglob("*"))

    assert repair_dataset_consistency(root) == 3
    assert sorted(p.name for p in root.rglob("*")) == before
    assert _quarantines(root) == []
    assert _info(root)["total_episodes"] == 3


def test_metadata_ahead_of_data_is_truncated(tmp_path):
    """元数据声明了 3 集，数据只写进去 2 集 —— 典型的硬杀现场。"""
    root = _build(tmp_path, [5, 5, 5], data_lengths=[5, 5, 0])

    assert repair_dataset_consistency(root) == 2
    info = _info(root)
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 10
    assert info["splits"] == {"train": "0:2"}

    eps = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    assert list(eps["episode_index"]) == [0, 1]


def test_orphan_data_rows_are_dropped(tmp_path):
    """反过来：数据写到第 3 集了，元数据只 flush 到第 2 集。"""
    root = _build(tmp_path, [5, 5], data_lengths=[5, 5, 4])

    assert repair_dataset_consistency(root) == 2
    data = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    assert set(data["episode_index"]) == {0, 1}
    assert len(data) == 10


def test_short_episode_truncates_at_that_episode(tmp_path):
    """第 2 集只写了一半，它以及它之后的都不能要。"""
    root = _build(tmp_path, [5, 5, 5], data_lengths=[5, 3, 5])
    assert repair_dataset_consistency(root) == 1
    assert _info(root)["total_frames"] == 5


def test_footerless_parquet_is_quarantined_not_deleted(tmp_path):
    """没有 footer 的 parquet 是读不了的，但里面可能还有能人工抢救的东西。"""
    root = _build(tmp_path, [5, 5])
    broken = root / "data" / "chunk-000" / "file-001.parquet"
    broken.write_bytes(b"PAR1\x00\x00truncated")

    repair_dataset_consistency(root)
    assert not broken.exists()
    quarantined = _quarantines(root)
    assert len(quarantined) == 1
    assert any("file-001.parquet" in p.name for p in quarantined[0].iterdir())


def test_unreferenced_videos_are_quarantined(tmp_path):
    root = _build(tmp_path, [5, 5, 5], data_lengths=[5, 5, 0], videos=[0, 0, 1])
    vdir = root / "videos" / "observation.images.cam" / "chunk-000"
    vdir.mkdir(parents=True)
    kept, dropped = vdir / "file-000.mp4", vdir / "file-001.mp4"
    kept.write_bytes(b"kept")
    dropped.write_bytes(b"dropped")

    repair_dataset_consistency(root)
    assert kept.exists(), "还被保留集引用的视频不能动"
    assert not dropped.exists()


def test_repair_is_idempotent(tmp_path):
    root = _build(tmp_path, [5, 5, 5], data_lengths=[5, 5, 0])
    assert repair_dataset_consistency(root) == 2
    # 第二次跑必须什么都不改，否则每次续采都会再削掉一集
    assert repair_dataset_consistency(root) == 2
    assert _info(root)["total_episodes"] == 2


def test_empty_dataset_reports_nothing_to_repair(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0"}))
    assert repair_dataset_consistency(tmp_path) == 0


def test_first_episode_broken_keeps_nothing(tmp_path):
    """第一集就残缺时保留 0 集，计数器也要跟着改写成 0，不能留着旧值。"""
    root = _build(tmp_path, [5, 5], data_lengths=[2, 5])
    assert repair_dataset_consistency(root) == 0
    assert _info(root)["total_episodes"] == 0
    assert _info(root)["total_frames"] == 0
