"""实现成功场景独立 LMDB、内容校验、compact 发布和可修复 checkpoint。

模块: data/data_collector/storage/storage.py
依赖: lmdb, msgpack, numpy, zstandard, config, data.data_collector.records
读取配置: data_collector.collector.output, data_collector.collector.master_seed,
    data_collector.storage.*
对外接口:
    - DatasetStore
    - encode_value(value, compression_level) -> bytes
    - decode_value(value) -> Any
    - config_fingerprint(cfg) -> str
    - validate_scene(path, cfg, deep=False) -> dict
    - validate_dataset(path, cfg, deep=False) -> dict
    - compact_scene(path, cfg) -> dict
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any
import uuid

import lmdb
import msgpack
import numpy as np
import zstandard

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.data_collector.records import SceneRecord
from data.data_collector.storage.checks.storage_checks import check_dataset_path, check_scene_record


SCHEMA_VERSION = "1.0.0"
SCENE_PATTERN = re.compile(r"^scene_(\d{6})_([0-9a-f]{12})\.lmdb$")


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {"__ndarray__": True, "dtype": contiguous.dtype.str, "shape": contiguous.shape, "data": contiguous.tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化类型: {type(value).__name__}")


def _msgpack_hook(value: dict[str, Any]) -> Any:
    if value.get("__ndarray__"):
        return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"]).copy()
    return value


def encode_value(value: Any, compression_level: int) -> bytes:
    """把含 NumPy 数组的对象封装为带校验的压缩值。"""
    raw = msgpack.packb(value, default=_msgpack_default, use_bin_type=True)
    envelope = {
        "schema": SCHEMA_VERSION,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "codec": "zstd",
        "payload": zstandard.ZstdCompressor(level=compression_level).compress(raw),
    }
    return msgpack.packb(envelope, use_bin_type=True)


def decode_value(value: bytes) -> Any:
    """校验并解码一个 LMDB 压缩值。"""
    envelope = msgpack.unpackb(value, raw=False)
    raw = zstandard.ZstdDecompressor().decompress(envelope["payload"])
    if hashlib.sha256(raw).hexdigest() != envelope["sha256"]:
        raise ValueError("LMDB value 内容哈希不匹配")
    return msgpack.unpackb(raw, raw=False, object_hook=_msgpack_hook)


def config_fingerprint(cfg: AppConfig) -> str:
    """计算排除采集目标数、路径和恢复开关后的兼容配置指纹。"""
    # data_vis 只影响派生产物，不得让同一采集数据集变得不兼容。
    data = {"data_collector": asdict(cfg.data_collector)}
    collector = data["data_collector"]["collector"]
    for key in ("scene_count", "resume", "output"):
        collector.pop(key)
    data["data_collector"]["render"].pop("viewer")
    storage = data["data_collector"]["storage"]
    storage.pop("atomic_replace_attempts")
    storage.pop("atomic_replace_retry_seconds")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, data: dict[str, Any], cfg: AppConfig) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    storage = cfg.data_collector.storage
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(storage.atomic_replace_attempts):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:
                transient = isinstance(error, PermissionError) or getattr(error, "winerror", None) in {5, 32, 33}
                if not transient or attempt + 1 >= storage.atomic_replace_attempts:
                    raise
                time.sleep(storage.atomic_replace_retry_seconds)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_remove(path: Path, boundary: Path) -> None:
    resolved = path.resolve()
    root = boundary.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"拒绝清理越界路径: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _open_databases(env: lmdb.Environment, create: bool) -> tuple[Any, Any, Any]:
    return (
        env.open_db(b"meta", create=create),
        env.open_db(b"frames", create=create),
        env.open_db(b"index", create=create),
    )


def validate_scene(path: str | Path, cfg: AppConfig, deep: bool = False) -> dict[str, Any]:
    """只读验证单个成功场景 LMDB。"""
    scene_path = Path(path)
    env = lmdb.open(str(scene_path), readonly=True, lock=False, readahead=False, max_dbs=cfg.data_collector.storage.max_dbs)
    try:
        meta_db, frames_db, index_db = _open_databases(env, create=False)
        with env.begin() as transaction:
            complete = transaction.get(b"complete", db=meta_db)
            encoded_meta = transaction.get(b"scene", db=meta_db)
            encoded_summary = transaction.get(b"summary", db=index_db)
            if complete != b"1" or encoded_meta is None or encoded_summary is None:
                raise ValueError(f"场景未完整提交: {scene_path.name}")
            metadata = decode_value(encoded_meta)
            summary = decode_value(encoded_summary)
            frame_count = int(summary["frame_count"])
            if not metadata["success_evidence"].get("success") or frame_count <= 0:
                raise ValueError(f"场景缺少成功证据或帧: {scene_path.name}")
            cursor = transaction.cursor(db=frames_db)
            keys = [key for key, _ in cursor]
            expected = [str(index).zfill(cfg.data_collector.storage.frame_key_width).encode("ascii") for index in range(frame_count)]
            if keys != expected:
                raise ValueError(f"场景帧索引不连续: {scene_path.name}")
            if deep:
                hashes = [hashlib.sha256(transaction.get(key, db=frames_db)).hexdigest() for key in expected]
                if hashes != summary["frame_hashes"]:
                    raise ValueError(f"场景帧摘要不匹配: {scene_path.name}")
                for key in expected:
                    decode_value(transaction.get(key, db=frames_db))
            return {"scene_id": metadata["scene_id"], "scene_index": metadata["spec"]["scene_index"], "frame_count": frame_count, "task_type": metadata["spec"]["task"]["task_type"]}
    finally:
        env.close()


def _copy_compact(source: Path, destination: Path, cfg: AppConfig) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    env = lmdb.open(str(source), readonly=True, lock=False, max_dbs=cfg.data_collector.storage.max_dbs)
    try:
        env.copy(str(destination), compact=True)
    finally:
        env.close()


def compact_scene(path: str | Path, cfg: AppConfig) -> dict[str, Any]:
    """用已校验 compact 副本原子替换单个最终场景。"""
    scene_path = Path(path).resolve()
    check_dataset_path(scene_path.parent)
    before = validate_scene(scene_path, cfg, deep=True)
    before_bytes = _directory_size(scene_path)
    compact_path = scene_path.with_name(scene_path.name + ".compact")
    backup_path = scene_path.with_name(scene_path.name + ".backup")
    _safe_remove(compact_path, scene_path.parent)
    _safe_remove(backup_path, scene_path.parent)
    _copy_compact(scene_path, compact_path, cfg)
    after = validate_scene(compact_path, cfg, deep=True)
    if before != after:
        _safe_remove(compact_path, scene_path.parent)
        raise ValueError("compact 副本逻辑内容不一致")
    scene_path.rename(backup_path)
    compact_path.rename(scene_path)
    validate_scene(scene_path, cfg, deep=True)
    _safe_remove(backup_path, scene_path.parent)
    return {"scene": scene_path.name, "before_bytes": before_bytes, "after_bytes": _directory_size(scene_path)}


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


class DatasetStore:
    """管理数据集清单、成功场景发布和断点续采状态。"""

    def __init__(self, root: str | Path, cfg: AppConfig, asset_hash: str):
        self.root = Path(root).resolve()
        check_dataset_path(self.root)
        self.cfg = cfg
        self.asset_hash = asset_hash
        self.config_hash = config_fingerprint(cfg)
        self.staging = self.root / "staging"

    def initialize(self) -> None:
        """创建或验证数据集清单，并恢复已完整写入的 staging。"""
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(exist_ok=True)
        manifest_path = self.root / "dataset.json"
        expected = {
            "schema_version": SCHEMA_VERSION,
            "master_seed": self.cfg.data_collector.collector.master_seed,
            "config_hash": self.config_hash,
            "asset_hash": self.asset_hash,
        }
        if manifest_path.exists():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(current.get(key) != value for key, value in expected.items()):
                raise ValueError("现有数据集与当前 schema、seed、配置或资产不兼容")
        else:
            _atomic_json(manifest_path, expected, self.cfg)
        self._recover_staging()

    def _recover_staging(self) -> None:
        for partial in sorted(self.staging.glob("scene_*.partial")):
            match = re.match(r"^scene_(\d{6})_([0-9a-f]{12})\.partial$", partial.name)
            if match is None:
                continue
            final = self.root / f"scene_{match.group(1)}_{match.group(2)}.lmdb"
            try:
                validate_scene(partial, self.cfg, deep=True)
            except Exception:
                _safe_remove(partial, self.staging)
                continue
            existing_for_index = list(self.root.glob(f"scene_{match.group(1)}_*.lmdb"))
            if final.exists() or existing_for_index:
                _safe_remove(partial, self.staging)
                continue
            self._publish_compacted(partial, final)

    def scan_completed(self) -> list[Path]:
        """扫描并验证连续的最终成功场景前缀。"""
        indexed: list[tuple[int, Path]] = []
        for path in self.root.glob("scene_*.lmdb"):
            match = SCENE_PATTERN.match(path.name)
            if match:
                validate_scene(path, self.cfg, deep=False)
                indexed.append((int(match.group(1)), path))
        indexed.sort(key=lambda item: item[0])
        indices = [index for index, _ in indexed]
        if indices != list(range(len(indices))):
            raise ValueError("成功场景序号存在重复或空洞，请先运行 validate")
        self._repair_checkpoint(len(indexed))
        return [path for _, path in indexed]

    def _repair_checkpoint(self, completed: int) -> None:
        path = self.root / "checkpoint.json"
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        stable = {
            "completed_scene_count": completed,
            "next_scene_index": completed,
            "config_hash": self.config_hash,
            "asset_hash": self.asset_hash,
        }
        if all(current.get(key) == value for key, value in stable.items()):
            return
        checkpoint = {
            **stable,
            "last_success_utc": datetime.now(timezone.utc).isoformat() if completed else None,
        }
        _atomic_json(path, checkpoint, self.cfg)

    def publish(self, record: SceneRecord) -> Path:
        """写入、压缩并原子发布一个已成功场景。"""
        check_scene_record(record)
        name = f"scene_{record.spec.scene_index:06d}_{record.scene_id}.lmdb"
        final = self.root / name
        partial = self.staging / name.replace(".lmdb", ".partial")
        if final.exists() or partial.exists():
            raise FileExistsError(f"场景已存在: {name}")
        self._write_partial(partial, record)
        validate_scene(partial, self.cfg, deep=True)
        self._publish_compacted(partial, final)
        validate_scene(final, self.cfg, deep=True)
        self._repair_checkpoint(record.spec.scene_index + 1)
        return final

    def _write_partial(self, path: Path, record: SceneRecord) -> None:
        path.mkdir(parents=True)
        storage = self.cfg.data_collector.storage
        encoded_frames = [encode_value(frame.to_dict(), storage.compression_level) for frame in record.frames]
        metadata = {
            "scene_id": record.scene_id,
            "spec": record.spec.to_dict(),
            "mjcf_xml": record.mjcf_xml,
            "success_evidence": record.success_evidence,
            "asset_hash": record.asset_hash,
            "config_hash": record.config_hash,
            "config_snapshot": record.config_snapshot,
            "versions": record.versions,
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "frame_count": len(encoded_frames),
            "frame_hashes": [hashlib.sha256(frame).hexdigest() for frame in encoded_frames],
        }
        map_size = storage.map_size_mb * 1024 * 1024
        env = lmdb.open(str(path), map_size=map_size, max_dbs=storage.max_dbs, sync=True, metasync=True, writemap=False)
        try:
            meta_db, frames_db, index_db = _open_databases(env, create=True)
            while True:
                try:
                    with env.begin(write=True) as transaction:
                        transaction.put(b"scene", encode_value(metadata, storage.compression_level), db=meta_db)
                        transaction.put(b"complete", b"1", db=meta_db)
                        transaction.put(b"summary", encode_value(summary, storage.compression_level), db=index_db)
                        for index, frame in enumerate(encoded_frames):
                            key = str(index).zfill(storage.frame_key_width).encode("ascii")
                            transaction.put(key, frame, db=frames_db)
                    break
                except lmdb.MapFullError:
                    env.set_mapsize(int(env.info()["map_size"] * storage.map_growth_factor))
            env.sync(True)
        finally:
            env.close()

    def _publish_compacted(self, partial: Path, final: Path) -> None:
        compact = partial.with_name(partial.name + ".compact")
        _safe_remove(compact, self.staging)
        _copy_compact(partial, compact, self.cfg)
        validate_scene(compact, self.cfg, deep=True)
        compact.rename(final)
        _safe_remove(partial, self.staging)


def validate_dataset(path: str | Path, cfg: AppConfig, deep: bool = False) -> dict[str, Any]:
    """验证数据集所有最终成功场景并返回任务统计。"""
    root = Path(path)
    check_dataset_path(root.resolve())
    reports = [validate_scene(scene, cfg, deep=deep) for scene in sorted(root.glob("scene_*.lmdb"))]
    indices = [report["scene_index"] for report in reports]
    if indices != list(range(len(indices))):
        raise ValueError("数据集场景序号不连续")
    tasks: dict[str, int] = {}
    for report in reports:
        tasks[report["task_type"]] = tasks.get(report["task_type"], 0) + 1
    return {"scene_count": len(reports), "tasks": tasks, "frames": sum(report["frame_count"] for report in reports)}


__all__ = ["DatasetStore", "compact_scene", "config_fingerprint", "decode_value", "encode_value", "validate_dataset", "validate_scene"]
