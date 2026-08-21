"""以按场景SQLite文件持久化在线重放结果，避免持有整场内存缓存。

模块: data/replay_cache/replay_cache.py
依赖: hashlib, json, sqlite3, config, data.data_collector.storage
读取配置: model_data.replay_cache.*, data_collector.render.cameras,
    data_collector.sensors.*, model.tactile_patch
对外接口:
    - ReplayDiskCache
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable

import mujoco

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.data_collector.storage import decode_value, encode_value
from data.replay_cache.checks import check_cache_directory


CACHE_SCHEMA = "1.0.0"


class ReplayDiskCache:
    """管理单个源场景的压缩磁盘缓存，连接随样本访问关闭。"""

    def __init__(self, scene: Path, source_signature: str, cfg: AppConfig):
        self.settings = cfg.model_data.replay_cache
        self.hits = 0
        self.misses = 0
        self.connection: sqlite3.Connection | None = None
        if not self.settings.enabled:
            return
        root = Path(self.settings.directory)
        root = root if root.is_absolute() else PROJECT_ROOT / root
        check_cache_directory(root)
        compatibility = {
            "schema": CACHE_SCHEMA,
            "mujoco": mujoco.__version__,
            "render_backend": os.environ.get("MUJOCO_GL", "default"),
            "render": [asdict(camera) for camera in cfg.data_collector.render.cameras],
            "sensors": asdict(cfg.data_collector.sensors),
            "tactile_patch": cfg.model.tactile_patch,
        }
        fingerprint = hashlib.sha256(json.dumps(
            compatibility, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()[:16]
        source = hashlib.sha256(f"{scene.resolve()}:{source_signature}".encode()).hexdigest()[:16]
        directory = root / fingerprint
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{scene.stem}_{source}.sqlite3"
        self.connection = sqlite3.connect(path, timeout=self.settings.sqlite_timeout_seconds)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS values_cache (key TEXT PRIMARY KEY, payload BLOB NOT NULL)")

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        """读取缓存值；缺失时计算并原子插入。"""
        if self.connection is None:
            return factory()
        row = self.connection.execute("SELECT payload FROM values_cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return decode_value(row[0])
        self.misses += 1
        value = factory()
        payload = encode_value(value, self.settings.compression_level)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO values_cache(key, payload) VALUES (?, ?)", (key, payload),
            )
        return value

    def close(self) -> None:
        """提交并关闭当前样本使用的SQLite连接。"""
        if self.connection is not None:
            self.connection.close()
            self.connection = None


__all__ = ["ReplayDiskCache"]
