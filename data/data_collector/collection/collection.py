"""编排成功场景生成、脚本执行、LMDB 发布和断点续采。

模块: data/data_collector/collection/collection.py
依赖: config, mujoco, numpy, lmdb, data.data_collector.controller, data.data_collector.scene,
    data.data_collector.simulation, data.data_collector.storage
读取配置: data_collector.collector.*, data_collector.tasks.*, data_collector.render.*,
    data_collector.sensors.*, data_collector.storage.*
对外接口:
    - collect_scenes(cfg, task_types=None) -> dict
    - resume_collection(cfg, task_types=None) -> dict
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import platform
import sys
from typing import Iterable

import lmdb
import mujoco
import numpy as np

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.data_collector.collection.checks.collection_checks import check_collection_inputs
from data.data_collector.controller import ScriptedExpert
from data.data_collector.records import SceneRecord
from data.data_collector.scene import asset_fingerprint, build_mjcf, generate_scene_spec, scene_identifier
from data.data_collector.simulation import EmbodiedSimulator
from data.data_collector.storage import DatasetStore, config_fingerprint


def _output_path(cfg: AppConfig) -> Path:
    path = Path(cfg.data_collector.collector.output)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "lmdb": lmdb.__version__,
    }


def collect_scenes(cfg: AppConfig, task_types: Iterable[str] | None = None) -> dict[str, object]:
    """采集到目标成功场景总数，并从最终 LMDB 连续数量处续采。"""
    task_list = list(task_types or [])
    check_collection_inputs(cfg, task_list)
    asset_hash = asset_fingerprint()
    store = DatasetStore(_output_path(cfg), cfg, asset_hash)
    store.initialize()
    completed = store.scan_completed()
    if completed and not cfg.data_collector.collector.resume:
        raise ValueError("数据集已有成功场景；继续采集必须启用 resume")
    target = cfg.data_collector.collector.scene_count
    if len(completed) >= target:
        return {"dataset": str(store.root), "completed_scene_count": len(completed), "new_scene_count": 0}
    published = 0
    config_hash = config_fingerprint(cfg)
    for scene_index in range(len(completed), target):
        success = False
        forced_task = task_list[scene_index % len(task_list)] if task_list else None
        for attempt in range(cfg.data_collector.collector.max_attempts_per_scene):
            simulator: EmbodiedSimulator | None = None
            evidence = None
            try:
                spec = generate_scene_spec(scene_index, attempt, cfg, forced_task)
                mjcf_xml = build_mjcf(spec, cfg)
                simulator = EmbodiedSimulator(spec, mjcf_xml, cfg)
                evidence = ScriptedExpert(simulator, spec, cfg).run()
                if evidence is None:
                    print(f"场景 {scene_index:06d} 候选 {attempt} 失败，未持久化", flush=True)
                    continue
                scene_id = scene_identifier(spec, config_hash)
                record = SceneRecord(
                    scene_id=scene_id,
                    spec=spec,
                    mjcf_xml=mjcf_xml,
                    frames=simulator.frames,
                    success_evidence=evidence,
                    asset_hash=asset_hash,
                    config_hash=config_hash,
                    config_snapshot=asdict(cfg),
                    versions=_versions(),
                )
                final_path = store.publish(record)
                published += 1
                success = True
                print(f"已发布成功场景 {scene_index:06d}: {final_path.name}", flush=True)
                break
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                if evidence is not None:
                    raise RuntimeError(f"成功场景 {scene_index:06d} 发布失败") from error
                print(f"场景 {scene_index:06d} 候选 {attempt} 异常，未持久化: {error}", file=sys.stderr, flush=True)
            finally:
                if simulator is not None:
                    simulator.close()
        if not success:
            raise RuntimeError(f"场景 {scene_index:06d} 在最大候选次数内未获得成功轨迹")
    return {"dataset": str(store.root), "completed_scene_count": target, "new_scene_count": published}


def resume_collection(cfg: AppConfig, task_types: Iterable[str] | None = None) -> dict[str, object]:
    """以最终成功场景数为断点继续采集。"""
    return collect_scenes(cfg, task_types)


__all__ = ["collect_scenes", "resume_collection"]
