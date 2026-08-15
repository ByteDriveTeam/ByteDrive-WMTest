"""重导出单臂具身数据采集系统公共 API。

模块: data/data_collector/__init__.py
依赖: data.data_collector.collection, data.data_collector.records,
    data.data_collector.replay, data.data_collector.storage
读取配置: 无
对外接口:
    - collect_scenes(cfg, task_types=None) -> dict
    - resume_collection(cfg, task_types=None) -> dict
    - rerender_scene(dataset, scene, frames, camera, output, cfg) -> dict
    - validate_scene(path, cfg, deep=False) -> dict
    - validate_dataset(path, cfg, deep=False) -> dict
    - compact_scene(path, cfg) -> dict
    - ActionStep
    - TaskSpec
    - FrameRecord
    - SceneRecord
"""

from data.data_collector.collection import collect_scenes, resume_collection
from data.data_collector.records import ActionStep, FrameRecord, SceneRecord, TaskSpec
from data.data_collector.replay import rerender_scene
from data.data_collector.storage import compact_scene, validate_dataset, validate_scene

__all__ = [
    "ActionStep", "FrameRecord", "SceneRecord", "TaskSpec", "collect_scenes", "compact_scene",
    "rerender_scene", "resume_collection", "validate_dataset", "validate_scene",
]

