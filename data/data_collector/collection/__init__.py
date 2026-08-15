"""重导出采集与断点续采接口。

模块: data/data_collector/collection/__init__.py
依赖: data.data_collector.collection.collection
读取配置: data_collector.collector.*, data_collector.tasks.*, data_collector.render.*,
    data_collector.sensors.*, data_collector.storage.*
对外接口:
    - collect_scenes(cfg, task_types=None) -> dict
    - resume_collection(cfg, task_types=None) -> dict
"""

from data.data_collector.collection.collection import collect_scenes, resume_collection

__all__ = ["collect_scenes", "resume_collection"]

