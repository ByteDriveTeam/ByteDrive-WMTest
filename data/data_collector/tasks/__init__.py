"""重导出任务生成接口。

模块: data/data_collector/tasks/__init__.py
依赖: data.data_collector.tasks.tasks
读取配置: data_collector.tasks.weights, data_collector.scene.object_count_min,
    data_collector.scene.object_count_max
对外接口:
    - choose_task_type(rng, cfg) -> str
    - object_count_for_task(task_type, rng, cfg) -> int
    - build_task(task_type, objects) -> TaskSpec
"""

from data.data_collector.tasks.tasks import build_task, choose_task_type, object_count_for_task

__all__ = ["build_task", "choose_task_type", "object_count_for_task"]
