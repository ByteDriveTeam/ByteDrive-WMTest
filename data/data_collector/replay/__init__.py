"""重导出二次渲染接口。

模块: data/data_collector/replay/__init__.py
依赖: data.data_collector.replay.replay
读取配置: data_collector.render.cameras, data_collector.storage.max_dbs
对外接口:
    - rerender_scene(dataset, scene, frames, camera, output, cfg) -> dict
"""

from data.data_collector.replay.replay import rerender_scene

__all__ = ["rerender_scene"]

