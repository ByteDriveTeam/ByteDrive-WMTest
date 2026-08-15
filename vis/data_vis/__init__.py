"""重导出 LMDB 场景可视化接口。

模块: vis/data_vis/__init__.py
依赖: vis.data_vis.data_vis
读取配置: data_vis.*
对外接口:
    - visualize_scene(dataset, scene, cfg, ...) -> dict
"""

from vis.data_vis.data_vis import visualize_scene

__all__ = ["visualize_scene"]

