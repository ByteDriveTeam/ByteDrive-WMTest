"""重导出脚本专家控制接口。

模块: data/data_collector/controller/__init__.py
依赖: data.data_collector.controller.controller
读取配置: data_collector.controller.*, data_collector.scene.slope_size, data_collector.tasks.*
对外接口:
    - ScriptedExpert
"""

from data.data_collector.controller.controller import ScriptedExpert

__all__ = ["ScriptedExpert"]

