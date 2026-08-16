"""重导出 MuJoCo 仿真接口。

模块: data/data_collector/simulation/__init__.py
依赖: data.data_collector.simulation.simulation
读取配置: data_collector.simulation.*, data_collector.controller.gripper_*, data_collector.render.*,
    data_collector.sensors.*
对外接口:
    - EmbodiedSimulator
    - compute_tactile_state(model, data, settings) -> tuple[list, dict]
"""

from data.data_collector.simulation.simulation import EmbodiedSimulator, compute_tactile_state

__all__ = ["EmbodiedSimulator", "compute_tactile_state"]
