"""重导出随机场景与 MJCF 构建接口。

模块: data/data_collector/scene/__init__.py
依赖: data.data_collector.scene.scene
读取配置: data_collector.collector.master_seed, data_collector.simulation.*, data_collector.scene.*,
    data_collector.render.cameras, data_collector.sensors.contact_enabled
对外接口:
    - add_virtual_tactile_sites(xml) -> str
    - generate_scene_spec(scene_index, attempt, cfg, task_type=None) -> SceneSpec
    - build_mjcf(spec, cfg) -> str
    - asset_fingerprint() -> str
    - materialize_mjcf(xml) -> str
    - scene_identifier(spec, config_hash) -> str
"""

from data.data_collector.scene.scene import (
    add_virtual_tactile_sites,
    asset_fingerprint,
    build_mjcf,
    generate_scene_spec,
    materialize_mjcf,
    scene_identifier,
)

__all__ = [
    "add_virtual_tactile_sites", "asset_fingerprint", "build_mjcf", "generate_scene_spec",
    "materialize_mjcf", "scene_identifier",
]
