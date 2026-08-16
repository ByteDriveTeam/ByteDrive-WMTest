"""定义并校验项目集中配置。

模块: config/schema.py
依赖: dataclasses, math, typing
读取配置: data_collector.*, data_vis.*
对外接口:
    - ConfigError
    - AppConfig
    - DataVisSettings
    - build_config(data) -> AppConfig
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import math
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints


class ConfigError(ValueError):
    """表示配置结构或取值不合法。"""


@dataclass(frozen=True)
class CollectorSettings:
    scene_count: int
    master_seed: int
    resume: bool
    max_attempts_per_scene: int
    output: str


@dataclass(frozen=True)
class SimulationSettings:
    timestep: float
    control_substeps: int
    settle_steps: int
    gravity: list[float]
    solver_iterations: int
    max_frames: int


@dataclass(frozen=True)
class ControllerSettings:
    ik_damping: float
    ik_gain: float
    max_joint_step: float
    position_tolerance: float
    transport_position_tolerance: float
    transport_waypoints: int
    transport_clearance: float
    transport_max_height: float
    orientation_tolerance: float
    orientation_weight: float
    phase_frames: int
    wait_frames: int
    gripper_frames: int
    settle_control_frames: int
    slide_initial_speed: float
    slide_direction: list[float]
    gripper_open: float
    gripper_closed: float
    grasp_distance: float
    grasp_height_offset: float
    grasp_retry_height_offsets: list[float]
    grasp_position_tolerance: float
    grasp_lateral_tolerance: float
    grasp_axial_tolerance: float
    grasp_min_normal_force: float
    grasp_validation_frames: int
    grasp_lift_min_height: float
    grasp_hold_distance: float
    grasp_loss_frames: int
    held_max_joint_step: float
    neutral_position: list[float]
    joint_home_tolerance: float
    lift_height: float
    approach_height: float
    joint_kp: float
    joint_damping: float
    finger_kp: float
    finger_damping: float


@dataclass(frozen=True)
class SceneSettings:
    table_height: float
    table_size: list[float]
    object_xy_min: list[float]
    object_xy_max: list[float]
    object_count_min: int
    object_count_max: int
    object_size_min: float
    object_size_max: float
    object_mass_min: float
    object_mass_max: float
    object_friction_min: float
    object_friction_max: float
    minimum_object_spacing: float
    placement_attempts: int
    object_shapes: list[str]
    slide_target_shapes: list[str]
    target_size: list[float]
    slope_size: list[float]
    table_rgba: list[float]
    floor_rgba: list[float]
    floor_plane_size: float
    table_friction: list[float]
    slope_friction: list[float]
    bin_friction: list[float]
    slope_rgba: list[float]
    target_rgba: list[float]
    bin_wall_height: float
    bin_wall_thickness: float
    bin_wall_rgba: list[float]
    target_positions: dict[str, list[float]]
    slope_angle_min: float
    slope_angle_max: float
    palette: dict[str, list[float]]


@dataclass(frozen=True)
class TaskSettings:
    weights: dict[str, float]
    stable_frames: int
    region_tolerance: float
    velocity_tolerance: float
    slide_distance: float


@dataclass(frozen=True)
class CameraSettings:
    name: str
    parent: str
    width: int
    height: int
    position: list[float]
    roll: float
    pitch: float
    yaw: float
    fov_x: float
    fov_y: float
    near: float
    far: float
    modalities: list[str]


@dataclass(frozen=True)
class RenderSettings:
    enabled: bool
    viewer: bool
    cameras: list[CameraSettings]


@dataclass(frozen=True)
class SensorSettings:
    contact_enabled: bool
    tactile_resolution: list[int]
    tactile_extent: list[float]
    tactile_sigma_pixels: float
    tactile_force_clip: float


@dataclass(frozen=True)
class StorageSettings:
    map_size_mb: int
    map_growth_factor: float
    compression_level: int
    max_dbs: int
    frame_key_width: int
    atomic_replace_attempts: int
    atomic_replace_retry_seconds: float


@dataclass(frozen=True)
class DataCollectorConfig:
    collector: CollectorSettings
    simulation: SimulationSettings
    controller: ControllerSettings
    scene: SceneSettings
    tasks: TaskSettings
    render: RenderSettings
    sensors: SensorSettings
    storage: StorageSettings


@dataclass(frozen=True)
class DataVisSettings:
    output: str
    camera: str
    modality: str
    start_frame: int
    end_frame: int
    stride: int
    max_frames: int
    gif_enabled: bool
    gif_fps: int
    panel_width: int
    panel_min_height: int
    padding: int
    text_line_height: int
    tactile_map_size: int
    tactile_force_max: float
    tactile_force_gamma: float
    joint_display_range: float
    depth_percentiles: list[float]
    background_rgb: list[int]
    text_rgb: list[int]
    force_replay: bool


@dataclass(frozen=True)
class AppConfig:
    data_collector: DataCollectorConfig
    data_vis: DataVisSettings


def _convert(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    if is_dataclass(annotation):
        return _construct(annotation, value)
    if origin is list:
        item_type = get_args(annotation)[0]
        return [_convert(item_type, item) for item in value]
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {_convert(key_type, key): _convert(value_type, item) for key, item in value.items()}
    if origin in (UnionType,):
        return value
    return value


def _construct(cls: type[Any], data: dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__} 必须是映射")
    hints = get_type_hints(cls)
    expected = {field.name for field in fields(cls)}
    missing = expected - data.keys()
    extra = data.keys() - expected
    if missing or extra:
        raise ConfigError(f"{cls.__name__} 缺少 {sorted(missing)}，多出 {sorted(extra)}")
    return cls(**{name: _convert(hints[name], data[name]) for name in expected})


def _validate(cfg: AppConfig) -> None:
    dc = cfg.data_collector
    # 校验对象: collector.scene_count —— 成功场景总数必须为正。
    if not 0 < dc.collector.scene_count <= 999999:
        raise ConfigError("collector.scene_count 必须位于 [1, 999999]")
    # 校验对象: collector.max_attempts_per_scene —— 每场景必须允许至少一次候选。
    if dc.collector.max_attempts_per_scene <= 0:
        raise ConfigError("collector.max_attempts_per_scene 必须 > 0")
    # 校验对象: simulation 时间参数 —— 固定步长和控制子步必须为正。
    if dc.simulation.timestep <= 0 or dc.simulation.control_substeps <= 0:
        raise ConfigError("simulation.timestep 与 control_substeps 必须 > 0")
    # 校验对象: controller —— IK、阶段和抓持参数必须为正，安全中立点必须为三维。
    controller_positive_values = (
        dc.controller.ik_damping, dc.controller.ik_gain, dc.controller.max_joint_step,
        dc.controller.position_tolerance, dc.controller.transport_position_tolerance,
        dc.controller.transport_clearance, dc.controller.transport_max_height,
        dc.controller.grasp_distance, dc.controller.grasp_position_tolerance,
        dc.controller.grasp_lateral_tolerance, dc.controller.grasp_axial_tolerance, dc.controller.joint_home_tolerance,
        dc.controller.grasp_min_normal_force, dc.controller.grasp_lift_min_height,
        dc.controller.grasp_hold_distance, dc.controller.held_max_joint_step,
        dc.controller.joint_kp, dc.controller.joint_damping, dc.controller.finger_kp, dc.controller.finger_damping,
    )
    if any(value <= 0 for value in controller_positive_values) or dc.controller.grasp_height_offset < 0 or len(dc.controller.neutral_position) != 3:
        raise ConfigError("controller 的 IK、抓持或中立点参数不合法")
    if len(dc.controller.slide_direction) != 3 or sum(value * value for value in dc.controller.slide_direction) <= 0:
        raise ConfigError("controller.slide_direction 必须是非零三维向量")
    if dc.controller.transport_waypoints <= 0 or dc.controller.grasp_validation_frames <= 0 or dc.controller.grasp_loss_frames <= 0:
        raise ConfigError("controller 的运输航点数、抓取验证帧数与掉落判定帧数必须 > 0")
    # 校验对象: controller.grasp_retry_height_offsets —— 至少保留一个有限的物理重抓高度。
    if not dc.controller.grasp_retry_height_offsets or not all(math.isfinite(value) for value in dc.controller.grasp_retry_height_offsets):
        raise ConfigError("controller.grasp_retry_height_offsets 必须包含有限数值")
    # 校验对象: scene 尺寸和质量范围 —— 下界不能超过上界。
    ranges = (
        (dc.scene.object_size_min, dc.scene.object_size_max, "object_size"),
        (dc.scene.object_mass_min, dc.scene.object_mass_max, "object_mass"),
        (dc.scene.object_friction_min, dc.scene.object_friction_max, "object_friction"),
        (dc.scene.slope_angle_min, dc.scene.slope_angle_max, "slope_angle"),
    )
    if any(low <= 0 or low > high for low, high, _ in ranges):
        raise ConfigError("scene 的尺寸、质量、摩擦或斜面范围不合法")
    # 校验对象: scene 几何列表 —— 维度和候选次数必须满足场景生成需要。
    if dc.scene.placement_attempts <= 0 or not dc.scene.object_shapes:
        raise ConfigError("scene.placement_attempts 与 object_shapes 不合法")
    # 校验对象: scene.slide_target_shapes —— 斜面目标形状必须来自全局物体形状词表。
    if not dc.scene.slide_target_shapes or not set(dc.scene.slide_target_shapes).issubset(dc.scene.object_shapes):
        raise ConfigError("scene.slide_target_shapes 必须是 object_shapes 的非空子集")
    # 校验对象: scene.object_count_* —— 随机数量区间必须为正向闭区间。
    if dc.scene.object_count_min <= 0 or dc.scene.object_count_min > dc.scene.object_count_max:
        raise ConfigError("scene.object_count_min/max 必须构成正整数闭区间")
    if dc.scene.object_count_max > len(dc.scene.palette):
        raise ConfigError("scene.palette 颜色数不得少于 object_count_max，场景内颜色不可重复")
    multi_object_tasks = {"SORT", "STACK", "SEQUENTIAL_REARRANGE"}
    if dc.scene.object_count_max < 2 and any(dc.tasks.weights.get(name, 0.0) > 0 for name in multi_object_tasks):
        raise ConfigError("启用多物体任务时 scene.object_count_max 必须 >= 2")
    if len(dc.scene.target_size) != 3 or len(dc.scene.slope_size) != 3:
        raise ConfigError("scene.target_size 与 slope_size 必须是三维")
    if dc.scene.bin_wall_height <= 0 or dc.scene.bin_wall_thickness <= 0:
        raise ConfigError("scene 的容器壁参数必须 > 0")
    # 校验对象: tasks.weights —— 至少启用一个正权重任务。
    if not dc.tasks.weights or any(weight < 0 for weight in dc.tasks.weights.values()) or sum(dc.tasks.weights.values()) <= 0:
        raise ConfigError("tasks.weights 必须包含至少一个正权重任务")
    # 校验对象: render.cameras —— 相机名唯一且内外参完整。
    names = [camera.name for camera in dc.render.cameras]
    if len(names) != len(set(names)):
        raise ConfigError("render.cameras 的 name 必须唯一")
    allowed_modalities = {"rgb", "depth", "segmentation"}
    for camera in dc.render.cameras:
        if camera.parent not in {"world", "base", "end_effector"}:
            raise ConfigError(f"相机 {camera.name} 的 parent 不合法")
        angles = (camera.roll, camera.pitch, camera.yaw)
        if camera.width <= 0 or camera.height <= 0 or not all(math.isfinite(value) for value in angles):
            raise ConfigError(f"相机 {camera.name} 的尺寸或 YPR 角度不合法")
        if not 0 < camera.fov_x < 180 or not 0 < camera.fov_y < 180:
            raise ConfigError(f"相机 {camera.name} 的尺寸或 FOV 不合法")
        if len(camera.position) != 3:
            raise ConfigError(f"相机 {camera.name} 的外参维度不合法")
        if not set(camera.modalities).issubset(allowed_modalities):
            raise ConfigError(f"相机 {camera.name} 包含未知视觉模态")
    # 校验对象: sensors.tactile_* —— 双指触觉网格固定为 32×32，力图参数必须为正。
    if dc.sensors.tactile_resolution != [32, 32]:
        raise ConfigError("sensors.tactile_resolution 必须为 [32, 32]")
    if len(dc.sensors.tactile_extent) != 2 or any(value <= 0 for value in dc.sensors.tactile_extent):
        raise ConfigError("sensors.tactile_extent 必须包含两个正值")
    if dc.sensors.tactile_sigma_pixels <= 0 or dc.sensors.tactile_force_clip <= 0:
        raise ConfigError("触觉核宽和力裁剪值必须 > 0")
    # 校验对象: storage —— LMDB 和压缩参数必须可用。
    if dc.storage.map_size_mb <= 0 or dc.storage.map_growth_factor <= 1 or not 1 <= dc.storage.compression_level <= 22:
        raise ConfigError("storage 参数不合法")
    # 校验对象: storage.atomic_replace_* —— 原子替换必须至少尝试一次，等待时间不得为负。
    if dc.storage.atomic_replace_attempts <= 0 or dc.storage.atomic_replace_retry_seconds < 0:
        raise ConfigError("storage.atomic_replace_attempts 必须 > 0，retry_seconds 必须 >= 0")
    # 校验对象: data_vis 帧范围与输出尺寸 —— 必须能形成有限、正向的可视化序列。
    vis = cfg.data_vis
    if vis.start_frame < 0 or vis.end_frame < -1 or vis.stride <= 0 or vis.max_frames <= 0:
        raise ConfigError("data_vis 的帧范围、步长或最大帧数不合法")
    if vis.gif_fps <= 0 or vis.panel_width <= 0 or vis.panel_min_height <= 0 or vis.padding < 0:
        raise ConfigError("data_vis 的 GIF 或面板尺寸参数不合法")
    if vis.text_line_height <= 0 or vis.tactile_map_size < 32:
        raise ConfigError("data_vis 的文字行高或触觉图尺寸不合法")
    if vis.panel_width < 64 + 3 * vis.padding:
        raise ConfigError("data_vis.panel_width 无法容纳两块 32×32 触觉图")
    if vis.tactile_force_max <= 0 or vis.tactile_force_gamma <= 0 or vis.joint_display_range <= 0:
        raise ConfigError("data_vis 的触觉或关节显示范围必须 > 0")
    if vis.modality not in allowed_modalities:
        raise ConfigError(f"data_vis.modality 不合法: {vis.modality}")
    if vis.camera not in names:
        raise ConfigError(f"data_vis.camera 未在 render.cameras 中定义: {vis.camera}")
    if len(vis.depth_percentiles) != 2 or not 0 <= vis.depth_percentiles[0] < vis.depth_percentiles[1] <= 100:
        raise ConfigError("data_vis.depth_percentiles 必须是递增的两个百分位")
    if any(len(color) != 3 or any(channel < 0 or channel > 255 for channel in color) for color in (vis.background_rgb, vis.text_rgb)):
        raise ConfigError("data_vis 的 RGB 颜色必须包含三个 [0, 255] 整数")


def build_config(data: dict[str, Any]) -> AppConfig:
    """从原始映射构造不可变配置并完成一次性校验。"""
    cfg = _construct(AppConfig, data)
    _validate(cfg)
    return cfg


__all__ = ["AppConfig", "ConfigError", "DataVisSettings", "build_config"]
