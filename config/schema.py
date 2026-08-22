"""定义并校验项目集中配置。

模块: config/schema.py
依赖: dataclasses, math, typing
读取配置: data_collector.*, data_vis.*, model_data.*, model.*, loss.*, model_vis.*,
    validation_vis.*, training.*
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
    force_tactile_replay: bool


@dataclass(frozen=True)
class ReplayCacheSettings:
    enabled: bool
    directory: str
    linux_render_backend: str
    linux_egl_device_id: int
    compression_level: int
    sqlite_timeout_seconds: float
    stats_log_interval_scenes: int


@dataclass(frozen=True)
class ModelDataSettings:
    dataset: str
    statistics: str
    history_seconds: float
    future_seconds: float
    rgb_hz: int
    sensor_hz: int
    language_length: int
    language_vocabulary: list[str]
    image_mean: list[float]
    image_std: list[float]
    train_fraction: float
    validation_fraction: float
    window_stride_seconds: float
    normalization_epsilon: float
    coordinate_fallback_bounds: list[list[float]]
    replay_cache: ReplayCacheSettings


@dataclass(frozen=True)
class ModelSettings:
    width: int
    heads: int
    backbone_layers: int
    predictor_layers: int
    ffn_width: int
    lora_rank: int
    register_tokens: int
    image_patch: int
    tactile_patch: int
    token_init_std: float
    norm_epsilon: float
    position_hidden: int
    time_frequencies: int
    petr_depth_samples: int
    overview_depth_range: list[float]
    wrist_depth_range: list[float]
    mask_ratio: float
    task_priority_sample_probability: float
    temporal_gradient_weight: float
    mask_group_probability: float
    rgb_relief_when_sensor_masked: float
    ema_decay: float
    flow_hidden: int
    phase_names: list[str]


@dataclass(frozen=True)
class LossSettings:
    velocity_weight: float
    endpoint_weight: float
    reconstruction_weight: float
    phase_weight: float
    visreg_weight: float
    visreg_regularization_mix: float
    visreg_num_projections: int
    visreg_scale_weight: float
    visreg_shape_weight: float
    visreg_center_weight: float
    visreg_epsilon: float
    masked_reconstruction_weight: float
    visible_reconstruction_start_weight: float
    visible_reconstruction_weight: float
    visible_reconstruction_warmup_fraction: float
    action_weight: float
    tactile_weight: float
    action_component_weights: list[float]
    tactile_component_weights: list[float]
    velocity_layer_weights: list[float]
    endpoint_warmup_fraction: float
    endpoint_start_weight: float
    endpoint_end_weight: float
    phase_label_smoothing: float


@dataclass(frozen=True)
class ModelVisSettings:
    output: str
    device: str
    split: str
    sample_index: int
    flow_noise_seed: int
    canvas_size: list[int]
    feature_panel_height: int
    feature_clip_percentile: float
    pca_clip_percentile: float
    pca_band_height: int
    background_rgb: list[int]
    panel_rgb: list[int]
    text_rgb: list[int]
    prediction_rgb: list[int]
    target_rgb: list[int]
    line_width: int


@dataclass(frozen=True)
class ValidationVisSettings:
    enabled: bool
    output: str
    sample_index: int
    fail_on_error: bool
    data_canvas_size: list[int]
    history_canvas_size: list[int]
    rgb_columns: int
    tactile_clip_percentile: float
    state_clip_percentile: float
    closed_loop_enabled: bool
    closed_loop_task: str
    closed_loop_scene_index: int
    closed_loop_attempt: int
    closed_loop_max_control_frames: int
    closed_loop_replan_action_steps: int
    closed_loop_video_fps: int
    closed_loop_video_stride: int


@dataclass(frozen=True)
class TrainingSettings:
    output: str
    device: str
    epochs: int
    batch_size: int
    gradient_accumulation: int
    num_workers: int
    dataloader_prefetch_factor: int
    device_prefetch: bool
    log_interval_steps: int
    constantization_monitor_enabled: bool
    constantization_relative_std_threshold: float
    constantization_patience_intervals: int
    learning_rate: float
    minimum_learning_rate: float
    weight_decay: float
    adam_betas: list[float]
    warmup_epochs: int
    teacher_forcing_fraction: float
    gradient_clip: float
    seed: int
    checkpoint_interval: int
    validation_interval: int


@dataclass(frozen=True)
class PostTrainingSettings:
    output: str
    epochs: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_epochs: int
    checkpoint_interval: int
    validation_interval: int


@dataclass(frozen=True)
class AppConfig:
    data_collector: DataCollectorConfig
    data_vis: DataVisSettings
    model_data: ModelDataSettings
    model: ModelSettings
    loss: LossSettings
    model_vis: ModelVisSettings
    validation_vis: ValidationVisSettings
    training: TrainingSettings
    post_training: PostTrainingSettings


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
    # 校验对象: model_data 采样与划分——必须能生成有限且互斥的训练窗口。
    md = cfg.model_data
    if md.history_seconds <= 0 or md.future_seconds <= 0 or md.rgb_hz <= 0 or md.sensor_hz <= 0:
        raise ConfigError("model_data 的时长与采样率必须 > 0")
    sample_counts = (
        round(md.history_seconds * md.rgb_hz), round(md.history_seconds * md.sensor_hz),
        round(md.future_seconds * md.sensor_hz),
    )
    if sample_counts != (10, 50, 50):
        raise ConfigError("当前固定序列设计要求 RGB/历史传感器/未来样本数为 10/50/50")
    source_hz = 1.0 / (cfg.data_collector.simulation.timestep * cfg.data_collector.simulation.control_substeps)
    if not source_hz.is_integer() or int(source_hz) % md.rgb_hz or int(source_hz) % md.sensor_hz:
        raise ConfigError("model_data 的采样率必须整除原始控制频率，保证精确索引")
    if md.sensor_hz % md.rgb_hz or md.language_length <= 0:
        raise ConfigError("model_data.sensor_hz 必须是 rgb_hz 的整数倍，且 language_length 必须 > 0")
    if len(md.language_vocabulary) != len(set(md.language_vocabulary)) or not md.language_vocabulary:
        raise ConfigError("model_data.language_vocabulary 必须是非空无重复闭集词表")
    if len(md.image_mean) != 3 or len(md.image_std) != 3 or any(value <= 0 for value in md.image_std):
        raise ConfigError("model_data.image_mean/image_std 必须是三个通道且标准差 > 0")
    if not 0 < md.train_fraction < 1 or not 0 <= md.validation_fraction < 1 or md.train_fraction + md.validation_fraction >= 1:
        raise ConfigError("model_data 的训练/验证划分比例不合法")
    if md.window_stride_seconds <= 0 or md.normalization_epsilon <= 0:
        raise ConfigError("model_data.window_stride_seconds 和归一化 epsilon 必须 > 0")
    if not (md.window_stride_seconds * source_hz).is_integer():
        raise ConfigError("model_data.window_stride_seconds 必须精确对齐原始控制帧")
    if len(md.coordinate_fallback_bounds) != 3 or any(len(axis) != 2 or axis[0] >= axis[1] for axis in md.coordinate_fallback_bounds):
        raise ConfigError("model_data.coordinate_fallback_bounds 必须是三个递增轴区间")
    cache = md.replay_cache
    if not cache.directory or not 1 <= cache.compression_level <= 22 or cache.sqlite_timeout_seconds <= 0 or cache.stats_log_interval_scenes <= 0:
        raise ConfigError("model_data.replay_cache 的目录、压缩、超时和日志间隔不合法")
    if cache.linux_render_backend not in {"egl", "glfw", "osmesa"}:
        raise ConfigError("model_data.replay_cache.linux_render_backend 必须是 egl、glfw 或 osmesa")
    if cache.linux_egl_device_id < 0:
        raise ConfigError("model_data.replay_cache.linux_egl_device_id 必须为非负数")
    # 校验对象: model 结构——头维度和深度区间必须与设计一致。
    model = cfg.model
    if model.width % model.heads or min(model.width, model.heads, model.backbone_layers, model.predictor_layers) <= 0:
        raise ConfigError("model.width 必须能被 heads 整除，且结构尺寸必须 > 0")
    if model.register_tokens != 2 or model.image_patch != 16 or model.tactile_patch != 8:
        raise ConfigError("RegisterToken 数量必须为 2，图像/触觉 Patch 必须为 16/8")
    if model.lora_rank <= 0 or model.time_frequencies <= 0 or model.petr_depth_samples <= 0 or min(model.token_init_std, model.norm_epsilon) <= 0:
        raise ConfigError("model 的 LoRA、时间频率和 PETR 采样数必须 > 0")
    if any(len(bounds) != 2 or bounds[0] <= 0 or bounds[0] >= bounds[1] for bounds in (model.overview_depth_range, model.wrist_depth_range)):
        raise ConfigError("model 的 PETR 深度范围必须是正向二元区间")
    probabilities = (
        model.mask_ratio, model.task_priority_sample_probability, model.mask_group_probability,
        model.rgb_relief_when_sensor_masked,
    )
    if any(not 0 <= value <= 1 for value in probabilities) or model.mask_ratio == 0 or model.temporal_gradient_weight < 0:
        raise ConfigError("model 的掩码率或时间梯度权重不合法")
    if not 0 < model.ema_decay < 1 or len(model.phase_names) != len(set(model.phase_names)) or not model.phase_names:
        raise ConfigError("model.ema_decay 或 phase_names 不合法")
    # 校验对象: loss 权重与调度——允许关闭单项，但每个归约层级必须保留至少一个正权重。
    loss = cfg.loss
    total_weights = (
        loss.velocity_weight, loss.endpoint_weight, loss.reconstruction_weight,
        loss.phase_weight, loss.visreg_weight,
    )
    reconstruction_weights = (
        loss.masked_reconstruction_weight, loss.visible_reconstruction_start_weight,
        loss.visible_reconstruction_weight,
    )
    behavior_weights = (loss.action_weight, loss.tactile_weight)
    visreg_weights = (loss.visreg_scale_weight, loss.visreg_shape_weight, loss.visreg_center_weight)
    all_weights = (*total_weights, *reconstruction_weights, *behavior_weights, *visreg_weights,
                   *loss.action_component_weights, *loss.tactile_component_weights,
                   *loss.velocity_layer_weights, loss.endpoint_start_weight, loss.endpoint_end_weight)
    if any(weight < 0 or not math.isfinite(weight) for weight in all_weights):
        raise ConfigError("loss 的所有权重必须是有限非负数")
    behavior_enabled = loss.velocity_weight > 0 or loss.endpoint_weight > 0
    if sum(total_weights) <= 0:
        raise ConfigError("loss 的总项权重不能全为0")
    if (
        loss.visreg_num_projections <= 0
        or loss.visreg_epsilon <= 0
        or not math.isfinite(loss.visreg_epsilon)
    ):
        raise ConfigError("VISReg的投影数必须为正，epsilon必须为有限正数")
    if not math.isfinite(loss.visreg_regularization_mix) or not 0 <= loss.visreg_regularization_mix <= 1:
        raise ConfigError("loss.visreg_regularization_mix必须在[0,1]内")
    if loss.visreg_weight > 0 and loss.visreg_regularization_mix > 0 and sum(visreg_weights) <= 0:
        raise ConfigError("启用VISReg时至少一个子项权重必须为正")
    if loss.reconstruction_weight > 0 and sum(reconstruction_weights) <= 0:
        raise ConfigError("启用重建总项时，loss 的重建权重不能全为0")
    if behavior_enabled and sum(behavior_weights) <= 0:
        raise ConfigError("启用行为总项时，loss 的动作/触觉权重不能全为0")
    if len(loss.action_component_weights) != 3 or behavior_enabled and loss.action_weight > 0 and sum(loss.action_component_weights) <= 0:
        raise ConfigError("loss.action_component_weights 必须是3个，启用动作项时须有正权重")
    if len(loss.tactile_component_weights) != 3 or behavior_enabled and loss.tactile_weight > 0 and sum(loss.tactile_component_weights) <= 0:
        raise ConfigError("loss.tactile_component_weights 必须是3个，启用触觉项时须有正权重")
    if len(loss.velocity_layer_weights) != model.backbone_layers or loss.velocity_weight > 0 and sum(loss.velocity_layer_weights) <= 0:
        raise ConfigError("loss.velocity_layer_weights 必须与骨干层数一致，启用速度项时须有正权重")
    if not 0 <= loss.endpoint_warmup_fraction <= 1 or not 0 <= loss.visible_reconstruction_warmup_fraction <= 1 or not 0 <= loss.phase_label_smoothing < 1:
        raise ConfigError("loss 的末端/可见重建预热比例或阶段标签平滑不合法")
    # 校验对象: model_vis 输入与画布——只允许固定划分和项目内输出所需的有限尺寸。
    model_vis = cfg.model_vis
    colors = (model_vis.background_rgb, model_vis.panel_rgb, model_vis.text_rgb, model_vis.prediction_rgb, model_vis.target_rgb)
    if not model_vis.output or model_vis.device not in {"cpu", "cuda"} or model_vis.split not in {"train", "validation", "test"} or model_vis.sample_index < 0:
        raise ConfigError("model_vis 的输出、设备、划分或样本索引不合法")
    if len(model_vis.canvas_size) != 2 or min(model_vis.canvas_size) <= 0 or not 0 < model_vis.feature_panel_height < model_vis.canvas_size[1]:
        raise ConfigError("model_vis 的画布或特征面板尺寸不合法")
    if not 0 < model_vis.feature_clip_percentile <= 100 or not 0 < model_vis.pca_clip_percentile <= 100:
        raise ConfigError("model_vis 的特征或PCA截断分位不合法")
    if model_vis.pca_band_height <= 0 or model_vis.pca_band_height >= model_vis.feature_panel_height or model_vis.line_width <= 0:
        raise ConfigError("model_vis 的PCA条带高度或线宽不合法")
    if any(len(color) != 3 or any(not 0 <= channel <= 255 for channel in color) for color in colors):
        raise ConfigError("model_vis 的RGB颜色必须是三个[0,255]整数")
    # 校验对象: validation_vis 输出与画布——每次验证的三类派生图必须可稳定排版。
    validation_vis = cfg.validation_vis
    canvas_sizes = (validation_vis.data_canvas_size, validation_vis.history_canvas_size)
    if not validation_vis.output or validation_vis.sample_index < 0 or validation_vis.rgb_columns <= 0:
        raise ConfigError("validation_vis 的输出、样本索引或RGB列数不合法")
    if any(len(size) != 2 or size[0] < 800 or size[1] < 900 for size in canvas_sizes):
        raise ConfigError("validation_vis 的画布宽必须>=800且高必须>=900")
    if not 0 < validation_vis.tactile_clip_percentile <= 100 or not 0 < validation_vis.state_clip_percentile <= 100:
        raise ConfigError("validation_vis 的触觉或状态截断分位不合法")
    closed_loop_positive = (
        validation_vis.closed_loop_max_control_frames,
        validation_vis.closed_loop_replan_action_steps,
        validation_vis.closed_loop_video_fps,
        validation_vis.closed_loop_video_stride,
    )
    if validation_vis.closed_loop_task != "PICK_PLACE":
        raise ConfigError("validation_vis.closed_loop_task 当前固定为 PICK_PLACE")
    if validation_vis.closed_loop_scene_index < 0 or validation_vis.closed_loop_attempt < 0:
        raise ConfigError("闭环验证场景序号与候选序号必须为非负数")
    if any(value <= 0 for value in closed_loop_positive):
        raise ConfigError("闭环验证超时、重规划和视频参数必须为正")
    future_steps = round(md.future_seconds * md.sensor_hz)
    if validation_vis.closed_loop_replan_action_steps > future_steps:
        raise ConfigError("闭环验证重规划步数不得超过模型未来动作长度")
    # 校验对象: training 优化参数——epoch 调度和 AdamW 参数必须可执行。
    training = cfg.training
    positive_training = (
        training.epochs, training.batch_size, training.gradient_accumulation, training.learning_rate,
        training.minimum_learning_rate, training.gradient_clip, training.checkpoint_interval,
        training.validation_interval,
    )
    if any(value <= 0 for value in positive_training) or training.num_workers < 0 or training.weight_decay < 0:
        raise ConfigError("training 的 epoch、batch、优化或间隔参数不合法")
    if training.dataloader_prefetch_factor <= 0 or training.log_interval_steps <= 0:
        raise ConfigError("training.dataloader_prefetch_factor 与 log_interval_steps 必须 > 0")
    if training.constantization_relative_std_threshold <= 0 or training.constantization_patience_intervals <= 0:
        raise ConfigError("training 的常数化监测阈值与连续区间数必须 > 0")
    if len(training.adam_betas) != 2 or not 0 < training.adam_betas[0] < 1 or not 0 < training.adam_betas[1] < 1:
        raise ConfigError("training.adam_betas 必须是两个 (0,1) 内的数")
    if not 0 <= training.warmup_epochs <= training.epochs:
        raise ConfigError("training.warmup_epochs 必须位于 [0, epochs]")
    if not 0 < training.teacher_forcing_fraction <= 1:
        raise ConfigError("training.teacher_forcing_fraction 必须位于 (0,1]")
    # 校验对象: post_training 行为微调调度——只定义独立阶段的输出、epoch和学习率。
    post = cfg.post_training
    post_positive = (
        post.epochs, post.learning_rate, post.minimum_learning_rate,
        post.checkpoint_interval, post.validation_interval,
    )
    if not post.output or any(value <= 0 for value in post_positive):
        raise ConfigError("post_training 的输出、epoch、学习率和间隔必须有效")
    if not 0 <= post.warmup_epochs <= post.epochs or post.minimum_learning_rate > post.learning_rate:
        raise ConfigError("post_training 的warmup或最小学习率不合法")


def build_config(data: dict[str, Any]) -> AppConfig:
    """从原始映射构造不可变配置并完成一次性校验。"""
    cfg = _construct(AppConfig, data)
    _validate(cfg)
    return cfg


__all__ = ["AppConfig", "ConfigError", "DataVisSettings", "build_config"]
