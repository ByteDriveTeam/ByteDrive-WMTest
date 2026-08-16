"""生成确定性任务场景并构造包含 Panda 的 MJCF。

模块: data/data_collector/scene/scene.py
依赖: config, numpy, xml, data.data_collector.records, data.data_collector.tasks
读取配置: data_collector.collector.master_seed, data_collector.simulation.*, data_collector.scene.*,
    data_collector.controller.gripper_open, data_collector.controller.finger_*,
    data_collector.controller.slide_direction,
    data_collector.render.cameras, data_collector.sensors.contact_enabled
对外接口:
    - generate_scene_spec(scene_index, attempt, cfg, task_type=None) -> SceneSpec
    - build_mjcf(spec, cfg) -> str
    - asset_fingerprint() -> str
    - materialize_mjcf(xml) -> str
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from config.schema import AppConfig
from data.data_collector.records import ObjectSpec, SceneSpec
from data.data_collector.scene.checks.scene_checks import check_scene_spec
from data.data_collector.tasks import build_task, choose_task_type, object_count_for_task


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "franka_emika_panda"
PANDA_XML = ASSET_ROOT / "panda.xml"
ASSET_TOKEN = "__PANDA_ASSET_DIR__"


def asset_fingerprint() -> str:
    """计算固定 Panda 资产目录的内容指纹。"""
    digest = hashlib.sha256()
    for path in sorted(item for item in ASSET_ROOT.rglob("*") if item.is_file()):
        digest.update(path.relative_to(ASSET_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sample_positions(count: int, rng: np.random.Generator, cfg: AppConfig) -> list[list[float]]:
    scene = cfg.data_collector.scene
    for _ in range(scene.placement_attempts):
        candidates = rng.uniform(scene.object_xy_min, scene.object_xy_max, size=(count, 2))
        if count > 1:
            distances = np.linalg.norm(candidates[:, None, :] - candidates[None, :, :], axis=-1)
            distances += np.eye(count) * scene.minimum_object_spacing
            if distances.min() < scene.minimum_object_spacing:
                continue
        positions = np.column_stack([candidates, np.full(count, scene.table_height)])
        return positions.tolist()
    raise RuntimeError("无法在配置范围内生成无重叠物体布局")


def generate_scene_spec(scene_index: int, attempt: int, cfg: AppConfig, task_type: str | None = None) -> SceneSpec:
    """根据主种子、场景序号与候选序号生成可复现的场景。"""
    seed_sequence = np.random.SeedSequence([cfg.data_collector.collector.master_seed, scene_index, attempt])
    task_rng, count_rng, layout_rng, physics_rng, appearance_rng = [
        np.random.default_rng(seed) for seed in seed_sequence.spawn(5)
    ]
    selected_task = task_type or choose_task_type(task_rng, cfg)
    count = object_count_for_task(selected_task, count_rng, cfg)
    positions = _sample_positions(count, layout_rng, cfg)
    scene_cfg = cfg.data_collector.scene
    colors = appearance_rng.choice(list(scene_cfg.palette), size=count, replace=False).tolist()
    shapes = [str(appearance_rng.choice(scene_cfg.object_shapes)) for _ in range(count)]
    if selected_task == "SLIDE_REGRASP":
        shapes[0] = str(appearance_rng.choice(scene_cfg.slide_target_shapes))
    objects = [
        ObjectSpec(
            name=f"object_{index}",
            shape=shape,
            color=str(color),
            size=float(physics_rng.uniform(scene_cfg.object_size_min, scene_cfg.object_size_max)),
            mass=float(physics_rng.uniform(scene_cfg.object_mass_min, scene_cfg.object_mass_max)),
            friction=float(physics_rng.uniform(scene_cfg.object_friction_min, scene_cfg.object_friction_max)),
            initial_position=position,
            initial_quaternion=[float(math.cos(yaw / 2)), 0.0, 0.0, float(math.sin(yaw / 2))],
        )
        for index, (position, yaw, shape, color) in enumerate(zip(
            positions, layout_rng.uniform(-math.pi, math.pi, count), shapes, colors, strict=True,
        ))
    ]
    task = build_task(selected_task, objects)
    seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
    spec = SceneSpec(
        scene_index=scene_index,
        seed=seed,
        task=task,
        objects=objects,
        slope_angle=float(physics_rng.uniform(scene_cfg.slope_angle_min, scene_cfg.slope_angle_max)),
        target_positions={name: list(position) for name, position in scene_cfg.target_positions.items()},
    )
    check_scene_spec(spec, cfg)
    return spec


def _numbers(values: list[float]) -> str:
    return " ".join(format(float(value), ".10g") for value in values)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"Panda 资产缺少 body: {name}")
    return body


def _add_environment(root: ET.Element, spec: SceneSpec, cfg: AppConfig) -> None:
    scene = cfg.data_collector.scene
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Panda MJCF 缺少 worldbody")
    link0 = _find_body(root, "link0")
    link0.set("pos", _numbers([0.0, 0.0, scene.table_height]))
    ET.SubElement(world, "geom", name="floor", type="plane", size=_numbers([0.0, 0.0, scene.floor_plane_size]), rgba=_numbers(scene.floor_rgba), friction=_numbers(scene.table_friction))
    table_position = [scene.table_size[0] / 2, 0.0, scene.table_height - scene.table_size[2] / 2]
    table_size = [value / 2 for value in scene.table_size]
    ET.SubElement(world, "geom", name="table", type="box", pos=_numbers(table_position), size=_numbers(table_size), rgba=_numbers(scene.table_rgba), friction=_numbers(scene.table_friction))
    active_targets = {step.target_ref for step in spec.task.steps if step.verb == "PLACE"}
    for name, position in spec.target_positions.items():
        if name not in active_targets:
            continue
        if name == "slope":
            continue
        marker_position = [position[0], position[1], scene.table_height + scene.target_size[2]]
        ET.SubElement(
            world, "geom", name=f"target_{name}", type="box", pos=_numbers(marker_position), size=_numbers(scene.target_size),
            rgba=_numbers(scene.target_rgba), contype="0", conaffinity="0", group="1",
        )
        if "bin" in name:
            half_x, half_y = scene.target_size[:2]
            half_height = scene.bin_wall_height / 2
            thickness = scene.bin_wall_thickness / 2
            wall_z = scene.table_height + half_height
            wall_specs = (
                ([position[0] - half_x, position[1], wall_z], [thickness, half_y, half_height]),
                ([position[0] + half_x, position[1], wall_z], [thickness, half_y, half_height]),
                ([position[0], position[1] - half_y, wall_z], [half_x, thickness, half_height]),
                ([position[0], position[1] + half_y, wall_z], [half_x, thickness, half_height]),
            )
            for wall_index, (wall_position, wall_size) in enumerate(wall_specs):
                ET.SubElement(world, "geom", name=f"{name}_wall_{wall_index}", type="box", pos=_numbers(wall_position), size=_numbers(wall_size), rgba=_numbers(scene.bin_wall_rgba), friction=_numbers(scene.bin_friction))
    if "slope" in active_targets:
        slope_position = spec.target_positions["slope"]
        slide_direction = np.asarray(cfg.data_collector.controller.slide_direction, dtype=np.float64)
        if abs(slide_direction[0]) >= abs(slide_direction[1]):
            slope_euler = [0.0, math.copysign(spec.slope_angle, slide_direction[0]), 0.0]
        else:
            slope_euler = [-math.copysign(spec.slope_angle, slide_direction[1]), 0.0, 0.0]
        ET.SubElement(
            world, "geom", name="slope", type="box", pos=_numbers(slope_position), size=_numbers(scene.slope_size),
            euler=_numbers(slope_euler), rgba=_numbers(scene.slope_rgba), friction=_numbers(scene.slope_friction),
        )
    for obj in spec.objects:
        body = ET.SubElement(world, "body", name=obj.name, pos=_numbers([obj.initial_position[0], obj.initial_position[1], scene.table_height + obj.size]), quat=_numbers(obj.initial_quaternion))
        ET.SubElement(body, "freejoint", name=f"{obj.name}_free")
        size = _numbers([obj.size] * 3) if obj.shape == "box" else _numbers([obj.size, obj.size])
        ET.SubElement(
            body, "geom", name=f"{obj.name}_geom", type=obj.shape, size=size, mass=format(obj.mass, ".10g"),
            friction=_numbers([obj.friction, 0.01, 0.001]), rgba=_numbers(scene.palette[obj.color]),
        )


def _add_sensors(root: ET.Element, cfg: AppConfig) -> None:
    sensor = ET.SubElement(root, "sensor")
    sites: list[str] = []
    for index in range(1, 8):
        site_name = f"joint{index}_imu"
        ET.SubElement(_find_body(root, f"link{index}"), "site", name=site_name, size="0.004", rgba="0 0 0 0")
        sites.append(site_name)
    hand = _find_body(root, "hand")
    ET.SubElement(hand, "site", name="ee_site", pos="0 0 0.103", size="0.006", rgba="0 0 0 0")
    sites.append("ee_site")
    for site_name in sites:
        ET.SubElement(sensor, "framepos", name=f"{site_name}_pos", objtype="site", objname=site_name, reftype="body", refname="link0")
        ET.SubElement(sensor, "framequat", name=f"{site_name}_quat", objtype="site", objname=site_name, reftype="body", refname="link0")
        ET.SubElement(sensor, "accelerometer", name=f"{site_name}_accel", site=site_name)
        ET.SubElement(sensor, "gyro", name=f"{site_name}_gyro", site=site_name)
        ET.SubElement(sensor, "framelinacc", name=f"{site_name}_linacc", objtype="site", objname=site_name)
        ET.SubElement(sensor, "frameangacc", name=f"{site_name}_angacc", objtype="site", objname=site_name)
        ET.SubElement(sensor, "framelinvel", name=f"{site_name}_linvel", objtype="site", objname=site_name)
        ET.SubElement(sensor, "frameangvel", name=f"{site_name}_angvel", objtype="site", objname=site_name)
    if cfg.data_collector.sensors.contact_enabled:
        for side in ("left", "right"):
            finger = _find_body(root, f"{side}_finger")
            ET.SubElement(finger, "site", name=f"{side}_tactile_site", type="box", pos="0 0.006 0.045", size="0.009 0.003 0.018", rgba="0 0 0 0")
            ET.SubElement(sensor, "touch", name=f"{side}_touch", site=f"{side}_tactile_site")


def _add_cameras(root: ET.Element, cfg: AppConfig) -> None:
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Panda MJCF 缺少 worldbody")
    for camera in cfg.data_collector.render.cameras:
        parent = world if camera.parent == "world" else _find_body(root, "link0" if camera.parent == "base" else "hand")
        focal_x = camera.width / (2.0 * math.tan(math.radians(camera.fov_x) / 2.0))
        focal_y = camera.height / (2.0 * math.tan(math.radians(camera.fov_y) / 2.0))
        attributes = {
            "name": camera.name, "pos": _numbers(camera.position),
            "resolution": f"{camera.width} {camera.height}",
            "sensorsize": f"{camera.width} {camera.height}",
            "focalpixel": _numbers([focal_x, focal_y]),
            # MuJoCo 把 principalpixel 定义为相对图像中心的偏移，零值才是居中主点。
            "principalpixel": _numbers([0.0, 0.0]),
            "euler": _numbers([math.radians(camera.roll), math.radians(camera.pitch), math.radians(camera.yaw)]),
        }
        ET.SubElement(parent, "camera", **attributes)


def _configure_actuators(root: ET.Element, cfg: AppConfig) -> None:
    actuator = root.find(".//actuator/general[@name='actuator8']")
    if actuator is None:
        raise RuntimeError("Panda MJCF 缺少夹爪执行器 actuator8")
    settings = cfg.data_collector.controller
    control_maximum = float(actuator.attrib["ctrlrange"].split()[1])
    gain = settings.finger_kp * settings.gripper_open / control_maximum
    actuator.set("gainprm", _numbers([gain, 0.0, 0.0]))
    actuator.set("biasprm", _numbers([0.0, -settings.finger_kp, -settings.finger_damping]))


def build_mjcf(spec: SceneSpec, cfg: AppConfig) -> str:
    """把固定 Panda 模型扩展为当前场景的可复现 MJCF。"""
    root = ET.parse(PANDA_XML).getroot()
    root.set("model", f"collector_scene_{spec.scene_index}")
    compiler = root.find("compiler")
    option = root.find("option")
    if compiler is None or option is None:
        raise RuntimeError("Panda MJCF 缺少 compiler 或 option")
    compiler.set("meshdir", ASSET_TOKEN)
    simulation = cfg.data_collector.simulation
    option.set("timestep", format(simulation.timestep, ".10g"))
    option.set("gravity", _numbers(simulation.gravity))
    option.set("iterations", str(simulation.solver_iterations))
    _configure_actuators(root, cfg)
    _add_environment(root, spec, cfg)
    _add_sensors(root, cfg)
    _add_cameras(root, cfg)
    return ET.tostring(root, encoding="unicode")


def materialize_mjcf(xml: str) -> str:
    """把可迁移资产占位符替换为当前项目内的绝对资产路径。"""
    return xml.replace(ASSET_TOKEN, (ASSET_ROOT / "assets").as_posix())


def scene_identifier(spec: SceneSpec, config_hash: str) -> str:
    """根据规范场景和配置指纹生成稳定短标识。"""
    payload = json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((config_hash + payload).encode("utf-8")).hexdigest()[:12]


__all__ = ["asset_fingerprint", "build_mjcf", "generate_scene_spec", "materialize_mjcf", "scene_identifier"]
