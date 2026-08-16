"""优先读取 LMDB 图像与触觉，并在缺失或强制时恢复物理状态重放和重算。

模块: vis/data_vis/data_vis.py
依赖: lmdb, mujoco, numpy, pillow, config, data.data_collector.scene,
    data.data_collector.storage, vis.data_vis.checks
读取配置: data_vis.*, data_collector.storage.max_dbs,
    data_collector.storage.frame_key_width, data_collector.render.cameras；触觉重算参数读取场景 config_snapshot
对外接口:
    - visualize_scene(dataset, scene, cfg, ...) -> dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lmdb
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from config import PROJECT_ROOT
from config.schema import AppConfig, SensorSettings
from data.data_collector.scene import add_virtual_tactile_sites, materialize_mjcf
from data.data_collector.simulation import compute_tactile_state
from data.data_collector.storage import decode_value, validate_scene
from vis.data_vis.checks import check_tactile_snapshot, check_visualization_inputs


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _select_scene(dataset: Path, scene: str | int) -> Path:
    if isinstance(scene, int) or str(scene).isdigit():
        matches = list(dataset.glob(f"scene_{int(scene):06d}_*.lmdb"))
    else:
        matches = list(dataset.glob(f"*{scene}*.lmdb"))
    if len(matches) != 1:
        raise ValueError(f"场景选择必须唯一，当前匹配 {len(matches)} 个")
    return matches[0]


def _camera_definition(metadata: dict[str, Any], camera_name: str, cfg: AppConfig) -> dict[str, Any]:
    snapshot = metadata.get("config_snapshot", {}).get("data_collector", {}).get("render", {})
    cameras = snapshot.get("cameras", [])
    camera = next((item for item in cameras if item.get("name") == camera_name), None)
    if camera is None:
        current = next((item for item in cfg.data_collector.render.cameras if item.name == camera_name), None)
        if current is not None:
            camera = dict(current.__dict__)
    if camera is None:
        raise ValueError(f"场景元数据和当前配置都未定义相机: {camera_name}")
    return camera


def _rgb_image(array: np.ndarray) -> Image.Image:
    rgb = np.asarray(array)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB 数组形状不合法: {rgb.shape}")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb, initial=0.0)) <= 1.0:
        rgb = rgb * 255.0
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _depth_image(array: np.ndarray, percentiles: list[float]) -> Image.Image:
    depth = np.asarray(array, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"深度数组形状不合法: {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        near, far = np.percentile(depth[valid], percentiles)
        if far <= near:
            far = near + np.finfo(np.float32).eps
        normalized[valid] = np.clip((depth[valid] - near) / (far - near), 0.0, 1.0)
    # 近处偏暖、远处偏蓝，黑色表示无效深度。
    red = 255.0 * (1.0 - normalized)
    green = 255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))
    blue = 255.0 * normalized
    rgb = np.stack([red, green, blue], axis=-1)
    rgb[~valid] = 0.0
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _segmentation_image(array: np.ndarray) -> Image.Image:
    segmentation = np.asarray(array)
    if segmentation.ndim == 3:
        first = segmentation[..., 0].astype(np.int64)
        second = segmentation[..., 1].astype(np.int64) if segmentation.shape[2] > 1 else 0
        identifiers = first + 4099 * second
    elif segmentation.ndim == 2:
        identifiers = segmentation.astype(np.int64)
    else:
        raise ValueError(f"实例分割数组形状不合法: {segmentation.shape}")
    valid = identifiers >= 0
    rgb = np.zeros((*identifiers.shape, 3), dtype=np.uint8)
    rgb[..., 0] = ((identifiers * 53 + 97) % 223 + 24).astype(np.uint8)
    rgb[..., 1] = ((identifiers * 97 + 31) % 223 + 24).astype(np.uint8)
    rgb[..., 2] = ((identifiers * 193 + 17) % 223 + 24).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb, mode="RGB")


def _visual_image(array: np.ndarray, modality: str, cfg: AppConfig) -> Image.Image:
    if modality == "rgb":
        return _rgb_image(array)
    if modality == "depth":
        return _depth_image(array, cfg.data_vis.depth_percentiles)
    return _segmentation_image(array)


def _stored_image_is_readable(value: Any, modality: str, camera: dict[str, Any]) -> bool:
    if not isinstance(value, np.ndarray) or value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return False
    height, width = int(camera["height"]), int(camera["width"])
    if value.shape[:2] != (height, width):
        return False
    if modality == "rgb":
        return value.ndim == 3 and value.shape[2] >= 3
    if modality == "depth":
        return value.ndim == 2 or (value.ndim == 3 and value.shape[2] == 1)
    return value.ndim == 2 or (value.ndim == 3 and value.shape[2] >= 1)


def _force_heatmap(force: np.ndarray, maximum: float, gamma: float) -> Image.Image:
    values = np.power(np.clip(np.asarray(force, dtype=np.float32) / maximum, 0.0, 1.0), gamma)
    red = np.clip(values * 2.0, 0.0, 1.0)
    green = np.clip(values * 2.0 - 0.5, 0.0, 1.0)
    blue = np.clip(values * 1.5, 0.0, 1.0)
    rgb = (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _dashboard(
    image: Image.Image,
    frame: dict[str, Any],
    source: str,
    tactile_source: str,
    modality: str,
    cfg: AppConfig,
) -> Image.Image:
    vis = cfg.data_vis
    panel_width = vis.panel_width
    lines = [
        f"Frame: {int(frame['frame_index']):08d}",
        f"Time: {float(frame['simulation_time']):.3f} s",
        f"Phase: {str(frame.get('phase', ''))[:24]}",
        f"Source: {source}",
        f"Tactile: {tactile_source}",
        f"Mode: {modality}",
    ]
    qpos = np.asarray(frame.get("robot", {}).get("joint_position", []), dtype=np.float32)
    joint_top = vis.padding + len(lines) * vis.text_line_height + vis.padding
    tactile_top = joint_top + (len(qpos[:7]) + 2) * vis.text_line_height
    available_width = panel_width - 3 * vis.padding
    map_size = max(32, min(vis.tactile_map_size, available_width // 2))
    content_height = tactile_top + 3 * vis.text_line_height + map_size + vis.padding
    height = max(image.height, vis.panel_min_height, content_height)
    canvas = Image.new("RGB", (image.width + panel_width, height), tuple(vis.background_rgb))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x0 = image.width + vis.padding
    text_color = tuple(vis.text_rgb)
    for line_index, line in enumerate(lines):
        draw.text((x0, vis.padding + line_index * vis.text_line_height), line, fill=text_color)

    draw.text((x0, joint_top), "Joint position", fill=text_color)
    bar_left = x0 + 3 * vis.padding
    bar_right = image.width + panel_width - vis.padding
    center = (bar_left + bar_right) // 2
    for index, value in enumerate(qpos[:7]):
        y = joint_top + vis.text_line_height + index * vis.text_line_height
        draw.text((x0, y - 2), f"J{index + 1}", fill=text_color)
        draw.line((bar_left, y + 4, bar_right, y + 4), fill=(70, 76, 84), width=3)
        endpoint = center + int(np.clip(value / vis.joint_display_range, -1.0, 1.0) * (bar_right - bar_left) / 2)
        draw.line((center, y + 4, endpoint, y + 4), fill=(70, 190, 240), width=5)

    draw.text((x0, tactile_top), "Tactile normal force (N)", fill=text_color)
    force_maps = frame.get("tactile", {}).get("force_maps", {})
    for index, side in enumerate(("left", "right")):
        map_x = x0 + index * (map_size + vis.padding)
        values = force_maps.get(side)
        if isinstance(values, np.ndarray) and values.shape == (32, 32, 3):
            peaks = np.max(np.abs(values), axis=(0, 1))
            draw.text((map_x, tactile_top + vis.text_line_height), f"{side} N:{peaks[0]:.2f}", fill=text_color)
            tactile_image = _force_heatmap(values[..., 0], vis.tactile_force_max, vis.tactile_force_gamma).resize((map_size, map_size), Image.Resampling.NEAREST)
            map_top = tactile_top + 2 * vis.text_line_height
            canvas.paste(tactile_image, (map_x, map_top))
            draw.text((map_x, map_top + map_size), f"Tx:{peaks[1]:.2f} Ty:{peaks[2]:.2f}", fill=text_color)
        else:
            draw.text((map_x, tactile_top + vis.text_line_height), side, fill=text_color)
            map_top = tactile_top + 2 * vis.text_line_height
            draw.rectangle((map_x, map_top, map_x + map_size - 1, map_top + map_size - 1), outline=(80, 86, 94))
            draw.text((map_x + vis.padding // 2, map_top + map_size // 2), "not available", fill=(140, 146, 154))
    return canvas


def _sensor_settings(metadata: dict[str, Any]) -> SensorSettings:
    snapshot = metadata.get("config_snapshot", {}).get("data_collector", {}).get("sensors")
    check_tactile_snapshot(snapshot)
    return SensorSettings(**snapshot)


def _stored_tactile_is_readable(frame: dict[str, Any], settings: SensorSettings) -> bool:
    maps = frame.get("tactile", {}).get("force_maps", {})
    shape = (*settings.tactile_resolution, 3)
    return all(
        isinstance(maps.get(side), np.ndarray)
        and maps[side].shape == shape
        and np.issubdtype(maps[side].dtype, np.number)
        for side in ("left", "right")
    )


class _PhysicsReplayer:
    def __init__(self, metadata: dict[str, Any], camera: dict[str, Any]) -> None:
        xml = add_virtual_tactile_sites(metadata["mjcf_xml"])
        self.model = mujoco.MjModel.from_xml_string(materialize_mjcf(xml))
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        self.sensor_settings = _sensor_settings(metadata)
        self.renderer: mujoco.Renderer | None = None

    def restore(self, frame: dict[str, Any]) -> None:
        mujoco.mj_setState(self.model, self.data, frame["physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, self.data)

    def render(self, modality: str) -> np.ndarray:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(
                self.model, int(self.camera["height"]), int(self.camera["width"]),
            )
        if modality == "depth":
            self.renderer.enable_depth_rendering()
        elif modality == "segmentation":
            self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(self.data, camera=self.camera["name"])
            return self.renderer.render().copy()
        finally:
            if modality == "depth":
                self.renderer.disable_depth_rendering()
            elif modality == "segmentation":
                self.renderer.disable_segmentation_rendering()

    def recompute_tactile(self) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        """从当前已恢复状态的接触约束重新计算双指三轴触觉力图。"""
        return compute_tactile_state(self.model, self.data, self.sensor_settings)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def _frame_indices(frame_count: int, start: int, end: int, stride: int, maximum: int) -> list[int]:
    if start >= frame_count:
        raise IndexError(f"起始帧 {start} 超出场景帧数 {frame_count}")
    final = frame_count - 1 if end < 0 else min(end, frame_count - 1)
    if final < start:
        raise ValueError("可视化末帧不能小于首帧")
    return list(range(start, final + 1, stride))[:maximum]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _clear_previous_outputs(target: Path, output_root: Path) -> None:
    resolved = target.resolve()
    boundary = output_root.resolve()
    if resolved == boundary or boundary not in resolved.parents:
        raise ValueError("拒绝清理可视化输出边界以外的路径")
    frames = resolved / "frames"
    if frames.is_dir():
        for image_path in frames.glob("frame_*.png"):
            image_path.unlink()
    for filename in ("animation.gif", "summary.json", "summary.json.tmp"):
        path = resolved / filename
        if path.is_file():
            path.unlink()


def visualize_scene(
    dataset: str | Path,
    scene: str | int,
    cfg: AppConfig,
    *,
    camera: str | None = None,
    modality: str | None = None,
    output: str | Path | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stride: int | None = None,
    max_frames: int | None = None,
    gif_enabled: bool | None = None,
    force_replay: bool | None = None,
    force_tactile_replay: bool | None = None,
) -> dict[str, Any]:
    """生成带遥测面板的 PNG 帧，并在触觉缺失或强制时从物理状态重算。"""
    vis = cfg.data_vis
    dataset_path = _project_path(dataset)
    output_root = _project_path(output if output is not None else vis.output)
    selected_camera = camera if camera is not None else vis.camera
    selected_modality = modality if modality is not None else vis.modality
    replay_only = force_replay if force_replay is not None else vis.force_replay
    tactile_replay_only = force_tactile_replay if force_tactile_replay is not None else vis.force_tactile_replay
    make_gif = gif_enabled if gif_enabled is not None else vis.gif_enabled
    check_visualization_inputs(dataset_path, output_root, selected_modality)
    scene_path = _select_scene(dataset_path, scene)
    report = validate_scene(scene_path, cfg, deep=False)

    env = lmdb.open(str(scene_path), readonly=True, lock=False, readahead=False, max_dbs=cfg.data_collector.storage.max_dbs)
    replayer: _PhysicsReplayer | None = None
    images: list[Image.Image] = []
    sources = {"stored": 0, "replayed": 0}
    tactile_sources = {"stored": 0, "recomputed": 0}
    try:
        meta_db = env.open_db(b"meta", create=False)
        frames_db = env.open_db(b"frames", create=False)
        index_db = env.open_db(b"index", create=False)
        with env.begin() as transaction:
            metadata = decode_value(transaction.get(b"scene", db=meta_db))
            summary = decode_value(transaction.get(b"summary", db=index_db))
            camera_definition = _camera_definition(metadata, selected_camera, cfg)
            sensor_settings = _sensor_settings(metadata)
            indices = _frame_indices(
                int(summary["frame_count"]),
                start_frame if start_frame is not None else vis.start_frame,
                end_frame if end_frame is not None else vis.end_frame,
                stride if stride is not None else vis.stride,
                max_frames if max_frames is not None else vis.max_frames,
            )
            target = output_root / f"{scene_path.stem}_{selected_camera}_{selected_modality}"
            _clear_previous_outputs(target, output_root)
            frames_output = target / "frames"
            frames_output.mkdir(parents=True, exist_ok=True)
            for frame_index in indices:
                key = str(frame_index).zfill(cfg.data_collector.storage.frame_key_width).encode("ascii")
                encoded = transaction.get(key, db=frames_db)
                if encoded is None:
                    raise IndexError(f"场景不存在帧 {frame_index}")
                frame = decode_value(encoded)
                stored = frame.get("cameras", {}).get(selected_camera, {}).get(selected_modality)
                image_replay = replay_only or not _stored_image_is_readable(stored, selected_modality, camera_definition)
                stored_tactile = _stored_tactile_is_readable(frame, sensor_settings)
                tactile_replay = tactile_replay_only or not stored_tactile
                if image_replay or tactile_replay:
                    if replayer is None:
                        replayer = _PhysicsReplayer(metadata, camera_definition)
                    replayer.restore(frame)
                if not image_replay:
                    pixels = stored
                    source = "stored"
                else:
                    pixels = replayer.render(selected_modality)
                    source = "replayed"
                display_frame = frame
                if tactile_replay:
                    contacts, force_maps = replayer.recompute_tactile()
                    display_frame = dict(frame)
                    display_frame["contacts"] = contacts
                    display_frame["tactile"] = {
                        "channel_order": ["normal", "tangent_x", "tangent_y"],
                        "force_maps": force_maps,
                    }
                    tactile_source = "recomputed"
                else:
                    tactile_source = "stored"
                sources[source] += 1
                tactile_sources[tactile_source] += 1
                dashboard = _dashboard(
                    _visual_image(pixels, selected_modality, cfg), display_frame, source,
                    tactile_source, selected_modality, cfg,
                )
                dashboard.save(frames_output / f"frame_{frame_index:08d}.png")
                images.append(dashboard)
    finally:
        if replayer is not None:
            replayer.close()
        env.close()

    gif_path: Path | None = None
    if make_gif and images:
        gif_path = target / "animation.gif"
        duration = max(1, round(1000 / vis.gif_fps))
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=False)
    result: dict[str, Any] = {
        "scene_id": report["scene_id"],
        "scene": scene_path.name,
        "camera": selected_camera,
        "modality": selected_modality,
        "frame_count": len(images),
        "frame_indices": indices,
        "source_counts": sources,
        "tactile_source_counts": tactile_sources,
        "frames_output": str(frames_output),
        "gif": str(gif_path) if gif_path is not None else None,
        "summary": str(target / "summary.json"),
    }
    _atomic_json(target / "summary.json", result)
    return result


__all__ = ["visualize_scene"]
