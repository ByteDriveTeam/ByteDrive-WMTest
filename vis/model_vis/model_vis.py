"""以分模态PCA RGB、特征热力图和动作曲线可视化检查点输出。

模块: vis/model_vis/model_vis.py
依赖: json, pathlib, numpy, pillow, torch, config, data.model_dataset, model.policy,
    vis.model_vis.checks
读取配置: model_vis.*, model_data.statistics, model_data.*, model.*
对外接口:
    - render_model_visualization(features, predicted_actions, target_actions, future_time, token_groups, output, cfg) -> dict
    - visualize_model_instance(model, sample, stats, cfg, output) -> dict
    - visualize_model_checkpoint(checkpoint, cfg, ...) -> dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.model_dataset import ByteDriveDataset, NormalizationStats, collate_policy_batches
from model.policy import ByteDrivePolicy, PolicyBatch, sensor_token_counts
from vis.model_vis.checks import check_model_visualization_arrays, check_model_visualization_inputs


ACTION_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper_width", "gripper_binary")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _feature_heatmap(features: np.ndarray, size: tuple[int, int], percentile: float, panel_rgb: list[int]) -> Image.Image:
    limit = max(float(np.percentile(np.abs(features), percentile)), np.finfo(np.float32).eps)
    unit = np.clip(features / limit, -1.0, 1.0)
    base = np.asarray(panel_rgb, dtype=np.float32)
    strength = np.abs(unit)[..., None]
    positive = np.asarray((255, 92, 72), dtype=np.float32)
    negative = np.asarray((68, 146, 255), dtype=np.float32)
    target = np.where((unit >= 0)[..., None], positive, negative)
    pixels = base + strength * (target - base)
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB").resize(size, Image.Resampling.BILINEAR)


def _pca_rgb(
    features: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """拟合一组Token自己的PCA基并返回三通道投影。"""
    values = features.astype(np.float32)
    mean = values.mean(0, keepdims=True)
    centered = values - mean
    component_count = min(3, *centered.shape)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    loadings = right[:component_count]
    # SVD的主成分符号不唯一，固定最大绝对载荷为正，使跨次可视化颜色可比。
    pivots = np.abs(loadings).argmax(1)
    signs = np.where(loadings[np.arange(component_count), pivots] < 0, -1.0, 1.0)
    loadings = loadings * signs[:, None]
    scores = (values - mean) @ loadings.T
    reference_scores = centered @ loadings.T
    scores = np.pad(scores, ((0, 0), (0, 3 - component_count)))
    reference_scores = np.pad(reference_scores, ((0, 0), (0, 3 - component_count)))
    limits = np.percentile(np.abs(reference_scores), percentile, axis=0)
    limits = np.maximum(limits, np.finfo(np.float32).eps)
    rgb = ((np.clip(scores / limits, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
    components = np.pad(loadings, ((0, 3 - component_count), (0, 0))).astype(np.float32)
    variance = singular.astype(np.float64) ** 2
    explained = np.zeros(3, dtype=np.float32)
    if float(variance.sum()) > 0:
        explained[:component_count] = (variance[:component_count] / variance.sum()).astype(np.float32)
    return scores.astype(np.float32), rgb, mean[0].astype(np.float32), components, explained


def _independent_spatial_pca(
    features: np.ndarray,
    layout: dict[str, Any],
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """分别拟合Overview、Wrist和触觉PCA，避免模态尺度主导同一个基。"""
    scores = np.zeros((len(features), 3), dtype=np.float32)
    rgb = np.full((len(features), 3), 128, dtype=np.uint8)
    names = ["overview", "wrist", "tactile"]
    means, components, explained = [], [], []
    for name in names:
        indices = layout[name]["slice"]
        group_scores, group_rgb, mean, basis, variance = _pca_rgb(features[indices], percentile)
        scores[indices], rgb[indices] = group_scores, group_rgb
        means.append(mean)
        components.append(basis)
        explained.append(variance)
    return scores, rgb, names, np.stack(means), np.stack(components), np.stack(explained)


def _spatial_layout(cfg: AppConfig) -> dict[str, Any]:
    """返回与模型flatten顺序严格一致的视觉、触觉和非空间Token布局。"""
    cameras = {camera.name: camera for camera in cfg.data_collector.render.cameras}
    rgb_frames = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
    sensor_frames = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
    tactile_height, tactile_width = cfg.data_collector.sensors.tactile_resolution
    cursor = cfg.model_data.language_length + 1 + cfg.model.register_tokens
    layout: dict[str, Any] = {"nonspatial_prefix": slice(0, cursor)}
    for name in ("overview", "wrist"):
        camera = cameras[name]
        rows = camera.height // cfg.model.image_patch
        columns = camera.width // cfg.model.image_patch
        count = rgb_frames * rows * columns
        layout[name] = {"slice": slice(cursor, cursor + count), "shape": (rgb_frames, rows, columns)}
        cursor += count
    tactile_rows = tactile_height // cfg.model.tactile_patch
    tactile_columns = tactile_width // cfg.model.tactile_patch
    tactile_count = sensor_frames * 2 * tactile_rows * tactile_columns
    layout["tactile"] = {
        "slice": slice(cursor, cursor + tactile_count),
        "shape": (sensor_frames, 2, tactile_rows, tactile_columns),
    }
    cursor += tactile_count
    layout["state"] = {"slice": slice(cursor, cursor + sensor_frames), "shape": (sensor_frames,)}
    layout["total"] = cursor + sensor_frames
    return layout


def _draw_spatial_sequence(
    canvas: Image.Image,
    frames: np.ndarray,
    bounds: tuple[int, int, int, int],
    columns: int,
    title: str,
    cfg: AppConfig,
) -> None:
    """以contact sheet展示时间序列，同时保持每帧Patch的二维邻接关系。"""
    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=8, fill=tuple(cfg.model_vis.panel_rgb))
    draw.text((left + 8, top + 6), title, fill=tuple(cfg.model_vis.text_rgb))
    inner_top, gap = top + 25, 4
    rows = max((len(frames) + columns - 1) // columns, 1)
    cell_width = max((right - left - 16 - gap * (columns - 1)) // columns, 1)
    cell_height = max((bottom - inner_top - 8 - gap * (rows - 1)) // rows, 1)
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame, mode="RGB")
        scale = max(min(cell_width / image.width, (cell_height - 10) / image.height), 1.0)
        resized = image.resize(
            (max(round(image.width * scale), 1), max(round(image.height * scale), 1)),
            Image.Resampling.NEAREST,
        )
        x0 = left + 8 + (index % columns) * (cell_width + gap)
        y0 = inner_top + (index // columns) * (cell_height + gap)
        x = x0 + (cell_width - resized.width) // 2
        y = y0 + max((cell_height - 10 - resized.height) // 2, 0)
        canvas.paste(resized, (x, y))
        draw.text((x0 + 2, y0 + cell_height - 10), f"t{index:02d}", fill=tuple(cfg.model_vis.text_rgb))


def _render_spatial_pca(pca_rgb: np.ndarray, layout: dict[str, Any], cfg: AppConfig) -> Image.Image:
    """把三组独立PCA RGB恢复成相机Patch网格和双指触觉Patch网格。"""
    width, height = cfg.model_vis.canvas_size[0], 1000
    canvas = Image.new("RGB", (width, height), tuple(cfg.model_vis.background_rgb))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Backbone spatial PCA RGB (independent modality bases)", fill=tuple(cfg.model_vis.text_rgb))
    draw.text((width - 445, 14), "PC1 / PC2 / PC3 -> R / G / B", fill=tuple(cfg.model_vis.text_rgb))
    gap, padding = 12, 20
    half = (width - 2 * padding - gap) // 2
    overview = pca_rgb[layout["overview"]["slice"]].reshape(*layout["overview"]["shape"], 3)
    wrist = pca_rgb[layout["wrist"]["slice"]].reshape(*layout["wrist"]["shape"], 3)
    _draw_spatial_sequence(canvas, overview, (padding, 42, padding + half, 365), 5, "Overview: 10 frames x 8x10 patches", cfg)
    _draw_spatial_sequence(canvas, wrist, (padding + half + gap, 42, width - padding, 365), 5, "Wrist: 10 frames x 6x6 patches", cfg)
    tactile = pca_rgb[layout["tactile"]["slice"]].reshape(*layout["tactile"]["shape"], 3)
    _draw_spatial_sequence(canvas, tactile[:, 0], (padding, 377, padding + half, 855), 10, "Left tactile: 50 frames x 4x4 patches", cfg)
    _draw_spatial_sequence(canvas, tactile[:, 1], (padding + half + gap, 377, width - padding, 855), 10, "Right tactile: 50 frames x 4x4 patches", cfg)
    prefix = pca_rgb[layout["nonspatial_prefix"]]
    state = pca_rgb[layout["state"]["slice"]]
    draw.text((padding, 870), "Non-spatial tokens: language + CLS + registers", fill=tuple(cfg.model_vis.text_rgb))
    prefix_image = Image.fromarray(prefix[None], mode="RGB").resize((half, 45), Image.Resampling.NEAREST)
    canvas.paste(prefix_image, (padding, 890))
    draw.text((padding + half + gap, 870), "State tokens: temporal order only", fill=tuple(cfg.model_vis.text_rgb))
    state_image = Image.fromarray(state[None], mode="RGB").resize((half, 45), Image.Resampling.NEAREST)
    canvas.paste(state_image, (padding + half + gap, 890))
    draw.text((padding, 950), "Overview, Wrist and tactile/contact colors are normalized within each modality and are not cross-modal coordinates.", fill=tuple(cfg.model_vis.text_rgb))
    return canvas


def _polyline(draw: ImageDraw.ImageDraw, values: np.ndarray, bounds: tuple[int, int, int, int], value_range: tuple[float, float], color: list[int], width: int) -> None:
    left, top, right, bottom = bounds
    low, high = value_range
    x = np.linspace(left, right, len(values))
    y = bottom - (np.asarray(values) - low) / max(high - low, np.finfo(np.float32).eps) * (bottom - top)
    draw.line(list(map(tuple, np.stack((x, y), -1))), fill=tuple(color), width=width, joint="curve")


def _action_panel(
    draw: ImageDraw.ImageDraw,
    predicted: np.ndarray,
    target: np.ndarray,
    name: str,
    bounds: tuple[int, int, int, int],
    cfg: AppConfig,
) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=8, fill=tuple(cfg.model_vis.panel_rgb))
    draw.text((left + 8, top + 6), name, fill=tuple(cfg.model_vis.text_rgb))
    plot = (left + 8, top + 26, right - 8, bottom - 12)
    low = float(min(predicted.min(), target.min()))
    high = float(max(predicted.max(), target.max()))
    margin = max((high - low) * 0.08, np.finfo(np.float32).eps)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=(82, 88, 98), width=1)
    _polyline(draw, target, plot, (low - margin, high + margin), cfg.model_vis.target_rgb, cfg.model_vis.line_width)
    _polyline(draw, predicted, plot, (low - margin, high + margin), cfg.model_vis.prediction_rgb, cfg.model_vis.line_width)


def _token_groups(cfg: AppConfig) -> list[tuple[str, int]]:
    overview, wrist, tactile, state = sensor_token_counts(cfg)
    return [
        ("language", cfg.model_data.language_length), ("CLS", 1), ("register", cfg.model.register_tokens),
        ("overview", overview), ("wrist", wrist), ("tactile", tactile), ("state", state),
    ]


def render_model_visualization(
    features: np.ndarray,
    predicted_actions: np.ndarray,
    target_actions: np.ndarray,
    future_time: np.ndarray,
    token_groups: list[tuple[str, int]],
    output: str | Path,
    cfg: AppConfig,
) -> dict[str, Any]:
    """生成特征热力图、PCA RGB、Token范数和9维动作对比。"""
    output_path = _project_path(output)
    check_model_visualization_arrays(
        features, predicted_actions, target_actions, future_time, token_groups, output_path,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    width, height = cfg.model_vis.canvas_size
    canvas = Image.new("RGB", (width, height), tuple(cfg.model_vis.background_rgb))
    draw = ImageDraw.Draw(canvas)
    padding, title_height = 24, 36
    feature_top = padding + title_height
    feature_bottom = cfg.model_vis.feature_panel_height
    pca_height = cfg.model_vis.pca_band_height
    heatmap_bounds = (padding, feature_top, width - padding, feature_bottom - pca_height - 95)
    draw.text((padding, padding), "Backbone final features", fill=tuple(cfg.model_vis.text_rgb))
    heatmap = _feature_heatmap(
        # Token放在横轴，使模态分界与下方逐Token范数严格对齐。
        np.asarray(features, dtype=np.float32).T,
        (heatmap_bounds[2] - heatmap_bounds[0], heatmap_bounds[3] - heatmap_bounds[1]),
        cfg.model_vis.feature_clip_percentile,
        cfg.model_vis.panel_rgb,
    )
    canvas.paste(heatmap, heatmap_bounds[:2])
    feature_values = np.asarray(features, dtype=np.float32)
    spatial_layout = _spatial_layout(cfg)
    has_spatial_layout = feature_values.shape[0] == spatial_layout["total"]
    if has_spatial_layout:
        (
            pca_scores, pca_rgb, pca_group_names, pca_means,
            pca_components, pca_explained,
        ) = _independent_spatial_pca(feature_values, spatial_layout, cfg.model_vis.pca_clip_percentile)
        pca_fit_tokens = {
            name: spatial_layout[name]["slice"].stop - spatial_layout[name]["slice"].start
            for name in pca_group_names
        }
    else:
        pca_scores, pca_rgb, pca_mean, pca_component, pca_variance = _pca_rgb(
            feature_values, cfg.model_vis.pca_clip_percentile,
        )
        pca_group_names = ["all"]
        pca_means = pca_mean[None]
        pca_components = pca_component[None]
        pca_explained = pca_variance[None]
        pca_fit_tokens = {"all": len(feature_values)}
    pca_top = heatmap_bounds[3] + 20
    pca_bounds = (padding, pca_top, width - padding, pca_top + pca_height)
    if has_spatial_layout:
        spatial_pca = _render_spatial_pca(pca_rgb, spatial_layout, cfg)
        draw.text((padding, pca_top - 17), "Spatial PCA RGB preview (full resolution saved separately)", fill=tuple(cfg.model_vis.text_rgb))
        pca_image = spatial_pca.resize(
            (pca_bounds[2] - pca_bounds[0], pca_height), Image.Resampling.LANCZOS,
        )
    else:
        spatial_pca = None
        draw.text((padding, pca_top - 17), "PCA RGB token strip (no spatial metadata)", fill=tuple(cfg.model_vis.text_rgb))
        pca_image = Image.fromarray(pca_rgb[None, ...], mode="RGB").resize(
            (pca_bounds[2] - pca_bounds[0], pca_height), Image.Resampling.NEAREST,
        )
    canvas.paste(pca_image, pca_bounds[:2])
    token_total = max(sum(count for _, count in token_groups), 1)
    cursor = 0
    norm = np.linalg.norm(features.astype(np.float32), axis=-1)
    norm_bounds = (padding, pca_bounds[3] + 5, width - padding, feature_bottom - 20)
    norm_high = max(float(norm.max()), np.finfo(np.float32).eps)
    _polyline(draw, norm, norm_bounds, (0.0, norm_high), cfg.model_vis.prediction_rgb, cfg.model_vis.line_width)
    for name, count in token_groups:
        x = padding + round(cursor / token_total * (width - 2 * padding))
        draw.line((x, feature_top, x, feature_bottom - 15), fill=(210, 214, 220), width=1)
        draw.text((x + 3, feature_bottom - 17), name, fill=tuple(cfg.model_vis.text_rgb))
        cursor += count

    action_top = feature_bottom + padding
    panel_gap = 12
    panel_width = (width - 2 * padding - 2 * panel_gap) // 3
    panel_height = (height - action_top - padding - 2 * panel_gap) // 3
    for index, name in enumerate(ACTION_NAMES):
        row, column = divmod(index, 3)
        left = padding + column * (panel_width + panel_gap)
        top = action_top + row * (panel_height + panel_gap)
        _action_panel(
            draw, predicted_actions[:, index], target_actions[:, index], name,
            (left, top, left + panel_width, top + panel_height), cfg,
        )
    draw.text((width - 330, padding), "prediction", fill=tuple(cfg.model_vis.prediction_rgb))
    draw.text((width - 220, padding), "target", fill=tuple(cfg.model_vis.target_rgb))

    image_path = output_path / "model_visualization.png"
    temporary_image = output_path / "model_visualization.tmp.png"
    canvas.save(temporary_image)
    os.replace(temporary_image, image_path)
    spatial_pca_path: Path | None = None
    if spatial_pca is not None:
        spatial_pca_path = output_path / "backbone_spatial_pca.png"
        temporary_spatial = output_path / "backbone_spatial_pca.tmp.png"
        spatial_pca.save(temporary_spatial)
        os.replace(temporary_spatial, spatial_pca_path)
    arrays_path = output_path / "model_visualization.npz"
    temporary_arrays = output_path / "model_visualization.npz.tmp"
    with temporary_arrays.open("wb") as stream:
        np.savez_compressed(
            stream, backbone_features=features, predicted_actions=predicted_actions,
            target_actions=target_actions, future_time=future_time,
            pca_scores=pca_scores, pca_rgb=pca_rgb,
            pca_group_names=np.asarray(pca_group_names), pca_means=pca_means,
            pca_components=pca_components, pca_explained_variance_ratio=pca_explained,
        )
    os.replace(temporary_arrays, arrays_path)
    return {
        "image": str(image_path), "arrays": str(arrays_path),
        "feature_shape": list(features.shape), "action_shape": list(predicted_actions.shape),
        "pca_shape": list(pca_scores.shape), "pca_rgb_shape": list(pca_rgb.shape),
        "spatial_pca": str(spatial_pca_path) if spatial_pca_path is not None else None,
        "pca_groups": pca_group_names,
        "pca_fit_tokens": pca_fit_tokens,
        "pca_explained_variance_ratio": pca_explained.tolist(),
        "token_groups": [{"name": name, "count": count} for name, count in token_groups],
    }


@torch.no_grad()
def visualize_model_instance(
    model: ByteDrivePolicy,
    sample: PolicyBatch,
    stats: NormalizationStats,
    cfg: AppConfig,
    output: str | Path,
) -> dict[str, Any]:
    """使用内存中的当前模型和固定噪声生成可跨epoch对比的推理图。"""
    device = next(model.parameters()).device
    batch = collate_policy_batches([sample]).to(device)
    batch.sensor_mask.zero_()
    generator = torch.Generator(device=device).manual_seed(cfg.model_vis.flow_noise_seed)
    noise = torch.randn(batch.flow_target.shape, generator=generator, device=device)
    was_training = model.training
    model.eval()
    try:
        model_output = model(batch, teacher_force_probability=0.0, flow_noise=noise)
    finally:
        model.train(was_training)
    if model_output.backbone_features is None:
        raise RuntimeError("策略未返回骨干末端特征")
    flow_mean = torch.as_tensor(stats.flow_mean, device=device)
    flow_std = torch.as_tensor(stats.flow_std, device=device)
    predicted = model_output.final_flow * flow_std + flow_mean
    target = batch.flow_target.float() * flow_std + flow_mean
    predicted_actions = predicted[0, :, :9].cpu().numpy()
    predicted_actions[:, 8] = np.where(predicted_actions[:, 8] >= 0, 1.0, -1.0)
    result = render_model_visualization(
        model_output.backbone_features[0].cpu().numpy(), predicted_actions,
        target[0, :, :9].cpu().numpy(), batch.future_time[0].cpu().numpy(),
        _token_groups(cfg), output, cfg,
    )
    output_path = _project_path(output)
    result.update({
        "phase_prediction": cfg.model.phase_names[int(model_output.phase_logits[0].argmax())],
        "summary": str(output_path / "summary.json"),
    })
    _atomic_json(output_path / "summary.json", result)
    return result


@torch.no_grad()
def visualize_model_checkpoint(
    checkpoint: str | Path | None,
    cfg: AppConfig,
    *,
    split: str | None = None,
    sample_index: int | None = None,
    output: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """读取固定滑窗；检查点缺省时用随机初始化模型生成同样的可视化。"""
    selected_split = split if split is not None else cfg.model_vis.split
    selected_index = sample_index if sample_index is not None else cfg.model_vis.sample_index
    selected_device = torch.device(device if device is not None else cfg.model_vis.device)
    checkpoint_path = None if checkpoint is None else Path(checkpoint).resolve()
    statistics = _project_path(cfg.model_data.statistics)
    output_root = _project_path(output if output is not None else cfg.model_vis.output)
    stats = NormalizationStats.load(statistics)
    dataset = ByteDriveDataset(cfg, selected_split, stats)
    check_model_visualization_inputs(
        checkpoint_path, statistics, output_root, selected_index, len(dataset), selected_device,
    )
    model = ByteDrivePolicy(cfg, (stats.flow_mean, stats.flow_std)).to(selected_device).eval()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    target_directory = output_root / f"{selected_split}_{selected_index:06d}"
    result = visualize_model_instance(model, dataset[selected_index], stats, cfg, target_directory)
    result.update({
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "initialization": "checkpoint" if checkpoint_path is not None else "random_initialization",
        "split": selected_split, "sample_index": selected_index,
        "summary": str(target_directory / "summary.json"),
    })
    _atomic_json(target_directory / "summary.json", result)
    return result


__all__ = ["render_model_visualization", "visualize_model_checkpoint", "visualize_model_instance"]
