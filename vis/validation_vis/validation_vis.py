"""在每次验证后生成固定数据概览、模型推理和历史loss曲线。

模块: vis/validation_vis/validation_vis.py
依赖: json, os, pathlib, numpy, pillow, torch, config, data.model_dataset,
    model.policy, vis.model_vis, vis.validation_vis.checks
读取配置: validation_vis.*, model_vis.*, model_data.*, model.*
对外接口:
    - render_validation_data(sample, stats, output, cfg) -> dict
    - render_training_history(history, output, cfg) -> dict
    - generate_validation_visualizations(model, dataset, stats, history, epoch, cfg) -> dict
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
from data.model_dataset import ByteDriveDataset, NormalizationStats
from model.policy import ByteDrivePolicy, PolicyBatch, sensor_token_counts
from vis.model_vis import visualize_model_instance
from vis.validation_vis.checks import check_validation_visualization_inputs, check_validation_visualization_output


LOSS_NAMES = ("total", "velocity", "endpoint", "reconstruction", "phase")


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_image(image: Image.Image, path: Path) -> None:
    temporary = path.with_suffix(".tmp" + path.suffix)
    image.save(temporary)
    os.replace(temporary, path)


def _rgb_frame(frame: torch.Tensor) -> Image.Image:
    pixels = frame.detach().cpu().numpy().transpose(1, 2, 0)
    if np.issubdtype(pixels.dtype, np.floating) and float(np.nanmax(pixels, initial=0.0)) <= 1.0:
        pixels = pixels * 255.0
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def _contact_sheet(frames: torch.Tensor, size: tuple[int, int], columns: int, background: list[int]) -> Image.Image:
    width, height = size
    rows = max((len(frames) + columns - 1) // columns, 1)
    gap = 4
    cell_width = (width - gap * (columns - 1)) // columns
    cell_height = (height - gap * (rows - 1)) // rows
    sheet = Image.new("RGB", size, tuple(background))
    for index, frame in enumerate(frames):
        image = _rgb_frame(frame)
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        x = (index % columns) * (cell_width + gap) + (cell_width - image.width) // 2
        y = (index // columns) * (cell_height + gap) + (cell_height - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def _matrix_heatmap(values: np.ndarray, size: tuple[int, int], percentile: float, background: list[int]) -> Image.Image:
    limit = max(float(np.percentile(np.abs(values), percentile)), np.finfo(np.float32).eps)
    unit = np.clip(values / limit, -1.0, 1.0)
    base = np.asarray(background, dtype=np.float32)
    positive, negative = np.asarray((255, 92, 72)), np.asarray((68, 146, 255))
    target = np.where((unit >= 0)[..., None], positive, negative)
    pixels = base + np.abs(unit)[..., None] * (target - base)
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB").resize(size, Image.Resampling.NEAREST)


def _draw_mask_ratios(draw: ImageDraw.ImageDraw, sample: PolicyBatch, y: int, width: int, cfg: AppConfig) -> None:
    names = ("overview", "wrist", "tactile", "state")
    counts = sensor_token_counts(cfg)
    boundaries = np.cumsum((0, *counts))
    ratios = [float(sample.sensor_mask[boundaries[index]:boundaries[index + 1]].float().mean()) for index in range(4)]
    panel_width = (width - 40) // len(names)
    for index, (name, ratio) in enumerate(zip(names, ratios)):
        left = 20 + index * panel_width
        draw.text((left, y), f"{name} mask {ratio:.1%}", fill=tuple(cfg.model_vis.text_rgb))
        draw.rectangle((left, y + 20, left + panel_width - 12, y + 38), fill=tuple(cfg.model_vis.panel_rgb))
        draw.rectangle((left, y + 20, left + round((panel_width - 12) * ratio), y + 38), fill=tuple(cfg.model_vis.prediction_rgb))


def render_validation_data(sample: PolicyBatch, stats: NormalizationStats, output: str | Path, cfg: AppConfig) -> dict[str, Any]:
    """将固定验证滑窗的双相机、触觉、状态和掩码分布汇总为一张图。"""
    output_path = _project_path(output)
    check_validation_visualization_output(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    width, height = cfg.validation_vis.data_canvas_size
    canvas = Image.new("RGB", (width, height), tuple(cfg.model_vis.background_rgb))
    draw = ImageDraw.Draw(canvas)
    padding, gap = 20, 12
    half = (width - 2 * padding - gap) // 2
    draw.text((padding, 12), "Validation data: overview RGB", fill=tuple(cfg.model_vis.text_rgb))
    draw.text((padding + half + gap, 12), "Validation data: wrist RGB", fill=tuple(cfg.model_vis.text_rgb))
    overview = _contact_sheet(sample.overview_rgb, (half, 300), cfg.validation_vis.rgb_columns, cfg.model_vis.panel_rgb)
    wrist = _contact_sheet(sample.wrist_rgb, (half, 300), cfg.validation_vis.rgb_columns, cfg.model_vis.panel_rgb)
    canvas.paste(overview, (padding, 36))
    canvas.paste(wrist, (padding + half + gap, 36))

    tactile_mean = np.asarray(stats.tactile_map_mean, dtype=np.float32)
    tactile_std = np.asarray(stats.tactile_map_std, dtype=np.float32)
    tactile = sample.tactile.detach().cpu().numpy() * tactile_std[None, None, :, None, None] + tactile_mean[None, None, :, None, None]
    tactile_temporal = tactile.mean((-1, -2)).transpose(1, 2, 0).reshape(6, -1)
    draw.text((padding, 350), "Tactile spatial mean: left/right x normal/tangent-x/tangent-y", fill=tuple(cfg.model_vis.text_rgb))
    tactile_image = _matrix_heatmap(
        tactile_temporal, (width - 2 * padding, 150), cfg.validation_vis.tactile_clip_percentile,
        cfg.model_vis.panel_rgb,
    )
    canvas.paste(tactile_image, (padding, 372))

    state = sample.state.detach().cpu().numpy().T
    draw.text((padding, 540), "Normalized state features: 37 x 50", fill=tuple(cfg.model_vis.text_rgb))
    state_image = _matrix_heatmap(
        state, (width - 2 * padding, 260), cfg.validation_vis.state_clip_percentile,
        cfg.model_vis.panel_rgb,
    )
    canvas.paste(state_image, (padding, 562))
    _draw_mask_ratios(draw, sample, min(840, height - 60), width, cfg)
    path = output_path / "validation_data.png"
    _save_image(canvas, path)
    return {"image": str(path), "overview_frames": len(sample.overview_rgb), "sensor_frames": len(sample.state)}


def _draw_loss_panel(draw: ImageDraw.ImageDraw, history: list[dict[str, Any]], name: str, bounds: tuple[int, int, int, int], cfg: AppConfig) -> None:
    left, top, right, bottom = bounds
    draw.rounded_rectangle(bounds, radius=8, fill=tuple(cfg.model_vis.panel_rgb))
    draw.text((left + 10, top + 8), name, fill=tuple(cfg.model_vis.text_rgb))
    train = [(record["epoch"] + 1, record[f"train_{name}"]) for record in history if f"train_{name}" in record]
    validation = [(record["epoch"] + 1, record["validation"][name]) for record in history if name in record.get("validation", {})]
    values = [value for _, value in (*train, *validation)]
    if not values:
        return
    low, high = min(values), max(values)
    margin = max((high - low) * 0.08, np.finfo(np.float32).eps)
    max_epoch = max(epoch for epoch, _ in (*train, *validation))
    plot = (left + 10, top + 30, right - 10, bottom - 15)
    for points, color in ((train, cfg.model_vis.prediction_rgb), (validation, cfg.model_vis.target_rgb)):
        xy = [(
            plot[0] + (epoch - 1) / max(max_epoch - 1, 1) * (plot[2] - plot[0]),
            plot[3] - (value - low + margin) / (high - low + 2 * margin) * (plot[3] - plot[1]),
        ) for epoch, value in points]
        if len(xy) > 1:
            draw.line(xy, fill=tuple(color), width=cfg.model_vis.line_width, joint="curve")
        elif xy:
            x, y = xy[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tuple(color))


def render_training_history(history: list[dict[str, Any]], output: str | Path, cfg: AppConfig) -> dict[str, Any]:
    """绘制截至当前验证epoch的训练/验证总损失与四个分项曲线。"""
    output_path = _project_path(output)
    check_validation_visualization_output(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    width, height = cfg.validation_vis.history_canvas_size
    canvas = Image.new("RGB", (width, height), tuple(cfg.model_vis.background_rgb))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Loss history", fill=tuple(cfg.model_vis.text_rgb))
    draw.text((width - 240, 14), "train", fill=tuple(cfg.model_vis.prediction_rgb))
    draw.text((width - 150, 14), "validation", fill=tuple(cfg.model_vis.target_rgb))
    gap, padding, top = 12, 20, 42
    panel_width = (width - 2 * padding - gap) // 2
    panel_height = (height - top - padding - 2 * gap) // 3
    for index, name in enumerate(LOSS_NAMES):
        row, column = divmod(index, 2)
        left = padding + column * (panel_width + gap)
        panel_top = top + row * (panel_height + gap)
        _draw_loss_panel(draw, history, name, (left, panel_top, left + panel_width, panel_top + panel_height), cfg)
    path = output_path / "training_history.png"
    _save_image(canvas, path)
    return {"image": str(path), "epochs": len(history)}


@torch.no_grad()
def generate_validation_visualizations(
    model: ByteDrivePolicy,
    dataset: ByteDriveDataset,
    stats: NormalizationStats,
    history: list[dict[str, Any]],
    epoch: int,
    cfg: AppConfig,
) -> dict[str, Any]:
    """为一次已完成的验证生成三类可对比产物并更新latest索引。"""
    output_root = _project_path(cfg.validation_vis.output)
    index = cfg.validation_vis.sample_index
    check_validation_visualization_inputs(output_root, index, len(dataset))
    sample = dataset[index]
    epoch_directory = output_root / f"epoch_{epoch:04d}"
    model_result = visualize_model_instance(model, sample, stats, cfg, epoch_directory / "model")
    data_result = render_validation_data(sample, stats, epoch_directory / "data", cfg)
    history_result = render_training_history(history, epoch_directory / "history", cfg)
    result = {
        "epoch": epoch, "sample_index": index, "model": model_result,
        "data": data_result, "history": history_result,
        "summary": str(epoch_directory / "summary.json"),
    }
    _atomic_json(epoch_directory / "summary.json", result)
    _atomic_json(output_root / "latest.json", result)
    return result


__all__ = ["generate_validation_visualizations", "render_training_history", "render_validation_data"]
