"""编排单GPU epoch训练、EMA更新、评估与可恢复检查点。

模块: train/engine/engine.py
依赖: copy, json, math, os, time, torch, config, data.model_dataset, model.policy,
    train.objectives, vis.validation_vis
读取配置: training.*, loss.*, validation_vis.*, model.ema_decay,
    model_data.statistics, model_data.replay_cache.enabled
对外接口:
    - constantization_metrics(output, teacher_output, cls_index) -> dict
    - create_ema_teacher(model) -> ByteDrivePolicy
    - update_ema(teacher, student, decay) -> None
    - save_checkpoint(path, ...) -> None
    - load_checkpoint(path, ...) -> int
    - evaluate_model(model, teacher, loader, cfg, epoch) -> dict
    - train_model(cfg, resume=None) -> dict
    - evaluate_checkpoint(cfg, checkpoint) -> dict
"""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.model_dataset import ByteDriveDataset, NormalizationStats, collate_policy_batches
from model.policy import ByteDrivePolicy, PolicyOutput, TeacherOutput
from train.engine.checks import check_project_output, check_training_environment
from train.objectives import (
    compute_policy_losses, endpoint_weight, teacher_force_probability,
    visible_reconstruction_weight,
)
from vis.validation_vis import generate_validation_visualizations


EMA_PREFIXES = (
    "overview_embed", "wrist_embed", "tactile_embed", "state_embed", "language_embed",
    "cls_token", "register_token", "position", "backbone", "backbone_mixer", "backbone_output_norm",
)

CONSTANTIZATION_COMPONENTS = (
    "cls", "teacher_cls", "backbone", "teacher", "predictor", "velocity", "endpoint",
)


def _relative_std(values: torch.Tensor) -> torch.Tensor:
    """测量跨样本/Token变化相对总体幅值；常数向量返回0。"""
    flattened = values.detach().float().reshape(-1, values.shape[-1])
    centered = flattened - flattened.mean(0, keepdim=True)
    centered_rms = centered.square().mean().sqrt()
    value_rms = flattened.square().mean().sqrt()
    return centered_rms / value_rms.clamp_min(torch.finfo(torch.float32).eps)


@torch.no_grad()
def constantization_metrics(
    output: PolicyOutput,
    teacher_output: TeacherOutput,
    cls_index: int,
) -> dict[str, torch.Tensor]:
    """返回Student/Teacher CLS、感知特征与行为输出的无量纲常数化指标。"""
    return {
        "cls": _relative_std(output.backbone_features[:, cls_index]),
        "teacher_cls": _relative_std(teacher_output.cls_features),
        "backbone": _relative_std(output.observation_features),
        "teacher": _relative_std(teacher_output.observation_features),
        "predictor": _relative_std(output.predictor_features),
        "velocity": _relative_std(output.velocities),
        "endpoint": _relative_std(output.final_flow),
    }


def _constantization_flags(metrics: dict[str, float], threshold: float) -> dict[str, bool]:
    return {name: metrics[name] < threshold for name in CONSTANTIZATION_COMPONENTS}


def _statistics_path(cfg: AppConfig) -> Path:
    path = Path(cfg.model_data.statistics)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _output_path(cfg: AppConfig) -> Path:
    path = Path(cfg.training.output)
    output = path if path.is_absolute() else PROJECT_ROOT / path
    check_project_output(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _loader(dataset: ByteDriveDataset, cfg: AppConfig, shuffle: bool) -> DataLoader:
    options: dict[str, Any] = {}
    if cfg.training.num_workers:
        options["prefetch_factor"] = cfg.training.dataloader_prefetch_factor
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=shuffle,
        num_workers=cfg.training.num_workers, pin_memory=cfg.training.device == "cuda",
        # 每个epoch重建worker，确保set_epoch后的新掩码随机性立即传入子进程。
        collate_fn=collate_policy_batches, persistent_workers=False, **options,
    )


def _device_batches(loader: DataLoader, device: torch.device, enabled: bool):
    """在独立CUDA Stream中预取下一批，CPU模式保持直接迭代。"""
    if device.type != "cuda" or not enabled:
        iterator = iter(loader)
        while True:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            yield batch.to(device, non_blocking=device.type == "cuda"), time.perf_counter() - started
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)

    def preload():
        started = time.perf_counter()
        try:
            host_batch = next(iterator)
        except StopIteration:
            return None, time.perf_counter() - started
        with torch.cuda.stream(stream):
            device_batch = host_batch.to(device, non_blocking=True)
        return device_batch, time.perf_counter() - started

    next_batch, next_wait = preload()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch, wait = next_batch, next_wait
        next_batch, next_wait = preload()
        yield batch, wait


def _write_log(stream, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def _load_epoch_history(path: Path, before_epoch: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "epoch" and int(record.get("epoch", -1)) < before_epoch:
            records.append(record)
    return records


def create_ema_teacher(model: ByteDrivePolicy) -> ByteDrivePolicy:
    """创建冻结的FP32 EMA教师。"""
    teacher = copy.deepcopy(model).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_ema(teacher: ByteDrivePolicy, student: ByteDrivePolicy, decay: float) -> None:
    """仅更新感知嵌入、位置网络与骨干的EMA权重。"""
    teacher_parameters = dict(teacher.named_parameters())
    for name, parameter in student.named_parameters():
        if name.startswith(EMA_PREFIXES):
            teacher_parameters[name].mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)
    teacher_buffers = dict(teacher.named_buffers())
    for name, buffer in student.named_buffers():
        if name.startswith(EMA_PREFIXES) and name in teacher_buffers:
            teacher_buffers[name].copy_(buffer)


def _optimizer(model: ByteDrivePolicy, cfg: AppConfig) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and not name.endswith("bias") else no_decay).append(parameter)
    return AdamW(
        [{"params": decay, "weight_decay": cfg.training.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.training.learning_rate, betas=tuple(cfg.training.adam_betas),
    )


def _scheduler(optimizer: AdamW, updates_per_epoch: int, cfg: AppConfig) -> LambdaLR:
    warmup = cfg.training.warmup_epochs * updates_per_epoch
    total = cfg.training.epochs * updates_per_epoch
    minimum_ratio = cfg.training.minimum_learning_rate / cfg.training.learning_rate

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def save_checkpoint(
    path: str | Path,
    model: ByteDrivePolicy,
    teacher: ByteDrivePolicy,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    cfg: AppConfig,
) -> None:
    """保存完整可恢复训练状态。"""
    output = Path(path)
    check_project_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "teacher": teacher.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "config": asdict(cfg),
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }, temporary)
    temporary.replace(output)


def load_checkpoint(
    path: str | Path,
    model: ByteDrivePolicy,
    teacher: ByteDrivePolicy,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
) -> int:
    """恢复模型及可选优化状态，返回下一epoch。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    teacher.load_state_dict(checkpoint["teacher"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    return int(checkpoint["epoch"]) + 1


@torch.no_grad()
def evaluate_model(
    model: ByteDrivePolicy,
    teacher: ByteDrivePolicy,
    loader: DataLoader,
    cfg: AppConfig,
    epoch: float,
) -> dict[str, float]:
    """在固定验证窗口上计算各监督损失。"""
    model.eval()
    teacher.eval()
    totals = {name: 0.0 for name in (
        "total", "velocity", "endpoint", "reconstruction", "phase", "visreg",
        "visreg_invariance", "visreg_regularization",
        "visreg_scale", "visreg_shape", "visreg_center",
    )}
    batches = 0
    device = next(model.parameters()).device
    for batch, _ in _device_batches(loader, device, cfg.training.device_prefetch):
        teacher_output = teacher.encode_teacher(batch)
        output = model(batch, teacher_force_probability=0.0)
        loss = compute_policy_losses(output, batch, teacher_output, epoch, cfg)
        for name in totals:
            totals[name] += float(getattr(loss, name))
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def train_model(cfg: AppConfig, resume: str | Path | None = None) -> dict[str, Any]:
    """按epoch训练单CUDA ByteDrive策略并周期保存检查点。"""
    device = check_training_environment(cfg)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    stats_path = _statistics_path(cfg)
    if not stats_path.exists():
        raise FileNotFoundError(f"请先运行 stats 生成归一化文件: {stats_path}")
    stats = NormalizationStats.load(stats_path)
    train_dataset = ByteDriveDataset(cfg, "train", stats)
    validation_dataset = ByteDriveDataset(cfg, "validation", stats)
    train_loader, validation_loader = _loader(train_dataset, cfg, True), _loader(validation_dataset, cfg, False)
    model = ByteDrivePolicy(cfg, (stats.flow_mean, stats.flow_std)).to(device)
    teacher = create_ema_teacher(model).to(device)
    optimizer = _optimizer(model, cfg)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation)
    scheduler = _scheduler(optimizer, updates_per_epoch, cfg)
    start_epoch = load_checkpoint(resume, model, teacher, optimizer, scheduler) if resume else 0
    output = _output_path(cfg)
    log_path = output / "train.jsonl"
    if not resume:
        log_path.write_text("", encoding="utf-8")
    history = _load_epoch_history(log_path, start_epoch) if resume else []
    loss_names = (
        "total", "velocity", "endpoint", "reconstruction", "phase", "visreg",
        "visreg_invariance", "visreg_regularization",
        "visreg_scale", "visreg_shape", "visreg_center",
    )
    with log_path.open("a", encoding="utf-8") as log_stream:
        _write_log(log_stream, {
            "event": "train_start", "start_epoch": start_epoch, "epochs": cfg.training.epochs,
            "train_samples": len(train_dataset), "validation_samples": len(validation_dataset),
            "train_scenes": len(train_dataset.scenes), "validation_scenes": len(validation_dataset.scenes),
            "window_seconds": cfg.model_data.history_seconds + cfg.model_data.future_seconds,
            "window_stride_seconds": cfg.model_data.window_stride_seconds,
            "ema_decay": cfg.model.ema_decay, "ema_student_update_rate": 1.0 - cfg.model.ema_decay,
            "loss": asdict(cfg.loss), "validation_visualization": asdict(cfg.validation_vis),
            "constantization_monitor": {
                "enabled": cfg.training.constantization_monitor_enabled,
                "relative_std_threshold": cfg.training.constantization_relative_std_threshold,
                "patience_intervals": cfg.training.constantization_patience_intervals,
            },
            "device": str(device), "replay_cache": cfg.model_data.replay_cache.enabled,
            "replay_cache_directory": cfg.model_data.replay_cache.directory,
            "mujoco_gl": os.environ.get("MUJOCO_GL", "default"),
        })
        constantization_streaks = {name: 0 for name in CONSTANTIZATION_COMPONENTS}
        for epoch in range(start_epoch, cfg.training.epochs):
            train_dataset.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = {name: torch.zeros((), device=device) for name in loss_names}
            interval = {name: torch.zeros((), device=device) for name in loss_names}
            constantization_running = {name: torch.zeros((), device=device) for name in CONSTANTIZATION_COMPONENTS}
            constantization_interval = {name: torch.zeros((), device=device) for name in CONSTANTIZATION_COMPONENTS}
            interval_batches = interval_samples = interval_hits = interval_misses = 0
            interval_wait = 0.0
            interval_start = time.perf_counter()
            for batch_index, (batch, data_wait) in enumerate(_device_batches(
                train_loader, device, cfg.training.device_prefetch,
            )):
                epoch_progress = epoch + batch_index / max(len(train_loader), 1)
                with torch.no_grad():
                    teacher_output = teacher.encode_teacher(batch)
                output_values = model(batch, teacher_force_probability(epoch_progress, cfg))
                loss = compute_policy_losses(output_values, batch, teacher_output, epoch_progress, cfg)
                if cfg.training.constantization_monitor_enabled:
                    monitor_values = constantization_metrics(
                        output_values, teacher_output, cfg.model_data.language_length,
                    )
                    for name, value in monitor_values.items():
                        constantization_running[name].add_(value)
                        constantization_interval[name].add_(value)
                (loss.total / cfg.training.gradient_accumulation).backward()
                for name in loss_names:
                    value = getattr(loss, name).detach()
                    running[name].add_(value)
                    interval[name].add_(value)
                interval_batches += 1
                interval_samples += batch.overview_rgb.shape[0]
                interval_hits += int(batch.cache_hits.sum())
                interval_misses += int(batch.cache_misses.sum())
                interval_wait += data_wait
                if (batch_index + 1) % cfg.training.gradient_accumulation == 0 or batch_index + 1 == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    update_ema(teacher, model, cfg.model.ema_decay)
                should_log = (batch_index + 1) % cfg.training.log_interval_steps == 0 or batch_index + 1 == len(train_loader)
                if should_log:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - interval_start
                    cache_total = interval_hits + interval_misses
                    record = {
                        "event": "train_step", "epoch": epoch, "batch": batch_index + 1,
                        "batches": len(train_loader), "learning_rate": scheduler.get_last_lr()[0],
                        **{name: float(interval[name] / interval_batches) for name in loss_names},
                        "samples_per_second": interval_samples / max(elapsed, torch.finfo(torch.float32).eps),
                        "data_wait_ms": 1000.0 * interval_wait / interval_batches,
                        "cache_hits": interval_hits, "cache_misses": interval_misses,
                        "cache_hit_rate": interval_hits / max(cache_total, 1),
                        "endpoint_weight": loss.endpoint_weight,
                        "visible_reconstruction_weight": loss.visible_reconstruction_weight,
                    }
                    newly_constantized: list[str] = []
                    if cfg.training.constantization_monitor_enabled:
                        monitor = {
                            name: float(constantization_interval[name] / interval_batches)
                            for name in CONSTANTIZATION_COMPONENTS
                        }
                        flags = _constantization_flags(
                            monitor, cfg.training.constantization_relative_std_threshold,
                        )
                        for name, flagged in flags.items():
                            constantization_streaks[name] = constantization_streaks[name] + 1 if flagged else 0
                            if constantization_streaks[name] == cfg.training.constantization_patience_intervals:
                                newly_constantized.append(name)
                        record["constantization"] = {
                            "relative_std": monitor,
                            "below_threshold": flags,
                            "consecutive_intervals": dict(constantization_streaks),
                            "warning": any(
                                streak >= cfg.training.constantization_patience_intervals
                                for streak in constantization_streaks.values()
                            ),
                        }
                    if device.type == "cuda":
                        record["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    _write_log(log_stream, record)
                    if newly_constantized:
                        _write_log(log_stream, {
                            "event": "constantization_warning", "epoch": epoch,
                            "batch": batch_index + 1, "components": newly_constantized,
                            "relative_std": record["constantization"]["relative_std"],
                            "threshold": cfg.training.constantization_relative_std_threshold,
                            "patience_intervals": cfg.training.constantization_patience_intervals,
                        })
                    interval = {name: torch.zeros((), device=device) for name in loss_names}
                    constantization_interval = {
                        name: torch.zeros((), device=device) for name in CONSTANTIZATION_COMPONENTS
                    }
                    interval_batches = interval_samples = interval_hits = interval_misses = 0
                    interval_wait = 0.0
                    interval_start = time.perf_counter()
            record = {
                "event": "epoch", "epoch": epoch,
                **{f"train_{name}": float(running[name] / max(len(train_loader), 1)) for name in loss_names},
                "learning_rate": scheduler.get_last_lr()[0],
                "endpoint_weight": endpoint_weight(epoch + 1, cfg),
                "visible_reconstruction_weight": visible_reconstruction_weight(epoch + 1, cfg),
            }
            if cfg.training.constantization_monitor_enabled:
                epoch_monitor = {
                    name: float(constantization_running[name] / max(len(train_loader), 1))
                    for name in CONSTANTIZATION_COMPONENTS
                }
                record["constantization"] = {
                    "relative_std": epoch_monitor,
                    "below_threshold": _constantization_flags(
                        epoch_monitor, cfg.training.constantization_relative_std_threshold,
                    ),
                    "consecutive_intervals": dict(constantization_streaks),
                    "warning": any(
                        streak >= cfg.training.constantization_patience_intervals
                        for streak in constantization_streaks.values()
                    ),
                }
            if (epoch + 1) % cfg.training.validation_interval == 0:
                record["validation"] = evaluate_model(model, teacher, validation_loader, cfg, epoch + 1)
            history.append(record)
            if "validation" in record and cfg.validation_vis.enabled:
                try:
                    visualization = generate_validation_visualizations(
                        model, validation_dataset, stats, history, epoch + 1, cfg,
                    )
                    record["visualization"] = visualization["summary"]
                except Exception as error:
                    record["visualization_error"] = f"{type(error).__name__}: {error}"
                    if cfg.validation_vis.fail_on_error:
                        raise
            _write_log(log_stream, record)
            if (epoch + 1) % cfg.training.checkpoint_interval == 0 or epoch + 1 == cfg.training.epochs:
                save_checkpoint(output / f"epoch_{epoch + 1:04d}.pt", model, teacher, optimizer, scheduler, epoch, cfg)
    metrics = {"epochs": cfg.training.epochs, "history": history}
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_checkpoint(cfg: AppConfig, checkpoint: str | Path) -> dict[str, float]:
    """加载检查点并在测试集固定窗口上评估。"""
    device = check_training_environment(cfg)
    stats = NormalizationStats.load(_statistics_path(cfg))
    dataset = ByteDriveDataset(cfg, "test", stats)
    model = ByteDrivePolicy(cfg, (stats.flow_mean, stats.flow_std)).to(device)
    teacher = create_ema_teacher(model).to(device)
    epoch = load_checkpoint(checkpoint, model, teacher)
    return evaluate_model(model, teacher, _loader(dataset, cfg, False), cfg, epoch)


__all__ = [
    "constantization_metrics", "create_ema_teacher", "evaluate_checkpoint", "evaluate_model", "load_checkpoint",
    "save_checkpoint", "train_model", "update_ema",
]
