"""编排单GPU epoch训练、EMA更新、评估与可恢复检查点。

模块: train/engine/engine.py
依赖: copy, json, math, torch, config, data.model_dataset, model.policy, train.objectives
读取配置: training.*, model.ema_decay, model_data.statistics
对外接口:
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
from pathlib import Path
import random
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
from model.policy import ByteDrivePolicy
from train.engine.checks import check_project_output, check_training_environment
from train.objectives import compute_policy_losses, teacher_force_probability


EMA_PREFIXES = (
    "overview_embed", "wrist_embed", "tactile_embed", "state_embed", "language_embed",
    "cls_token", "register_token", "position", "backbone", "backbone_mixer", "backbone_output_norm",
)


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
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=shuffle,
        num_workers=cfg.training.num_workers, pin_memory=cfg.training.device == "cuda",
        # 每个epoch重建worker，确保set_epoch后的随机窗口立即传入子进程。
        collate_fn=collate_policy_batches, persistent_workers=False,
    )


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
    totals = {name: 0.0 for name in ("total", "velocity", "endpoint", "reconstruction", "phase")}
    batches = 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = batch.to(device)
        teacher_features = teacher.encode_teacher(batch)
        output = model(batch, teacher_force_probability=0.0)
        loss = compute_policy_losses(output, batch, teacher_features, epoch, cfg)
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
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, cfg.training.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for batch_index, batch in enumerate(train_loader):
            epoch_progress = epoch + batch_index / max(len(train_loader), 1)
            batch = batch.to(device)
            with torch.no_grad():
                teacher_features = teacher.encode_teacher(batch)
            output_values = model(batch, teacher_force_probability(epoch_progress, cfg))
            loss = compute_policy_losses(output_values, batch, teacher_features, epoch_progress, cfg)
            (loss.total / cfg.training.gradient_accumulation).backward()
            running += float(loss.total.detach())
            if (batch_index + 1) % cfg.training.gradient_accumulation == 0 or batch_index + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_ema(teacher, model, cfg.model.ema_decay)
        record: dict[str, Any] = {"epoch": epoch, "train_total": running / max(len(train_loader), 1), "learning_rate": scheduler.get_last_lr()[0]}
        if (epoch + 1) % cfg.training.validation_interval == 0:
            record["validation"] = evaluate_model(model, teacher, validation_loader, cfg, epoch + 1)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
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
    "create_ema_teacher", "evaluate_checkpoint", "evaluate_model", "load_checkpoint",
    "save_checkpoint", "train_model", "update_ema",
]
