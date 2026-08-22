"""用完整观测仅后训练23维行为流与阶段分类，并生成完整验证产物。

模块: train/post_training/post_training.py
依赖: json, math, pathlib, random, numpy, torch, config, data.model_dataset,
    model.policy, train.engine.checks, train.objectives, train.post_training.checks,
    vis.closed_loop_validation, vis.validation_vis
读取配置: post_training.*, training.batch_size, training.gradient_accumulation,
    training.num_workers, training.dataloader_prefetch_factor, training.device_prefetch,
    training.weight_decay, training.adam_betas, training.gradient_clip, training.seed,
    training.device, loss.*, validation_vis.*, model_data.statistics
对外接口:
    - evaluate_post_training_checkpoint(cfg, checkpoint) -> dict
    - post_train_model(cfg, checkpoint, resume=False) -> dict
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
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
from model.policy import ByteDrivePolicy
from train.engine.checks import check_project_output, check_training_environment
from train.objectives import compute_post_training_losses
from train.post_training.checks import check_post_training_checkpoint, check_post_training_load
from vis.closed_loop_validation import run_fixed_closed_loop_validation
from vis.validation_vis import generate_validation_visualizations


PREDICTOR_PREFIXES = ("mask_token", "predictor.", "predictor_mixer.", "predictor_output_norm.")
LOSS_NAMES = (
    "total", "velocity", "endpoint", "reconstruction", "phase", "visreg",
    "visreg_invariance", "visreg_regularization", "visreg_scale", "visreg_shape", "visreg_center",
)


def _statistics_path(cfg: AppConfig) -> Path:
    path = Path(cfg.model_data.statistics)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _output_path(cfg: AppConfig) -> Path:
    path = Path(cfg.post_training.output)
    output = path if path.is_absolute() else PROJECT_ROOT / path
    check_project_output(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _is_predictor_parameter(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix) for prefix in PREDICTOR_PREFIXES)


def _behavior_state_dict(model: ByteDrivePolicy) -> dict[str, torch.Tensor]:
    return {name: value for name, value in model.state_dict().items() if not _is_predictor_parameter(name)}


def _freeze_predictor(model: ByteDrivePolicy) -> None:
    for name, parameter in model.named_parameters():
        if _is_predictor_parameter(name):
            parameter.requires_grad_(False)


def _load_checkpoint(
    path: str | Path,
    model: ByteDrivePolicy,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    *,
    resume: bool = False,
) -> tuple[int, dict]:
    source = Path(path).resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    check_post_training_checkpoint(source, checkpoint, resume)
    expected_excluded = {name for name in model.state_dict() if _is_predictor_parameter(name)}
    filtered = {
        name: value for name, value in checkpoint["model"].items()
        if not _is_predictor_parameter(name)
    }
    loaded = model.load_state_dict(filtered, strict=False)
    check_post_training_load(loaded.missing_keys, loaded.unexpected_keys, expected_excluded)
    if not resume:
        return 0, checkpoint
    assert optimizer is not None and scheduler is not None
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "torch_rng" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    return int(checkpoint["epoch"]) + 1, checkpoint


def _save_checkpoint(
    path: Path,
    model: ByteDrivePolicy,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    cfg: AppConfig,
    source_checkpoint: str | Path,
) -> None:
    check_project_output(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "stage": "post_training", "model": _behavior_state_dict(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "config": asdict(cfg), "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }, temporary)
    temporary.replace(path)


def _loader(dataset: ByteDriveDataset, cfg: AppConfig, shuffle: bool) -> DataLoader:
    options: dict[str, Any] = {}
    if cfg.training.num_workers:
        options["prefetch_factor"] = cfg.training.dataloader_prefetch_factor
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=shuffle,
        num_workers=cfg.training.num_workers, pin_memory=cfg.training.device == "cuda",
        collate_fn=collate_policy_batches, persistent_workers=False, **options,
    )


def _device_batches(loader: DataLoader, device: torch.device, prefetch: bool):
    if device.type != "cuda" or not prefetch:
        for batch in loader:
            yield batch.to(device, non_blocking=device.type == "cuda")
        return
    stream = torch.cuda.Stream(device=device)
    iterator = iter(loader)

    def preload():
        try:
            host = next(iterator)
        except StopIteration:
            return None
        with torch.cuda.stream(stream):
            return host.to(device, non_blocking=True)

    next_batch = preload()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch, next_batch = next_batch, preload()
        yield batch


def _optimizer(model: ByteDrivePolicy, cfg: AppConfig) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 and not name.endswith("bias") else no_decay).append(parameter)
    return AdamW(
        [{"params": decay, "weight_decay": cfg.training.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.post_training.learning_rate, betas=tuple(cfg.training.adam_betas),
    )


def _scheduler(optimizer: AdamW, updates_per_epoch: int, cfg: AppConfig) -> LambdaLR:
    settings = cfg.post_training
    warmup, total = settings.warmup_epochs * updates_per_epoch, settings.epochs * updates_per_epoch
    minimum_ratio = settings.minimum_learning_rate / settings.learning_rate

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def _full_observation(batch):
    batch.sensor_mask.zero_()
    return batch


@torch.no_grad()
def _evaluate(
    model: ByteDrivePolicy,
    loader: DataLoader,
    cfg: AppConfig,
    epoch: int,
    stats: NormalizationStats,
) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    totals = {name: 0.0 for name in LOSS_NAMES}
    batches = 0
    for batch in _device_batches(loader, device, cfg.training.device_prefetch):
        output = model(_full_observation(batch), 0.0, run_predictor=False)
        loss = compute_post_training_losses(output, batch, cfg)
        for name in LOSS_NAMES:
            totals[name] += float(getattr(loss, name))
        batches += 1
    result: dict[str, Any] = {name: value / max(batches, 1) for name, value in totals.items()}
    if cfg.validation_vis.closed_loop_enabled:
        try:
            result["closed_loop"] = run_fixed_closed_loop_validation(model, stats, cfg, epoch)
        except Exception as error:
            result["closed_loop"] = {"status": "error", "success": False, "error": f"{type(error).__name__}: {error}"}
            if cfg.validation_vis.fail_on_error:
                raise
    return result


def _history(path: Path, before_epoch: int) -> list[dict[str, Any]]:
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


def _write_log(stream, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def post_train_model(
    cfg: AppConfig,
    checkpoint: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """从Student权重启动完整观测行为后训练；可恢复后训练自身的优化状态。"""
    device = check_training_environment(cfg)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    stats = NormalizationStats.load(_statistics_path(cfg))
    train_dataset = ByteDriveDataset(cfg, "train", stats)
    validation_dataset = ByteDriveDataset(cfg, "validation", stats)
    train_loader, validation_loader = _loader(train_dataset, cfg, True), _loader(validation_dataset, cfg, False)
    model = ByteDrivePolicy(cfg, (stats.flow_mean, stats.flow_std)).to(device)
    _freeze_predictor(model)
    optimizer = _optimizer(model, cfg)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation)
    scheduler = _scheduler(optimizer, updates_per_epoch, cfg)
    start_epoch, source = _load_checkpoint(checkpoint, model, optimizer, scheduler, resume=resume)
    output = _output_path(cfg)
    log_path = output / "post_train.jsonl"
    if not resume:
        log_path.write_text("", encoding="utf-8")
    history = _history(log_path, start_epoch) if resume else []
    validation_cfg = replace(
        cfg, validation_vis=replace(cfg.validation_vis, output=str(output / "validation_visualizations")),
    )
    with log_path.open("a", encoding="utf-8") as log_stream:
        _write_log(log_stream, {
            "event": "post_train_start", "start_epoch": start_epoch,
            "epochs": cfg.post_training.epochs, "source_checkpoint": str(Path(checkpoint).resolve()),
            "source_stage": source.get("stage", "pre_training"), "device": str(device),
            "full_observation": True, "teacher_forcing": 0.0,
            "teacher_loaded": False, "predictor_loaded": False,
            "supervision": ["velocity_23d_all_layers", "endpoint_23d", "phase"],
        })
        for epoch in range(start_epoch, cfg.post_training.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = {name: torch.zeros((), device=device) for name in LOSS_NAMES}
            started = time.perf_counter()
            for batch_index, batch in enumerate(_device_batches(train_loader, device, cfg.training.device_prefetch)):
                output_values = model(_full_observation(batch), 0.0, run_predictor=False)
                loss = compute_post_training_losses(output_values, batch, cfg)
                (loss.total / cfg.training.gradient_accumulation).backward()
                for name in LOSS_NAMES:
                    running[name].add_(getattr(loss, name).detach())
                if (batch_index + 1) % cfg.training.gradient_accumulation == 0 or batch_index + 1 == len(train_loader):
                    nn.utils.clip_grad_norm_(filter(lambda parameter: parameter.requires_grad, model.parameters()), cfg.training.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            record: dict[str, Any] = {
                "event": "epoch", "epoch": epoch,
                **{f"train_{name}": float(running[name] / max(len(train_loader), 1)) for name in LOSS_NAMES},
                "learning_rate": scheduler.get_last_lr()[0], "seconds": time.perf_counter() - started,
                "endpoint_weight": cfg.loss.endpoint_end_weight,
                "visible_reconstruction_weight": 0.0,
            }
            if (epoch + 1) % cfg.post_training.validation_interval == 0:
                record["validation"] = _evaluate(model, validation_loader, validation_cfg, epoch + 1, stats)
            history.append(record)
            if "validation" in record and cfg.validation_vis.enabled:
                try:
                    visualization = generate_validation_visualizations(
                        model, validation_dataset, stats, history, epoch + 1, validation_cfg,
                        run_predictor=False,
                    )
                    record["visualization"] = visualization["summary"]
                except Exception as error:
                    record["visualization_error"] = f"{type(error).__name__}: {error}"
                    if cfg.validation_vis.fail_on_error:
                        raise
            _write_log(log_stream, record)
            if (epoch + 1) % cfg.post_training.checkpoint_interval == 0 or epoch + 1 == cfg.post_training.epochs:
                _save_checkpoint(output / f"epoch_{epoch + 1:04d}.pt", model, optimizer, scheduler, epoch, cfg, checkpoint)
    metrics = {"stage": "post_training", "epochs": cfg.post_training.epochs, "history": history}
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def evaluate_post_training_checkpoint(cfg: AppConfig, checkpoint: str | Path) -> dict[str, Any]:
    """加载无Teacher/Predictor后训练检查点，生成测试指标、完整图表和固定闭环评估。"""
    device = check_training_environment(cfg)
    stats = NormalizationStats.load(_statistics_path(cfg))
    dataset = ByteDriveDataset(cfg, "test", stats)
    model = ByteDrivePolicy(cfg, (stats.flow_mean, stats.flow_std)).to(device)
    _freeze_predictor(model)
    epoch, source = _load_checkpoint(checkpoint, model)
    output = _output_path(cfg)
    validation_cfg = replace(
        cfg, validation_vis=replace(cfg.validation_vis, output=str(output / "test_visualizations")),
    )
    result = _evaluate(model, _loader(dataset, cfg, False), validation_cfg, int(source.get("epoch", 0)) + 1, stats)
    history = _history(output / "post_train.jsonl", int(source.get("epoch", 0)) + 1)
    result["visualization"] = generate_validation_visualizations(
        model, dataset, stats, history, int(source.get("epoch", 0)) + 1,
        validation_cfg, run_predictor=False,
    )["summary"]
    result["checkpoint"] = str(Path(checkpoint).resolve())
    return result


__all__ = ["evaluate_post_training_checkpoint", "post_train_model"]
