"""计算逐层速度、最终积分、感知重建和阶段分类损失。

模块: train/objectives/objectives.py
依赖: torch, config, model.policy, train.objectives.checks
读取配置: loss.*, training.epochs, training.teacher_forcing_fraction
对外接口:
    - LossOutput
    - endpoint_weight(epoch, cfg) -> float
    - visible_reconstruction_weight(epoch, cfg) -> float
    - teacher_force_probability(epoch, cfg) -> float
    - compute_policy_losses(output, batch, teacher_features, epoch, cfg) -> LossOutput
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from config.schema import AppConfig
from model.policy import PolicyBatch, PolicyOutput
from train.objectives.checks import check_loss_shapes


@dataclass
class LossOutput:
    """保存总损失及可记录的各监督分量。"""

    total: torch.Tensor
    velocity: torch.Tensor
    endpoint: torch.Tensor
    reconstruction: torch.Tensor
    phase: torch.Tensor
    endpoint_weight: float
    visible_reconstruction_weight: float


ACTION_GROUPS = (tuple(range(7)), (7,), (8,))
TACTILE_GROUPS = ((9, 10, 11, 16, 17, 18), (12, 13, 19, 20), (14, 15, 21, 22))


def endpoint_weight(epoch: float, cfg: AppConfig) -> float:
    """在配置的epoch区间内线性插值最终积分监督权重。"""
    loss = cfg.loss
    duration = cfg.training.epochs * loss.endpoint_warmup_fraction
    progress = 1.0 if duration == 0 else min(max(epoch / duration, 0.0), 1.0)
    return loss.endpoint_start_weight + progress * (loss.endpoint_end_weight - loss.endpoint_start_weight)


def visible_reconstruction_weight(epoch: float, cfg: AppConfig) -> float:
    """将未掩码Token的重建权重从配置起点缓慢线性升至最终值。"""
    loss = cfg.loss
    duration = cfg.training.epochs * loss.visible_reconstruction_warmup_fraction
    progress = 1.0 if duration == 0 else min(max(epoch / duration, 0.0), 1.0)
    return loss.visible_reconstruction_start_weight + progress * (
        loss.visible_reconstruction_weight - loss.visible_reconstruction_start_weight
    )


def teacher_force_probability(epoch: float, cfg: AppConfig) -> float:
    """在教师强制区间内从1线性降至0。"""
    duration = cfg.training.epochs * cfg.training.teacher_forcing_fraction
    return max(1.0 - max(epoch, 0.0) / duration, 0.0)


def _weighted_mean(values: torch.Tensor, weights: list[float]) -> torch.Tensor:
    tensor = torch.as_tensor(weights, dtype=torch.float32, device=values.device)
    return (values * tensor).sum() / tensor.sum().clamp_min(torch.finfo(torch.float32).eps)


def _masked_group_losses(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    losses = []
    for group in groups:
        indices = torch.as_tensor(group, device=prediction.device)
        squared = (prediction.index_select(-1, indices) - target.index_select(-1, indices)).float().square()
        mask = valid.unsqueeze(-1).expand_as(squared)
        losses.append((squared * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(losses)


def _behavior_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, cfg: AppConfig) -> torch.Tensor:
    action = _weighted_mean(
        _masked_group_losses(prediction, target, valid, ACTION_GROUPS), cfg.loss.action_component_weights,
    )
    tactile = _weighted_mean(
        _masked_group_losses(prediction, target, valid, TACTILE_GROUPS), cfg.loss.tactile_component_weights,
    )
    return _weighted_mean(torch.stack((action, tactile)), [cfg.loss.action_weight, cfg.loss.tactile_weight])


def compute_policy_losses(
    output: PolicyOutput,
    batch: PolicyBatch,
    teacher_features: torch.Tensor,
    epoch: float,
    cfg: AppConfig,
) -> LossOutput:
    """计算FP32总损失，失败行为只从行为监督中排除。"""
    if batch.flow_target is None:
        raise ValueError("训练损失需要 flow_target")
    check_loss_shapes(output.velocities, output.final_flow, batch.flow_target)
    target_velocity = batch.flow_target.float() - output.flow_noise.float()
    expanded_target = target_velocity.unsqueeze(1).expand_as(output.velocities)
    velocity_layers = torch.stack([
        _behavior_loss(output.velocities[:, index].float(), expanded_target[:, index], batch.behavior_valid, cfg)
        for index in range(output.velocities.shape[1])
    ])
    velocity = _weighted_mean(velocity_layers, cfg.loss.velocity_layer_weights)
    endpoint = _behavior_loss(output.final_flow.float(), batch.flow_target.float(), batch.behavior_valid, cfg)
    reconstruction_squared = (output.predictor_features.float() - teacher_features.detach().float()).square().mean(-1)
    visible_weight = visible_reconstruction_weight(epoch, cfg)
    reconstruction_weights = torch.where(
        batch.sensor_mask,
        torch.tensor(cfg.loss.masked_reconstruction_weight, device=batch.state.device),
        torch.tensor(visible_weight, device=batch.state.device),
    )
    reconstruction = (reconstruction_squared * reconstruction_weights).sum() / reconstruction_weights.sum().clamp_min(1)
    phase = F.cross_entropy(
        output.phase_logits.float(), batch.phase_target.long(), ignore_index=-100,
        label_smoothing=cfg.loss.phase_label_smoothing,
    )
    if torch.all(batch.phase_target == -100):
        phase = output.phase_logits.float().sum() * 0.0
    weight = endpoint_weight(epoch, cfg)
    total = (
        cfg.loss.velocity_weight * velocity
        + cfg.loss.endpoint_weight * weight * endpoint
        + cfg.loss.reconstruction_weight * reconstruction
        + cfg.loss.phase_weight * phase
    )
    return LossOutput(total, velocity, endpoint, reconstruction, phase, weight, visible_weight)


__all__ = [
    "LossOutput", "compute_policy_losses", "endpoint_weight", "teacher_force_probability",
    "visible_reconstruction_weight",
]
