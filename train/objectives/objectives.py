"""计算逐层速度、最终积分、感知重建和阶段分类损失。

模块: train/objectives/objectives.py
依赖: torch, config, model.policy, train.objectives.checks
读取配置: model.masked_loss_weight, model.visible_loss_weight,
    training.epochs, training.endpoint_warmup_fraction, training.teacher_forcing_fraction
对外接口:
    - LossOutput
    - endpoint_weight(epoch, cfg) -> float
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


ACTION_GROUPS = (tuple(range(7)), (7,), (8,))
TACTILE_GROUPS = ((9, 10, 11, 16, 17, 18), (12, 13, 19, 20), (14, 15, 21, 22))


def endpoint_weight(epoch: float, cfg: AppConfig) -> float:
    """在前20% epoch内将最终积分监督从0线性升至1。"""
    duration = cfg.training.epochs * cfg.training.endpoint_warmup_fraction
    return min(max(epoch / duration, 0.0), 1.0)


def teacher_force_probability(epoch: float, cfg: AppConfig) -> float:
    """在教师强制区间内从1线性降至0。"""
    duration = cfg.training.epochs * cfg.training.teacher_forcing_fraction
    return max(1.0 - max(epoch, 0.0) / duration, 0.0)


def _masked_group_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    losses = []
    for group in groups:
        indices = torch.tensor(group, device=prediction.device)
        squared = (prediction.index_select(-1, indices) - target.index_select(-1, indices)).float().square()
        mask = valid.unsqueeze(-1)
        mask = mask.expand(*squared.shape[:-1], 1)
        denominator = (mask.sum() * squared.shape[-1]).clamp_min(1)
        losses.append((squared * mask).sum() / denominator)
    return torch.stack(losses).mean()


def _behavior_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    action = _masked_group_loss(prediction, target, valid, ACTION_GROUPS)
    tactile = _masked_group_loss(prediction, target, valid, TACTILE_GROUPS)
    return 0.5 * (action + tactile)


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
    velocity_valid = batch.behavior_valid.unsqueeze(1)
    velocity = _behavior_loss(output.velocities.float(), expanded_target, velocity_valid)
    endpoint = _behavior_loss(output.final_flow.float(), batch.flow_target.float(), batch.behavior_valid)
    reconstruction_squared = (output.predictor_features.float() - teacher_features.detach().float()).square().mean(-1)
    reconstruction_weights = torch.where(
        batch.sensor_mask,
        torch.tensor(cfg.model.masked_loss_weight, device=batch.state.device),
        torch.tensor(cfg.model.visible_loss_weight, device=batch.state.device),
    )
    reconstruction = (reconstruction_squared * reconstruction_weights).sum() / reconstruction_weights.sum().clamp_min(1)
    phase = F.cross_entropy(output.phase_logits.float(), batch.phase_target.long(), ignore_index=-100)
    if torch.all(batch.phase_target == -100):
        phase = output.phase_logits.float().sum() * 0.0
    weight = endpoint_weight(epoch, cfg)
    total = velocity + weight * endpoint + reconstruction + phase
    return LossOutput(total, velocity, endpoint, reconstruction, phase, weight)


__all__ = ["LossOutput", "compute_policy_losses", "endpoint_weight", "teacher_force_probability"]
