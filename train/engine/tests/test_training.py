"""验证失败行为屏蔽、感知重建保留和epoch调度。

模块: train/engine/tests/test_training.py
依赖: pathlib, torch, config, model.policy, train.engine, train.objectives
读取配置: training.epochs, training.teacher_forcing_fraction,
    training.learning_rate, loss.*
对外接口: 无（由pytest发现测试函数）
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import PROJECT_ROOT, load_config
from model.policy import ByteDrivePolicy, PolicyOutput, sensor_token_counts
from model.policy.tests.test_policy import _batch, _tiny_config
from train.engine import create_ema_teacher, load_checkpoint, save_checkpoint, update_ema
from train.objectives import compute_policy_losses, endpoint_weight, teacher_force_probability


def test_epoch_schedules() -> None:
    cfg = load_config()
    assert endpoint_weight(0, cfg) == 0
    assert endpoint_weight(cfg.training.epochs * cfg.loss.endpoint_warmup_fraction, cfg) == cfg.loss.endpoint_end_weight
    assert teacher_force_probability(0, cfg) == 1
    assert teacher_force_probability(cfg.training.epochs * cfg.training.teacher_forcing_fraction, cfg) == 0


def test_ema_absorbs_five_ten_thousandths_per_update() -> None:
    cfg = _tiny_config()
    student = ByteDrivePolicy(cfg)
    teacher = create_ema_teacher(student)
    teacher.cls_token.zero_()
    student.cls_token.data.fill_(1.0)
    update_ema(teacher, student, cfg.model.ema_decay)
    expected = torch.full_like(teacher.cls_token, 1.0 - cfg.model.ema_decay)
    assert torch.allclose(teacher.cls_token, expected)


def test_failed_behavior_does_not_change_behavior_loss() -> None:
    cfg, batch = load_config(), _batch()
    sensor_tokens = sum(sensor_token_counts(cfg))
    batch.behavior_valid[:, 0] = False
    output = PolicyOutput(
        velocities=torch.zeros(1, 12, 50, 23), final_flow=torch.zeros(1, 50, 23),
        phase_logits=torch.zeros(1, 12), predictor_features=torch.zeros(1, sensor_tokens, 384),
        observation_features=torch.zeros(1, sensor_tokens, 384), flow_noise=torch.zeros(1, 50, 23),
    )
    teacher = torch.ones(1, sensor_tokens, 384)
    baseline = compute_policy_losses(output, batch, teacher, 0, cfg)
    batch.flow_target[:, 0] = 1000
    changed = compute_policy_losses(output, batch, teacher, 0, cfg)
    assert torch.allclose(baseline.velocity, changed.velocity)
    assert torch.allclose(baseline.endpoint, changed.endpoint)
    assert baseline.reconstruction > 0
    batch.phase_target.fill_(-100)
    failed = compute_policy_losses(output, batch, teacher, 0, cfg)
    assert failed.phase == 0
    assert failed.reconstruction == changed.reconstruction


def test_all_loss_levels_are_configurable() -> None:
    """总项、行为组、子项和逐层权重均来自集中配置。"""
    cfg, batch = load_config(), _batch()
    sensor_tokens = sum(sensor_token_counts(cfg))
    output = PolicyOutput(
        velocities=torch.ones(1, 12, 50, 23), final_flow=torch.ones(1, 50, 23),
        phase_logits=torch.zeros(1, 12), predictor_features=torch.zeros(1, sensor_tokens, 384),
        observation_features=torch.zeros(1, sensor_tokens, 384), flow_noise=torch.zeros(1, 50, 23),
    )
    teacher = torch.ones(1, sensor_tokens, 384)
    only_reconstruction = replace(
        cfg,
        loss=replace(cfg.loss, velocity_weight=0.0, endpoint_weight=0.0, phase_weight=0.0, reconstruction_weight=2.5),
    )
    result = compute_policy_losses(output, batch, teacher, cfg.training.epochs, only_reconstruction)
    assert torch.allclose(result.total, 2.5 * result.reconstruction)


def test_checkpoint_restores_next_epoch() -> None:
    cfg = _tiny_config()
    model = ByteDrivePolicy(cfg)
    teacher = create_ema_teacher(model)
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = PROJECT_ROOT / "train" / "output" / "tests_runtime_checkpoint.pt"
    original = model.cls_token.detach().clone()
    try:
        save_checkpoint(checkpoint, model, teacher, optimizer, scheduler, 6, cfg)
        model.cls_token.data.add_(1)
        assert load_checkpoint(checkpoint, model, teacher, optimizer, scheduler) == 7
        assert torch.equal(model.cls_token, original)
    finally:
        checkpoint.unlink(missing_ok=True)
