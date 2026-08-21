"""验证失败行为屏蔽、感知重建保留和epoch调度。

模块: train/engine/tests/test_training.py
依赖: pathlib, torch, config, model.policy, train.engine, train.objectives
读取配置: training.epochs, training.teacher_forcing_fraction,
    training.learning_rate, loss.*
对外接口: 无（由pytest发现测试函数）
"""

from __future__ import annotations

from dataclasses import replace
import json

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import PROJECT_ROOT, load_config
from model.policy import ByteDrivePolicy, PolicyOutput, TeacherOutput, sensor_token_counts
from model.policy.tests.test_policy import _batch, _tiny_config
from train.engine import constantization_metrics, create_ema_teacher, load_checkpoint, save_checkpoint, update_ema
from train.engine.engine import _load_epoch_history
from train.objectives import (
    compute_policy_losses, endpoint_weight, teacher_force_probability,
    visible_reconstruction_weight, visreg_loss,
)


def test_epoch_schedules() -> None:
    cfg = load_config()
    assert endpoint_weight(0, cfg) == 0
    assert endpoint_weight(cfg.training.epochs * cfg.loss.endpoint_warmup_fraction, cfg) == cfg.loss.endpoint_end_weight
    assert teacher_force_probability(0, cfg) == 1
    assert teacher_force_probability(cfg.training.epochs * cfg.training.teacher_forcing_fraction, cfg) == 0
    assert visible_reconstruction_weight(0, cfg) == cfg.loss.visible_reconstruction_start_weight
    assert visible_reconstruction_weight(
        cfg.training.epochs * cfg.loss.visible_reconstruction_warmup_fraction, cfg,
    ) == cfg.loss.visible_reconstruction_weight


def test_constantization_monitor_distinguishes_constant_and_varying_outputs() -> None:
    constant = torch.ones(2, 3, 4)
    varying = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    output = PolicyOutput(
        velocities=constant.unsqueeze(1), final_flow=constant,
        phase_logits=torch.zeros(2, 12), predictor_features=constant,
        observation_features=constant, flow_noise=torch.zeros_like(constant),
        backbone_features=constant,
    )
    collapsed = constantization_metrics(output, TeacherOutput(constant, constant[:, 1]), 1)
    assert all(value == 0 for value in collapsed.values())
    output.velocities = varying.unsqueeze(1)
    output.final_flow = varying
    output.predictor_features = varying
    output.observation_features = varying
    output.backbone_features = varying
    healthy = constantization_metrics(output, TeacherOutput(varying, varying[:, 1]), 1)
    assert all(value > 0.1 for value in healthy.values())


def test_ema_absorbs_five_ten_thousandths_per_update() -> None:
    cfg = _tiny_config()
    student = ByteDrivePolicy(cfg)
    teacher = create_ema_teacher(student)
    teacher.cls_token.zero_()
    student.cls_token.data.fill_(1.0)
    update_ema(teacher, student, cfg.model.ema_decay)
    expected = torch.full_like(teacher.cls_token, 1.0 - cfg.model.ema_decay)
    assert torch.allclose(teacher.cls_token, expected)


def test_ema_teacher_is_fully_outside_autograd() -> None:
    cfg = _tiny_config()
    batch = _batch(cfg)
    batch.state.requires_grad_(True)
    student = ByteDrivePolicy(cfg)
    teacher = create_ema_teacher(student)
    teacher_output = teacher.encode_teacher(batch)
    assert not teacher.training
    assert not teacher_output.observation_features.requires_grad
    assert not teacher_output.cls_features.requires_grad
    assert teacher_output.observation_features.grad_fn is None
    assert teacher_output.cls_features.grad_fn is None
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    output = student(batch, flow_noise=torch.zeros_like(batch.flow_target))
    loss = compute_policy_losses(output, batch, teacher_output, 0, cfg)
    loss.total.backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in student.parameters())


def test_visreg_uses_cls_batch_distribution_and_backpropagates() -> None:
    cfg = load_config()
    student_cls = torch.randn(6, cfg.model.width, requires_grad=True)
    teacher_cls = torch.randn_like(student_cls, requires_grad=True)
    total, invariance, regularization, scale, shape, center = visreg_loss(student_cls, teacher_cls, cfg)
    assert all(torch.isfinite(value) for value in (
        total, invariance, regularization, scale, shape, center,
    ))
    mix = cfg.loss.visreg_regularization_mix
    assert torch.allclose(total, (1.0 - mix) * invariance + mix * regularization)
    total.backward()
    assert student_cls.grad is not None and torch.isfinite(student_cls.grad).all()
    assert teacher_cls.grad is None
    equal_view = visreg_loss(student_cls.detach(), student_cls.detach(), cfg)
    assert equal_view[1] == 0


def test_failed_behavior_does_not_change_behavior_loss() -> None:
    cfg, batch = load_config(), _batch()
    sensor_tokens = sum(sensor_token_counts(cfg))
    batch.behavior_valid[:, 0] = False
    batch.sensor_mask[:, 0] = True
    output = PolicyOutput(
        velocities=torch.zeros(1, 12, 50, 23), final_flow=torch.zeros(1, 50, 23),
        phase_logits=torch.zeros(1, 12), predictor_features=torch.zeros(1, sensor_tokens, 384),
        observation_features=torch.zeros(1, sensor_tokens, 384), flow_noise=torch.zeros(1, 50, 23),
        backbone_features=torch.zeros(1, cfg.model_data.language_length + 1, 384),
    )
    teacher = TeacherOutput(torch.ones(1, sensor_tokens, 384), torch.zeros(1, 384))
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
        backbone_features=torch.zeros(1, cfg.model_data.language_length + 1, 384),
    )
    teacher = TeacherOutput(torch.ones(1, sensor_tokens, 384), torch.zeros(1, 384))
    only_reconstruction = replace(
        cfg,
        loss=replace(
            cfg.loss, velocity_weight=0.0, endpoint_weight=0.0, phase_weight=0.0,
            visreg_weight=0.0, reconstruction_weight=2.5,
        ),
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


def test_resume_loads_previous_epoch_history_for_curves() -> None:
    path = PROJECT_ROOT / "train" / "output" / "tests_runtime_history.jsonl"
    records = [
        {"event": "train_start"},
        {"event": "epoch", "epoch": 0, "train_total": 2.0},
        {"event": "epoch", "epoch": 1, "train_total": 1.0},
    ]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        assert _load_epoch_history(path, 1) == [records[1]]
    finally:
        path.unlink(missing_ok=True)
