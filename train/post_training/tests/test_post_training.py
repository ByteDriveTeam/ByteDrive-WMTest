"""验证后训练完整行为监督、完整观测和Teacher/Predictor排除。

模块: train/post_training/tests/test_post_training.py
依赖: dataclasses, pathlib, torch, config, model.policy, train.objectives,
    train.post_training
读取配置: loss.*, training.learning_rate, post_training.*
对外接口: 无（由pytest发现测试函数）
"""

from __future__ import annotations

from dataclasses import replace

import torch

from config import PROJECT_ROOT
from model.policy import ByteDrivePolicy, PolicyOutput
from model.policy.tests.test_policy import _batch, _tiny_config
from train.objectives import compute_post_training_losses
from train.post_training.post_training import _freeze_predictor, _full_observation, _load_checkpoint


def _output(velocities: torch.Tensor, final_flow: torch.Tensor, flow_noise: torch.Tensor) -> PolicyOutput:
    batch, _, steps, _ = velocities.shape
    return PolicyOutput(
        velocities=velocities, final_flow=final_flow,
        phase_logits=torch.zeros(batch, 12), predictor_features=torch.empty(batch, 0, 16),
        observation_features=torch.empty(batch, 0, 16), flow_noise=flow_noise,
        backbone_features=torch.empty(batch, 1, 16),
    )


def test_post_training_supervises_tactile_summary_at_every_velocity_layer() -> None:
    cfg = _tiny_config()
    cfg = replace(cfg, loss=replace(cfg.loss, velocity_layer_weights=[0.0, 1.0]))
    batch = _batch(cfg)
    steps = batch.flow_target.shape[1]
    batch.flow_target[..., 9] = 1.0
    flow_noise = torch.zeros_like(batch.flow_target)
    first_only = torch.zeros(1, 2, steps, 23)
    first_only[:, 0, :, 9] = 1.0
    second_only = torch.zeros_like(first_only)
    second_only[:, 1, :, 9] = 1.0
    first_loss = compute_post_training_losses(_output(first_only, torch.zeros_like(flow_noise), flow_noise), batch, cfg)
    second_loss = compute_post_training_losses(_output(second_only, torch.zeros_like(flow_noise), flow_noise), batch, cfg)
    assert first_loss.velocity > 0
    assert second_loss.velocity == 0
    assert first_loss.reconstruction == first_loss.visreg == 0


def test_post_training_uses_full_observation_without_running_predictor() -> None:
    cfg, batch = _tiny_config(), _batch(_tiny_config())
    batch.sensor_mask.fill_(True)
    model = ByteDrivePolicy(cfg).eval()

    def fail_predictor(*_args, **_kwargs):
        raise AssertionError("后训练不应运行Predictor")

    model._run_predictor = fail_predictor
    with torch.no_grad():
        output = model(
            _full_observation(batch), 0.0,
            flow_noise=torch.zeros_like(batch.flow_target), run_predictor=False,
        )
    assert not batch.sensor_mask.any()
    assert output.predictor_features.shape[1] == 0


def test_post_training_loads_student_but_excludes_predictor_and_teacher() -> None:
    cfg = _tiny_config()
    source_model = ByteDrivePolicy(cfg)
    source_model.cls_token.data.fill_(3.0)
    source_model.mask_token.data.fill_(7.0)
    for parameter in source_model.predictor.parameters():
        parameter.data.fill_(7.0)
    target_model = ByteDrivePolicy(cfg)
    original_mask = target_model.mask_token.detach().clone()
    original_predictor = next(target_model.predictor.parameters()).detach().clone()
    checkpoint = PROJECT_ROOT / "train" / "output" / "tests_post_training_source.pt"
    try:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": source_model.state_dict(), "teacher": source_model.state_dict(), "epoch": 2}, checkpoint)
        _freeze_predictor(target_model)
        start_epoch, _ = _load_checkpoint(checkpoint, target_model)
        assert start_epoch == 0
        assert torch.all(target_model.cls_token == 3.0)
        assert torch.equal(target_model.mask_token, original_mask)
        assert torch.equal(next(target_model.predictor.parameters()), original_predictor)
        assert all(not parameter.requires_grad for parameter in target_model.predictor.parameters())
    finally:
        checkpoint.unlink(missing_ok=True)

