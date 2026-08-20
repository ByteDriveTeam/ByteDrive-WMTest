"""验证两个无位置RegisterToken、23维流与Predictor边界。

模块: model/policy/tests/test_policy.py
依赖: dataclasses, pytest, torch, config, model.policy
读取配置: model.*, model_data.*, data_collector.render.cameras,
    data_collector.sensors.tactile_resolution
对外接口: 无（由pytest发现测试函数）
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import load_config
from model import ByteDrivePolicy, PolicyBatch, sensor_token_counts


def _tiny_config():
    cfg = load_config()
    return replace(cfg, model=replace(
        cfg.model, width=16, heads=2, backbone_layers=1, predictor_layers=1,
        ffn_width=32, position_hidden=16, flow_hidden=16,
    ))


def _batch(cfg=None) -> PolicyBatch:
    cfg = cfg or load_config()
    batch = 1
    rgb_frames = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
    sensor_frames = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
    future_frames = round(cfg.model_data.future_seconds * cfg.model_data.sensor_hz)
    sensor_tokens = sum(sensor_token_counts(cfg))
    identity_k = torch.eye(3).view(1, 1, 3, 3).repeat(batch, rgb_frames, 1, 1)
    identity_t = torch.eye(4).view(1, 1, 4, 4).repeat(batch, rgb_frames, 1, 1)
    bounds = torch.tensor([[[-2.0, 2.0], [-2.0, 2.0], [-1.0, 3.0]]])
    return PolicyBatch(
        overview_rgb=torch.zeros(batch, rgb_frames, 3, 128, 160), wrist_rgb=torch.zeros(batch, rgb_frames, 3, 96, 96),
        tactile=torch.zeros(batch, sensor_frames, 2, 3, 32, 32), state=torch.zeros(batch, sensor_frames, 37),
        language_ids=torch.zeros(batch, 40, dtype=torch.long), language_valid=torch.zeros(batch, 40, dtype=torch.bool),
        overview_intrinsics=identity_k, overview_transform=identity_t, wrist_intrinsics=identity_k, wrist_transform=identity_t,
        tactile_geometry=torch.zeros(batch, sensor_frames, 2, 16, 3), state_geometry=torch.zeros(batch, sensor_frames, 3), coordinate_bounds=bounds,
        rgb_time=(torch.arange(rgb_frames) - (rgb_frames - 1)).repeat(batch, 1) / cfg.model_data.rgb_hz,
        sensor_time=(torch.arange(sensor_frames) - (sensor_frames - 1)).repeat(batch, 1) / cfg.model_data.sensor_hz,
        future_time=torch.arange(1, future_frames + 1).repeat(batch, 1) / cfg.model_data.sensor_hz,
        sensor_mask=torch.zeros(batch, sensor_tokens, dtype=torch.bool), task_patch_mask=torch.zeros(batch, sensor_tokens, dtype=torch.bool),
        behavior_valid=torch.ones(batch, future_frames, dtype=torch.bool), phase_target=torch.zeros(batch, dtype=torch.long),
        flow_target=torch.zeros(batch, future_frames, 23),
    )


def test_policy_shapes_and_register_position() -> None:
    cfg = _tiny_config()
    batch = _batch(cfg)
    model = ByteDrivePolicy(cfg).eval()
    assert model.register_token.shape[1] == 2
    positions, _ = model._observation_conditions(batch)
    encoded = model.position(positions, 0)
    assert torch.equal(encoded[:, 41:43], torch.zeros_like(encoded[:, 41:43]))
    assert not model.predictor[0].ffn.enabled
    with torch.no_grad():
        output = model(batch, flow_noise=torch.zeros(1, 50, 23))
    assert output.velocities.shape == (1, 1, 50, 23)
    assert output.final_flow.shape == (1, 50, 23)
    assert output.predictor_features.shape == (1, 2810, 16)
    assert output.velocities.dtype == output.final_flow.dtype == output.predictor_features.dtype == torch.float32
    assert model.overview_embed.weight.dtype == model.velocity_decoder.first.weight.dtype == torch.float32
    assert model.velocity_decoder.modulation.in_features == cfg.model.backbone_layers
    assert model.velocity_decoder.second.out_features == 23
    assert model.backbone[0].ffn.enabled and not model.predictor[0].ffn.enabled
    with pytest.raises(RuntimeError):
        model.predict(batch, flow_noise=torch.zeros(1, 50, 23))
    physical_model = ByteDrivePolicy(cfg, ([0.0] * 23, [1.0] * 23)).eval()
    with torch.no_grad():
        prediction = physical_model.predict(batch, flow_noise=torch.zeros(1, 50, 23))
    assert prediction["actions"].shape == (1, 50, 9)
    assert prediction["tactile_summary"].shape == (1, 50, 2, 7)
    assert prediction["phase"].shape == (1, 12)
    assert torch.all((prediction["actions"][..., 8] == -1) | (prediction["actions"][..., 8] == 1))


def test_policy_rejects_invalid_flow_statistics() -> None:
    cfg = _tiny_config()
    invalid_std = [1.0] * 23
    invalid_std[0] = 0.0
    with pytest.raises(ValueError):
        ByteDrivePolicy(cfg, ([0.0] * 23, invalid_std))
