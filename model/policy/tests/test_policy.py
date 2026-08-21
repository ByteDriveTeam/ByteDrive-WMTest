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
from model.position import MODALITY_PREDICT, far_dense_depths, logarithmic_depths
from model.transformer import DenseResidualMixer


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
        rgb_time=torch.arange(rgb_frames).repeat(batch, 1) / cfg.model_data.rgb_hz,
        sensor_time=torch.arange(sensor_frames).repeat(batch, 1) / cfg.model_data.sensor_hz,
        future_time=cfg.model_data.history_seconds + torch.arange(future_frames).repeat(batch, 1) / cfg.model_data.sensor_hz,
        sensor_mask=torch.zeros(batch, sensor_tokens, dtype=torch.bool), task_patch_mask=torch.zeros(batch, sensor_tokens, dtype=torch.bool),
        behavior_valid=torch.ones(batch, future_frames, dtype=torch.bool), phase_target=torch.zeros(batch, dtype=torch.long),
        cache_hits=torch.zeros(batch, dtype=torch.long), cache_misses=torch.zeros(batch, dtype=torch.long),
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
    assert output.backbone_features.shape == (1, 2853, 16)
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


def test_dense_residual_logits_start_at_equal_softmax_weights() -> None:
    mixer = DenseResidualMixer(4)
    for logits in mixer.logits:
        assert torch.equal(logits, torch.zeros_like(logits))
        weights = torch.softmax(logits.float(), dim=0)
        assert torch.allclose(weights, torch.full_like(weights, 1.0 / len(weights)))


def test_predict_tokens_use_future_window_time_and_predict_modality() -> None:
    """PredictToken只携带2--4秒的实际窗口时间和Predict模态。"""
    cfg = _tiny_config()
    batch = _batch(cfg)
    model = ByteDrivePolicy(cfg).eval()
    observation, _ = model._observation_conditions(batch)
    predict = model._predict_conditions(batch)

    sensor_start = cfg.model_data.language_length + 1 + cfg.model.register_tokens
    assert observation.physical_time[:, sensor_start:].min() >= 0.0
    assert observation.physical_time[:, sensor_start:].max() <= cfg.model_data.history_seconds
    assert torch.all(predict.modality == MODALITY_PREDICT)
    assert torch.all(predict.physical_valid)
    assert predict.physical_time[0, 0] == cfg.model_data.history_seconds
    assert predict.physical_time[0, -1] == pytest.approx(
        cfg.model_data.history_seconds + cfg.model_data.future_seconds - 1 / cfg.model_data.sensor_hz,
    )
    assert not torch.any(predict.language_valid)
    assert not torch.any(predict.geometry_valid)


def test_camera_depth_sampling_has_opposite_density_biases() -> None:
    """Overview近疏远密，Wrist近密远疏，且都严格包含配置端点。"""
    cfg = _tiny_config()
    count = cfg.model.petr_depth_samples
    overview = far_dense_depths(*cfg.model.overview_depth_range, count, "cpu")
    wrist = logarithmic_depths(*cfg.model.wrist_depth_range, count, "cpu")

    assert overview[0] == pytest.approx(cfg.model.overview_depth_range[0])
    assert overview[-1] == pytest.approx(cfg.model.overview_depth_range[1])
    assert wrist[0] == pytest.approx(cfg.model.wrist_depth_range[0])
    assert wrist[-1] == pytest.approx(cfg.model.wrist_depth_range[1])
    assert overview.diff()[0] > overview.diff()[-1]
    assert wrist.diff()[0] < wrist.diff()[-1]


def test_rgb_uint8_normalization_runs_before_gpu_patch_embedding() -> None:
    """Dataset保留uint8传输，模型侧FP32归一化必须与显式公式一致。"""
    cfg = _tiny_config()
    model = ByteDrivePolicy(cfg).eval()
    images = torch.full((1, 1, 3, 16, 16), 127, dtype=torch.uint8)
    normalized = (images.float() / 255.0 - model.image_mean) / model.image_std
    expected = model.overview_embed(normalized.flatten(0, 1)).flatten(2).transpose(1, 2).reshape(1, 1, 1, -1)
    actual = model._embed_images(images, model.overview_embed)
    assert torch.allclose(actual, expected)


def test_policy_rejects_invalid_flow_statistics() -> None:
    cfg = _tiny_config()
    invalid_std = [1.0] * 23
    invalid_std[0] = 0.0
    with pytest.raises(ValueError):
        ByteDrivePolicy(cfg, ([0.0] * 23, invalid_std))
