"""验证触觉摘要、失败门控、语言定长与75%掩码。

模块: data/model_dataset/tests/test_model_dataset.py
依赖: dataclasses, numpy, pytest, torch, config, data.model_dataset, model.policy
读取配置: model_data.*, model.*, data_collector.render.cameras,
    data_collector.sensors.tactile_resolution
对外接口: 无（由pytest发现测试函数）
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from config import load_config
from config import PROJECT_ROOT
from data.model_dataset import (
    ClosedLanguageTokenizer, behavior_validity, build_sensor_mask, canonical_phase, tactile_summary,
)
from data.model_dataset.checks import check_dataset_path
from data.model_dataset.model_dataset import (
    _nearest_time_indices, _resolve_normalization_stats, _RunningMoments, _sampling_times,
    _shared_finger_statistics,
)
from model import sensor_token_counts


def test_tactile_summary_center_and_empty() -> None:
    maps = torch.zeros(1, 2, 3, 32, 32)
    assert torch.equal(tactile_summary(maps), torch.zeros(1, 2, 7))
    maps[0, 0, 0, 16, 24] = 8.0
    maps[0, 0, 1, 16, 24] = 2.0
    summary = tactile_summary(maps)
    assert summary.shape == (1, 2, 7)
    assert summary[0, 0, 0] > 0
    assert summary[0, 0, 3] > 0
    assert torch.allclose(summary[0, 0, 5:], torch.zeros(2), atol=1.0e-6)


def test_failure_attempt_and_phase_mapping() -> None:
    cfg = load_config()
    phases = ["HOME_BEFORE_object_0", "APPROACH_object_0", "REOPEN_RETRY_1_object_0", "HOME_BEFORE_object_0_RETRY_1"]
    assert behavior_validity(phases).tolist() == [False, False, True, True]
    assert canonical_phase("REOPEN_RETRY_1_object_0", cfg.model.phase_names) == "OPEN"
    assert canonical_phase("APPROACH_object_0_RETRY_1", cfg.model.phase_names) == "APPROACH"


def test_language_and_mask_are_fixed() -> None:
    cfg = load_config()
    ids, valid = ClosedLanguageTokenizer(cfg.model_data.language_length, cfg.model_data.language_vocabulary).encode("PICK red box AS object_0.")
    assert ids.shape == valid.shape == (40,)
    assert valid.sum() == 8
    counts = sensor_token_counts(cfg)
    required = torch.zeros(sum(counts), dtype=torch.bool)
    required[:10] = True
    gradient = torch.linspace(0, 1, sum(counts))
    structured = replace(cfg, model=replace(
        cfg.model, task_priority_sample_probability=1.0, mask_group_probability=0.0,
    ))
    mask = build_sensor_mask(required, gradient, counts, structured, 7)
    assert mask.sum() == round(sum(counts) * cfg.model.mask_ratio)
    assert mask[:10].all()


def test_temporal_gradient_and_cross_modal_compensation() -> None:
    cfg = load_config()
    counts = sensor_token_counts(cfg)
    task = torch.zeros(sum(counts), dtype=torch.bool)
    gradient = torch.zeros(sum(counts))
    gradient[:10] = 1
    weighted = replace(cfg, model=replace(
        cfg.model, task_priority_sample_probability=1.0, mask_group_probability=0.0,
    ))
    masks = torch.stack([build_sensor_mask(task, gradient, counts, weighted, seed) for seed in range(100)])
    assert masks[:, :10].float().mean() > masks[:, 10:].float().mean()
    grouped = replace(cfg, model=replace(
        cfg.model, task_priority_sample_probability=1.0, mask_group_probability=1.0,
    ))
    grouped_mask = build_sensor_mask(task, gradient, counts, grouped, 3)
    rgb_count = counts[0] + counts[1]
    assert grouped_mask[:rgb_count].float().mean() < cfg.model.mask_ratio
    task[:600] = True
    task_grouped_mask = build_sensor_mask(task, gradient, counts, grouped, 3)
    assert task_grouped_mask.sum() == round(sum(counts) * cfg.model.mask_ratio)
    assert task_grouped_mask[:600].all()


def test_sampling_uses_uniform_physical_time_grid() -> None:
    cfg = load_config()
    rgb_time, sensor_time, future_time = _sampling_times(cfg)
    assert np.allclose(np.diff(rgb_time), 1 / cfg.model_data.rgb_hz)
    assert np.allclose(np.diff(sensor_time), 1 / cfg.model_data.sensor_hz)
    assert np.allclose(np.diff(future_time), 1 / cfg.model_data.sensor_hz)
    stored_times = np.arange(251, dtype=np.float64) / 50
    rgb_target, sensor_target = 2.0 + rgb_time, 2.0 + sensor_time
    rgb_selected = _nearest_time_indices(stored_times, rgb_target)
    sensor_selected = _nearest_time_indices(stored_times, sensor_target)
    assert np.all(np.diff(rgb_selected) == 10)
    assert np.all(np.diff(sensor_selected) == 2)
    assert np.allclose(stored_times[rgb_selected], rgb_target)
    assert np.allclose(stored_times[sensor_selected], sensor_target)


def test_left_and_right_fingers_share_statistics() -> None:
    moments = _RunningMoments(23)
    values = np.zeros((2, 23), dtype=np.float32)
    values[:, 9:16], values[:, 16:23] = 1.0, 3.0
    moments.update(values)
    mean, std = _shared_finger_statistics(moments, 1.0e-6)
    assert np.allclose(mean, 2.0)
    assert np.allclose(std, 1.0)


def test_normalized_dataset_rejects_missing_statistics() -> None:
    cfg = load_config()
    missing = replace(
        cfg, model_data=replace(cfg.model_data, statistics="train/output/tests_missing_statistics.json"),
    )
    with pytest.raises(FileNotFoundError):
        _resolve_normalization_stats(missing, None, True)
    identity = _resolve_normalization_stats(missing, None, False)
    assert identity.dataset_schema == "1.2.0"


def test_dataset_can_be_read_from_outside_project() -> None:
    """项目边界仅约束产物写入，不约束只读数据集来源。"""
    check_dataset_path(PROJECT_ROOT.parent)
    with pytest.raises(ValueError, match="已存在目录"):
        check_dataset_path(PROJECT_ROOT / "tests_missing_dataset_directory")
