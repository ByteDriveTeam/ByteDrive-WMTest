"""验证在线批次构造、传感器画面和MP4编码。

模块: vis/closed_loop_validation/tests/test_closed_loop_validation.py
依赖: imageio-ffmpeg, numpy, pytest, config, data.data_collector.records, data.model_dataset,
    vis.closed_loop_validation
读取配置: model_data.*, model.*, data_collector.render.*, data_vis.*,
    validation_vis.closed_loop_*
对外接口: 无（由pytest发现测试函数）
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import config as config_module
import vis.closed_loop_validation.closed_loop_validation as closed_loop_module
import vis.closed_loop_validation.checks.closed_loop_validation_checks as checks_module
from config import PROJECT_ROOT, configure_mujoco_rendering, load_config
from data.data_collector.records import FrameRecord
from data.model_dataset import NormalizationStats
from vis.closed_loop_validation.checks import check_closed_loop_rendering
from vis.closed_loop_validation.closed_loop_validation import (
    _build_policy_batch, _compose_video_frame, _write_mp4, _write_sensor_archive,
)


def _frame(index: int) -> FrameRecord:
    cameras = {
        "overview": {
            "rgb": np.full((128, 160, 3), index % 255, dtype=np.uint8),
            "K": np.eye(3, dtype=np.float32), "T_base_camera": np.eye(4, dtype=np.float32),
        },
        "wrist": {
            "rgb": np.full((96, 96, 3), index % 255, dtype=np.uint8),
            "K": np.eye(3, dtype=np.float32), "T_base_camera": np.eye(4, dtype=np.float32),
        },
    }
    ee = {
        "position_base": np.zeros(3, dtype=np.float32),
        "quaternion_base_wxyz": np.asarray([1, 0, 0, 0], dtype=np.float32),
        "linear_velocity_base": np.zeros(3, dtype=np.float32),
        "angular_velocity_base": np.zeros(3, dtype=np.float32),
        "linear_acceleration_base": np.zeros(3, dtype=np.float32),
        "angular_acceleration_base": np.zeros(3, dtype=np.float32),
    }
    robot = {
        "frames": {"ee_site": ee}, "gripper_width": np.full(2, 0.04, dtype=np.float32),
        "actuator_control": np.zeros(8, dtype=np.float32), "actuator_force": np.zeros(8, dtype=np.float32),
        "joint_position": np.zeros(7, dtype=np.float32), "joint_velocity": np.zeros(7, dtype=np.float32),
    }
    tactile = {
        "force_maps": {side: np.zeros((32, 32, 3), dtype=np.float32) for side in ("left", "right")},
        "patch_geometry_base": np.zeros((2, 16, 3), dtype=np.float32),
    }
    return FrameRecord(index, index * 0.02, "TEST", {}, robot, {}, "", np.zeros(1), cameras, [], tactile)


def test_online_history_builds_checkpoint_compatible_policy_batch() -> None:
    cfg = load_config()
    frames = [_frame(index) for index in range(99)]
    batch = _build_policy_batch(frames, "PICK red cylinder AS object_0; PLACE object_0 IN center_zone.", NormalizationStats.identity(cfg), cfg)
    assert batch.overview_rgb.shape == (1, 10, 3, 128, 160)
    assert batch.wrist_rgb.shape == (1, 10, 3, 96, 96)
    assert batch.tactile.shape == (1, 50, 2, 3, 32, 32)
    assert batch.state.shape == (1, 50, 37)
    assert batch.tactile_geometry.shape == (1, 50, 2, 16, 3)
    assert not batch.sensor_mask.any()
    assert batch.flow_target is None


def test_sensor_frame_has_even_rgb_video_shape() -> None:
    frame = _compose_video_frame(_frame(0), "timeout", load_config())
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0


def test_linux_closed_loop_requires_configured_egl_device(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    monkeypatch.setattr(config_module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(checks_module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID", raising=False)
    with pytest.raises(RuntimeError, match="MUJOCO_GL"):
        check_closed_loop_rendering(cfg)
    configure_mujoco_rendering(cfg)
    check_closed_loop_rendering(cfg)
    assert checks_module.os.environ["MUJOCO_GL"] == cfg.model_data.replay_cache.linux_render_backend
    assert checks_module.os.environ["MUJOCO_EGL_DEVICE_ID"] == str(cfg.model_data.replay_cache.linux_egl_device_id)


def test_sensor_archive_and_mp4_are_written_without_system_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    frames = [_frame(index) for index in range(6)]
    output = PROJECT_ROOT / "train" / "output" / "tests_closed_loop"
    video, sensors = output / "test.mp4", output / "test.npz"
    try:
        monkeypatch.setattr(closed_loop_module.shutil, "which", lambda _: None)
        output.mkdir(parents=True, exist_ok=True)
        _write_sensor_archive(frames, sensors)
        _write_mp4(frames, video, "timeout", cfg)
        assert video.stat().st_size > 0 and sensors.stat().st_size > 0
    finally:
        video.unlink(missing_ok=True)
        sensors.unlink(missing_ok=True)
        if output.exists() and not any(output.iterdir()):
            output.rmdir()
