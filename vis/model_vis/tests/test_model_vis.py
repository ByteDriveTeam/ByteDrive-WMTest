"""验证骨干特征热力图、PCA RGB、动作曲线和原始数组输出。

模块: vis/model_vis/tests/test_model_vis.py
依赖: pathlib, shutil, numpy, pillow, config, vis.model_vis
读取配置: model_vis.*
对外接口: 无
"""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from config import load_config
from vis.model_vis import render_model_visualization
from vis.model_vis.run import _parser


RUNTIME = Path(__file__).resolve().parent / "_runtime"


def test_checkpoint_argument_is_optional() -> None:
    assert _parser().parse_args([]).checkpoint is None
    assert _parser().parse_args(["checkpoint.pt"]).checkpoint == "checkpoint.pt"


def test_model_visualization_writes_png_and_raw_arrays() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    try:
        cfg = load_config()
        features = np.linspace(-2.0, 2.0, 160, dtype=np.float32).reshape(20, 8)
        target = np.zeros((50, 9), dtype=np.float32)
        predicted = np.sin(np.linspace(0, np.pi, 50, dtype=np.float32))[:, None] * np.ones((1, 9), dtype=np.float32)
        result = render_model_visualization(
            features, predicted, target, np.arange(50, dtype=np.float32) / 25,
            [("language", 4), ("sensor", 16)], RUNTIME, cfg,
        )
        with Image.open(result["image"]) as image:
            assert image.size == tuple(cfg.model_vis.canvas_size)
        with np.load(result["arrays"]) as arrays:
            assert arrays["backbone_features"].shape == (20, 8)
            assert arrays["predicted_actions"].shape == (50, 9)
            assert arrays["pca_scores"].shape == (20, 3)
            assert arrays["pca_rgb"].shape == (20, 3)
            assert arrays["pca_rgb"].dtype == np.uint8
            assert arrays["pca_group_names"].tolist() == ["all"]
            assert arrays["pca_means"].shape == (1, 8)
            assert arrays["pca_components"].shape == (1, 3, 8)
    finally:
        if RUNTIME.exists():
            shutil.rmtree(RUNTIME)


def test_model_visualization_restores_spatial_patch_grids() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    try:
        cfg = load_config()
        cameras = {camera.name: camera for camera in cfg.data_collector.render.cameras}
        rgb_frames = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
        sensor_frames = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
        overview = rgb_frames * (cameras["overview"].height // cfg.model.image_patch) * (cameras["overview"].width // cfg.model.image_patch)
        wrist = rgb_frames * (cameras["wrist"].height // cfg.model.image_patch) * (cameras["wrist"].width // cfg.model.image_patch)
        tactile = sensor_frames * 2 * (32 // cfg.model.tactile_patch) ** 2
        groups = [
            ("language", cfg.model_data.language_length), ("CLS", 1),
            ("register", cfg.model.register_tokens), ("overview", overview),
            ("wrist", wrist), ("tactile", tactile), ("state", sensor_frames),
        ]
        token_count = sum(count for _, count in groups)
        features = np.arange(token_count * 8, dtype=np.float32).reshape(token_count, 8) % 97
        actions = np.zeros((sensor_frames, 9), dtype=np.float32)
        result = render_model_visualization(
            features, actions, actions, np.arange(sensor_frames, dtype=np.float32) / cfg.model_data.sensor_hz,
            groups, RUNTIME, cfg,
        )
        assert result["spatial_pca"] is not None
        with Image.open(result["spatial_pca"]) as image:
            assert image.size == (cfg.model_vis.canvas_size[0], 1000)
        with np.load(result["arrays"]) as arrays:
            assert arrays["pca_group_names"].tolist() == ["overview", "wrist", "tactile"]
            assert arrays["pca_components"].shape == (3, 3, features.shape[1])
            assert arrays["pca_means"].shape == (3, features.shape[1])
            assert arrays["pca_explained_variance_ratio"].shape == (3, 3)
            spatial_start = cfg.model_data.language_length + 1 + cfg.model.register_tokens
            expected_means = np.stack((
                features[spatial_start:spatial_start + overview].mean(0),
                features[spatial_start + overview:spatial_start + overview + wrist].mean(0),
                features[spatial_start + overview + wrist:spatial_start + overview + wrist + tactile].mean(0),
            ))
            np.testing.assert_allclose(arrays["pca_means"], expected_means)
        assert result["pca_fit_tokens"] == {
            "overview": overview, "wrist": wrist, "tactile": tactile,
        }
    finally:
        if RUNTIME.exists():
            shutil.rmtree(RUNTIME)
