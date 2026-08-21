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


RUNTIME = Path(__file__).resolve().parent / "_runtime"


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
    finally:
        if RUNTIME.exists():
            shutil.rmtree(RUNTIME)
