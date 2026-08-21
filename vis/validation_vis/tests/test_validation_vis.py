"""验证固定验证数据、模型推理和loss历史图自动输出。

模块: vis/validation_vis/tests/test_validation_vis.py
依赖: pathlib, shutil, pillow, config, data.model_dataset, model.policy, vis.validation_vis
读取配置: validation_vis.*, model_vis.*, model_data.*, model.*
对外接口: 无
"""

from pathlib import Path
import shutil
from dataclasses import replace

from PIL import Image

from config import load_config
from data.model_dataset import NormalizationStats
from model.policy import ByteDrivePolicy, PolicyBatch
from model.policy.tests.test_policy import _batch, _tiny_config
from vis.validation_vis import generate_validation_visualizations, render_training_history, render_validation_data


RUNTIME = Path(__file__).resolve().parent / "_runtime"


def _sample(cfg=None) -> PolicyBatch:
    batch = _batch(cfg)
    return PolicyBatch(**{
        name: value[0] if hasattr(value, "shape") and value.shape[0] == 1 else value
        for name, value in batch.__dict__.items()
    })


def test_validation_data_and_history_images() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    try:
        cfg = load_config()
        data = render_validation_data(_sample(), NormalizationStats.identity(cfg), RUNTIME / "data", cfg)
        history = [
            {
                "epoch": epoch,
                **{f"train_{name}": float(3 - epoch + index / 10) for index, name in enumerate(("total", "velocity", "endpoint", "reconstruction", "phase", "visreg"))},
                "validation": {name: float(3.2 - epoch + index / 10) for index, name in enumerate(("total", "velocity", "endpoint", "reconstruction", "phase", "visreg"))},
            }
            for epoch in range(2)
        ]
        curves = render_training_history(history, RUNTIME / "history", cfg)
        for result, expected in ((data, cfg.validation_vis.data_canvas_size), (curves, cfg.validation_vis.history_canvas_size)):
            with Image.open(result["image"]) as image:
                assert image.size == tuple(expected)
    finally:
        if RUNTIME.exists():
            shutil.rmtree(RUNTIME)


def test_validation_step_generates_all_three_visualizations() -> None:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    try:
        cfg = _tiny_config()
        cfg = replace(cfg, validation_vis=replace(cfg.validation_vis, output=str(RUNTIME)))
        stats = NormalizationStats.identity(cfg)
        sample = _sample(cfg)
        model = ByteDrivePolicy(cfg, (stats.flow_mean, stats.flow_std)).train()
        history = [{
            "epoch": 0,
            **{f"train_{name}": 1.0 for name in ("total", "velocity", "endpoint", "reconstruction", "phase", "visreg")},
            "validation": {name: 1.1 for name in ("total", "velocity", "endpoint", "reconstruction", "phase", "visreg")},
        }]
        result = generate_validation_visualizations(model, [sample], stats, history, 1, cfg)
        assert model.training
        assert Path(result["model"]["image"]).is_file()
        assert Path(result["data"]["image"]).is_file()
        assert Path(result["history"]["image"]).is_file()
        assert (RUNTIME / "latest.json").is_file()
    finally:
        if RUNTIME.exists():
            shutil.rmtree(RUNTIME)
