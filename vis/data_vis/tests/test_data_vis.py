"""验证可视化优先读取图像与触觉，并在缺失或强制时完成物理重放。

模块: vis/data_vis/tests/test_data_vis.py
依赖: pytest, pillow, config, data.data_collector, vis.data_vis
读取配置: data_collector.*, data_vis.*
对外接口: 无
"""

from dataclasses import asdict, replace
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from config import load_config
from data.data_collector.records import SceneRecord
from data.data_collector.scene import asset_fingerprint, build_mjcf, generate_scene_spec, scene_identifier
from data.data_collector.simulation import EmbodiedSimulator
from data.data_collector.storage import DatasetStore, config_fingerprint
from vis.data_vis import visualize_scene
from vis.data_vis.data_vis import _force_heatmap


RUNTIME = Path(__file__).resolve().parent / "_runtime"


@pytest.fixture(autouse=True)
def clean_runtime():
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True)
    yield
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)


def _publish_single_frame(cfg, dataset: Path) -> None:
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    mjcf_xml = build_mjcf(spec, cfg)
    simulator = EmbodiedSimulator(spec, mjcf_xml, cfg)
    try:
        simulator.capture("VIS_TEST", {"skill": "OBSERVE"}, {"success": True})
        config_hash = config_fingerprint(cfg)
        record = SceneRecord(
            scene_id=scene_identifier(spec, config_hash),
            spec=spec,
            mjcf_xml=mjcf_xml,
            frames=simulator.frames,
            success_evidence={"success": True, "test": True},
            asset_hash=asset_fingerprint(),
            config_hash=config_hash,
            config_snapshot=asdict(cfg),
            versions={"test": "true"},
        )
        store = DatasetStore(dataset, cfg, asset_fingerprint())
        store.initialize()
        store.publish(record)
    finally:
        simulator.close()


def _assert_png(result: dict) -> None:
    frames = sorted(Path(result["frames_output"]).glob("*.png"))
    assert len(frames) == 1
    with Image.open(frames[0]) as image:
        assert image.width > 320
        assert image.height >= 240


def test_visualizer_reads_stored_image_before_replay():
    cfg = load_config()
    render = replace(cfg.data_collector.render, enabled=True)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, render=render))
    dataset = RUNTIME / "stored_dataset"
    _publish_single_frame(cfg, dataset)

    result = visualize_scene(
        dataset, 0, cfg, output=RUNTIME / "stored_output", stride=1, max_frames=1, force_replay=False,
    )

    assert result["source_counts"] == {"stored": 1, "replayed": 0}
    assert Path(result["gif"]).is_file()
    _assert_png(result)


def test_visualizer_replays_when_image_is_missing():
    cfg = load_config()
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, render=replace(cfg.data_collector.render, enabled=False)))
    dataset = RUNTIME / "replay_dataset"
    _publish_single_frame(cfg, dataset)

    result = visualize_scene(
        dataset,
        0,
        cfg,
        output=RUNTIME / "replay_output",
        stride=1,
        max_frames=1,
        gif_enabled=False,
    )

    assert result["source_counts"] == {"stored": 0, "replayed": 1}
    assert result["tactile_source_counts"] == {"stored": 0, "recomputed": 1}
    assert result["gif"] is None
    _assert_png(result)


def test_visualizer_can_force_recompute_stored_tactile():
    cfg = load_config()
    sensors = replace(cfg.data_collector.sensors, contact_enabled=True)
    render = replace(cfg.data_collector.render, enabled=True)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, sensors=sensors, render=render))
    dataset = RUNTIME / "tactile_dataset"
    _publish_single_frame(cfg, dataset)

    stored = visualize_scene(
        dataset, 0, cfg, output=RUNTIME / "tactile_stored", stride=1, max_frames=1,
        force_replay=False, force_tactile_replay=False, gif_enabled=False,
    )
    replayed = visualize_scene(
        dataset, 0, cfg, output=RUNTIME / "tactile_replayed", stride=1, max_frames=1,
        force_replay=False, force_tactile_replay=True, gif_enabled=False,
    )

    assert stored["tactile_source_counts"] == {"stored": 1, "recomputed": 0}
    assert replayed["tactile_source_counts"] == {"stored": 0, "recomputed": 1}


def test_visualization_settings_do_not_change_collection_fingerprint():
    cfg = load_config()
    changed = replace(
        cfg,
        data_vis=replace(
            cfg.data_vis,
            gif_fps=cfg.data_vis.gif_fps + 1,
            force_replay=True,
            force_tactile_replay=True,
        ),
    )
    assert config_fingerprint(changed) == config_fingerprint(cfg)


def test_tactile_gamma_makes_low_force_visible_without_lighting_zero_force():
    cfg = load_config()
    zero = np.asarray(_force_heatmap(np.zeros((32, 32), dtype=np.float32), cfg.data_vis.tactile_force_max, cfg.data_vis.tactile_force_gamma))
    low_force = np.asarray(_force_heatmap(np.full((32, 32), 2.0, dtype=np.float32), cfg.data_vis.tactile_force_max, cfg.data_vis.tactile_force_gamma))
    assert int(zero.max()) == 0
    assert int(low_force.max()) >= 100
