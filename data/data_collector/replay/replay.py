"""从单场景 LMDB 恢复完整物理状态并进行二次渲染。

模块: data/data_collector/replay/replay.py
依赖: lmdb, mujoco, numpy, pillow, config, data.data_collector.scene, data.data_collector.storage
读取配置: data_collector.render.cameras, data_collector.storage.max_dbs
对外接口:
    - rerender_scene(dataset, scene, frames, camera, output, cfg) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import lmdb
import mujoco
import numpy as np
from PIL import Image

from config.schema import AppConfig
from data.data_collector.replay.checks.replay_checks import check_replay_inputs
from data.data_collector.scene import materialize_mjcf
from data.data_collector.storage import decode_value, validate_scene


def _select_scene(dataset: Path, scene: str | int) -> Path:
    if isinstance(scene, int) or str(scene).isdigit():
        matches = list(dataset.glob(f"scene_{int(scene):06d}_*.lmdb"))
    else:
        matches = list(dataset.glob(f"*{scene}*.lmdb"))
    if len(matches) != 1:
        raise ValueError(f"场景选择必须唯一，当前匹配 {len(matches)} 个")
    return matches[0]


def rerender_scene(
    dataset: str | Path,
    scene: str | int,
    frames: Iterable[int],
    camera: str,
    output: str | Path,
    cfg: AppConfig,
) -> dict[str, object]:
    """恢复指定帧并输出 RGB、深度和实例分割。"""
    dataset_path = Path(dataset).resolve()
    output_path = Path(output).resolve()
    check_replay_inputs(dataset_path, output_path, camera, cfg)
    scene_path = _select_scene(dataset_path, scene)
    report = validate_scene(scene_path, cfg, deep=False)
    env = lmdb.open(str(scene_path), readonly=True, lock=False, max_dbs=cfg.data_collector.storage.max_dbs)
    rendered = 0
    try:
        meta_db = env.open_db(b"meta", create=False)
        frames_db = env.open_db(b"frames", create=False)
        with env.begin() as transaction:
            metadata = decode_value(transaction.get(b"scene", db=meta_db))
            model = mujoco.MjModel.from_xml_string(materialize_mjcf(metadata["mjcf_xml"]))
            data = mujoco.MjData(model)
            camera_cfg = next(item for item in cfg.data_collector.render.cameras if item.name == camera)
            renderer = mujoco.Renderer(model, camera_cfg.height, camera_cfg.width)
            output_path.mkdir(parents=True, exist_ok=True)
            try:
                for frame_index in frames:
                    key = str(frame_index).zfill(cfg.data_collector.storage.frame_key_width).encode("ascii")
                    encoded = transaction.get(key, db=frames_db)
                    if encoded is None:
                        raise IndexError(f"场景不存在帧 {frame_index}")
                    frame = decode_value(encoded)
                    mujoco.mj_setState(model, data, frame["physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    mujoco.mj_forward(model, data)
                    stem = f"{scene_path.stem}_frame_{frame_index:08d}_{camera}"
                    if "rgb" in camera_cfg.modalities:
                        renderer.update_scene(data, camera=camera)
                        Image.fromarray(renderer.render()).save(output_path / f"{stem}_rgb.png")
                    if "depth" in camera_cfg.modalities:
                        renderer.enable_depth_rendering()
                        renderer.update_scene(data, camera=camera)
                        np.save(output_path / f"{stem}_depth.npy", renderer.render().astype(np.float32), allow_pickle=False)
                        renderer.disable_depth_rendering()
                    if "segmentation" in camera_cfg.modalities:
                        renderer.enable_segmentation_rendering()
                        renderer.update_scene(data, camera=camera)
                        np.save(output_path / f"{stem}_segmentation.npy", renderer.render().astype(np.int32), allow_pickle=False)
                        renderer.disable_segmentation_rendering()
                    rendered += 1
            finally:
                renderer.close()
    finally:
        env.close()
    return {"scene": report["scene_id"], "rendered_frames": rendered, "output": str(output_path)}


__all__ = ["rerender_scene"]

