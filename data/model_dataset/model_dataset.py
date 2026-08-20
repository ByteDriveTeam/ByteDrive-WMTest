"""从单场景 LMDB 采样固定多频率窗口并在线重放视觉与触觉。

模块: data/model_dataset/model_dataset.py
依赖: json, lmdb, mujoco, numpy, torch, config, data.data_collector, model.policy
读取配置: model_data.*, model.*, training.seed, data_collector.render.cameras,
    data_collector.sensors.*, data_collector.storage.frame_key_width,
    data_collector.storage.max_dbs, data_collector.controller.gripper_open,
    data_collector.controller.gripper_closed
对外接口:
    - NormalizationStats
    - ClosedLanguageTokenizer
    - ByteDriveDataset
    - tactile_summary(force_maps) -> Tensor
    - build_sensor_mask(task_related, temporal_gradient, counts, cfg, seed) -> Tensor
    - canonical_phase(phase, phase_names) -> str
    - behavior_validity(phases) -> ndarray
    - collate_policy_batches(samples) -> PolicyBatch
    - fit_normalization_statistics(cfg) -> NormalizationStats
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re
from typing import Any, Iterable

import lmdb
import mujoco
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.data_collector.scene import add_virtual_tactile_sites, materialize_mjcf
from data.data_collector.simulation import compute_tactile_state
from data.data_collector.storage import decode_value
from data.model_dataset.checks import (
    check_dataset_path, check_frame_times, check_statistics_input, check_statistics_output,
    check_statistics_values,
)
from model.policy import PolicyBatch, sensor_token_counts
from model.position import build_petr_points, logarithmic_depths, patch_centers


@dataclass(frozen=True)
class NormalizationStats:
    """保存训练集派生的逐通道归一化统计。"""

    state_mean: list[float]
    state_std: list[float]
    tactile_map_mean: list[float]
    tactile_map_std: list[float]
    flow_mean: list[float]
    flow_std: list[float]
    coordinate_bounds: list[list[float]]
    dataset_schema: str

    @classmethod
    def identity(cls, cfg: AppConfig) -> "NormalizationStats":
        """返回仅用于统计扫描和测试的单位归一化。"""
        return cls(
            [0.0] * 37, [1.0] * 37, [0.0] * 3, [1.0] * 3,
            [0.0] * 23, [1.0] * 23, cfg.model_data.coordinate_fallback_bounds, "1.2.0",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        """从项目内JSON读取归一化统计。"""
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        check_statistics_values(values)
        return cls(**values)

    def save(self, path: str | Path) -> None:
        """原子写入项目内归一化统计。"""
        output = Path(path)
        check_statistics_output(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)


class ClosedLanguageTokenizer:
    """将受控指令映射到固定40位闭集词表。"""

    def __init__(self, length: int, vocabulary: list[str]):
        self.length = length
        self.token_to_id = {token: index + 4 for index, token in enumerate(vocabulary)}

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """返回含BOS/EOS的ID和有效位置掩码。"""
        lexical = re.findall(r"[A-Za-z0-9_]+|[;.]", text)
        ids = [1, *(self.token_to_id.get(token, 3) for token in lexical), 2]
        if len(ids) > self.length:
            raise ValueError(f"指令 Token 数 {len(ids)} 超过上限 {self.length}")
        valid = torch.zeros(self.length, dtype=torch.bool)
        valid[:len(ids)] = True
        return torch.tensor(ids + [0] * (self.length - len(ids)), dtype=torch.long), valid


def _resolve_normalization_stats(
    cfg: AppConfig,
    stats: NormalizationStats | None,
    normalize: bool,
) -> NormalizationStats:
    if stats is not None:
        return stats
    if not normalize:
        return NormalizationStats.identity(cfg)
    statistics_path = Path(cfg.model_data.statistics)
    statistics_path = statistics_path if statistics_path.is_absolute() else PROJECT_ROOT / statistics_path
    check_statistics_input(statistics_path)
    return NormalizationStats.load(statistics_path)


def tactile_summary(force_maps: torch.Tensor) -> torch.Tensor:
    """将形状 (...,2,3,H,W) 触觉图压缩为每指7维统计。"""
    if force_maps.ndim < 4 or force_maps.shape[-4:-2] != (2, 3):
        raise ValueError("force_maps 期望 (...,2,3,H,W)")
    values = force_maps.float()
    means = values.mean(dim=(-1, -2))
    normal = values[..., 0, :, :].clamp_min(0.0)
    total = normal.sum(dim=(-1, -2), keepdim=True)
    y_coordinates = torch.linspace(-1.0, 1.0, values.shape[-2], device=values.device)
    x_coordinates = torch.linspace(-1.0, 1.0, values.shape[-1], device=values.device)
    yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    safe_total = total.clamp_min(torch.finfo(values.dtype).tiny)
    center_x = (normal * xx).sum(dim=(-1, -2), keepdim=True) / safe_total
    center_y = (normal * yy).sum(dim=(-1, -2), keepdim=True) / safe_total
    std_x = torch.sqrt((normal * (xx - center_x) ** 2).sum(dim=(-1, -2), keepdim=True) / safe_total)
    std_y = torch.sqrt((normal * (yy - center_y) ** 2).sum(dim=(-1, -2), keepdim=True) / safe_total)
    contact = total > 0
    geometry = torch.cat((center_x, center_y, std_x, std_y), dim=-1).squeeze(-2)
    geometry = torch.where(contact.squeeze(-1), geometry, torch.zeros_like(geometry))
    return torch.cat((means, geometry), dim=-1)


def canonical_phase(phase: str, phase_names: list[str]) -> str:
    """将对象名与重试后缀去除，且不产生RETRY类别。"""
    if phase.startswith("REOPEN_RETRY"):
        return "OPEN"
    cleaned = re.sub(r"_RETRY_\d+.*$", "", phase)
    return next((prefix for prefix in phase_names if cleaned.startswith(prefix)), phase_names[-1])


def behavior_validity(phases: Iterable[str]) -> np.ndarray:
    """将成功轨迹中被后续REOPEN证明失败的抓取尝试标为无行为监督。"""
    phase_list = list(phases)
    valid = np.ones(len(phase_list), dtype=bool)
    attempt_start: int | None = None
    for index, phase in enumerate(phase_list):
        if phase.startswith("HOME_BEFORE"):
            attempt_start = index
        elif phase.startswith("REOPEN_RETRY") and attempt_start is not None:
            valid[attempt_start:index] = False
            attempt_start = None
    return valid


def _unit_gradient(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    span = values.max() - values.min()
    return torch.zeros_like(values) if span <= torch.finfo(values.dtype).eps else (values - values.min()) / span


def _temporal_patch_gradient(values: torch.Tensor, patch: int) -> torch.Tensor:
    """将相邻帧变化量连续映射到每个时刻的Patch。"""
    if values.ndim == 4:
        values = values.unsqueeze(1)
    time, streams, channels, height, width = values.shape
    difference = (values[1:].float() - values[:-1].float()).abs().mean(2)
    pooled = F.avg_pool2d(difference.reshape(-1, 1, height, width), patch, patch)
    pooled = pooled.flatten(1).reshape(time - 1, streams, -1)
    gradient = torch.zeros((time, streams, pooled.shape[-1]), dtype=torch.float32)
    gradient[:-1] = torch.maximum(gradient[:-1], pooled)
    gradient[1:] = torch.maximum(gradient[1:], pooled)
    return _unit_gradient(gradient).flatten()


def _temporal_state_gradient(values: torch.Tensor) -> torch.Tensor:
    difference = (values[1:].float() - values[:-1].float()).abs().mean(-1)
    gradient = torch.zeros(values.shape[0], dtype=torch.float32)
    gradient[:-1] = torch.maximum(gradient[:-1], difference)
    gradient[1:] = torch.maximum(gradient[1:], difference)
    return _unit_gradient(gradient)


def build_sensor_mask(
    task_related: torch.Tensor,
    temporal_gradient: torch.Tensor,
    counts: tuple[int, int, int, int],
    cfg: AppConfig,
    seed: int,
) -> torch.Tensor:
    """在全传感器序列上组合任务优先、时间梯度和跨模态补偿掩码。"""
    total = sum(counts)
    if task_related.shape != (total,) or temporal_gradient.shape != (total,) or task_related.dtype != torch.bool:
        raise ValueError("任务掩码、时间梯度与传感器Token数必须对齐")
    target = round(total * cfg.model.mask_ratio)
    generator = torch.Generator().manual_seed(seed)
    structured = torch.rand((), generator=generator) < cfg.model.task_priority_sample_probability
    if not structured:
        selected = torch.zeros(total, dtype=torch.bool)
        selected[torch.randperm(total, generator=generator)[:target]] = True
        return selected
    selected = task_related.clone()
    if int(selected.sum()) >= target:
        task_indices = torch.where(selected)[0]
        selected.zero_()
        selected[task_indices[torch.randperm(len(task_indices), generator=generator)[:target]]] = True
        return selected
    overview, wrist, tactile, state = counts
    tactile_start, state_start = overview + wrist, overview + wrist + tactile
    sensor_group_masked = False
    for group in (torch.arange(tactile_start, state_start), torch.arange(state_start, total)):
        if torch.rand((), generator=generator) < cfg.model.mask_group_probability:
            room = target - int(selected.sum())
            candidates = group[~selected[group]]
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:room]]
            selected[chosen] = True
            sensor_group_masked = sensor_group_masked or len(chosen) > 0
    room = target - int(selected.sum())
    if room == 0:
        return selected
    candidates = torch.where(~selected)[0]
    weights = 1.0 + cfg.model.temporal_gradient_weight * temporal_gradient[candidates].clamp(0.0, 1.0)
    if sensor_group_masked:
        weights = torch.where(
            candidates < tactile_start,
            weights * cfg.model.rgb_relief_when_sensor_masked,
            weights,
        )
    selected[candidates[torch.multinomial(weights, room, replacement=False, generator=generator)]] = True
    return selected


class _SceneReplay:
    def __init__(self, metadata: dict[str, Any], cfg: AppConfig, create_renderers: bool):
        self.cfg = cfg
        xml = metadata["mjcf_xml"]
        if "left_tactile_site" not in xml:
            xml = add_virtual_tactile_sites(xml)
        self.model = mujoco.MjModel.from_xml_string(materialize_mjcf(xml))
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("link0").id
        self.renderers = {
            camera.name: mujoco.Renderer(self.model, camera.height, camera.width)
            for camera in cfg.data_collector.render.cameras
        } if create_renderers else {}
        references = {
            value for step in metadata["spec"]["task"]["steps"]
            for key in ("object_ref", "target_ref") if (value := step.get(key))
        }
        self.relevant_geoms = {
            index for index in range(self.model.ngeom)
            if any(reference in (self.model.geom(index).name or "") for reference in references)
        }

    def restore(self, frame: dict[str, Any]) -> None:
        mujoco.mj_setState(self.model, self.data, frame["physics_state"], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, self.data)

    def camera(self, frame: dict[str, Any], name: str, render_rgb: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.restore(frame)
        camera_cfg = next(camera for camera in self.cfg.data_collector.render.cameras if camera.name == name)
        camera_id = self.model.camera(name).id
        rotation = self.data.cam_xmat[camera_id].reshape(3, 3).copy()
        position = self.data.cam_xpos[camera_id].copy()
        base_rotation = self.data.xmat[self.base].reshape(3, 3)
        base_position = self.data.xpos[self.base]
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3], transform[:3, 3] = base_rotation.T @ rotation, base_rotation.T @ (position - base_position)
        fx, fy, px, py = self.model.cam_intrinsic[camera_id]
        intrinsics = np.asarray([[fx, 0, camera_cfg.width / 2 + px], [0, fy, camera_cfg.height / 2 + py], [0, 0, 1]], dtype=np.float32)
        image = np.zeros((camera_cfg.height, camera_cfg.width, 3), dtype=np.uint8)
        segmentation = np.zeros((camera_cfg.height, camera_cfg.width), dtype=bool)
        if render_rgb:
            renderer = self.renderers[name]
            renderer.update_scene(self.data, camera=name)
            image = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            renderer.update_scene(self.data, camera=name)
            raw = renderer.render().copy()
            renderer.disable_segmentation_rendering()
            if raw.ndim == 3:
                segmentation = (raw[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(raw[..., 0], list(self.relevant_geoms))
            else:
                segmentation = np.isin(raw, list(self.relevant_geoms))
        return image, segmentation, intrinsics, transform

    def tactile(self, frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        self.restore(frame)
        _, maps = compute_tactile_state(self.model, self.data, self.cfg.data_collector.sensors)
        force = np.stack([maps[side].transpose(2, 0, 1) for side in ("left", "right")])
        base_rotation = self.data.xmat[self.base].reshape(3, 3)
        base_position = self.data.xpos[self.base]
        extent_x, extent_y = self.cfg.data_collector.sensors.tactile_extent
        resolution_y, resolution_x = self.cfg.data_collector.sensors.tactile_resolution
        patches_y = resolution_y // self.cfg.model.tactile_patch
        patches_x = resolution_x // self.cfg.model.tactile_patch
        x = np.linspace(-extent_x / 2, extent_x / 2, patches_x, dtype=np.float32)
        z = np.linspace(-extent_y / 2, extent_y / 2, patches_y, dtype=np.float32)
        zz, xx = np.meshgrid(z, x, indexing="ij")
        local = np.stack((xx.flatten(), np.zeros(patches_y * patches_x, dtype=np.float32), zz.flatten()), axis=-1)
        geometry = []
        for side in ("left", "right"):
            site = self.model.site(f"{side}_tactile_site").id
            world = local @ self.data.site_xmat[site].reshape(3, 3).T + self.data.site_xpos[site]
            geometry.append((world - base_position) @ base_rotation)
        return force.astype(np.float32), np.stack(geometry).astype(np.float32)

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()


def _state_vector(frame: dict[str, Any]) -> np.ndarray:
    ee = frame["robot"]["frames"]["ee_site"]
    return np.concatenate((
        ee["position_base"], ee["quaternion_base_wxyz"], ee["linear_velocity_base"],
        ee["angular_velocity_base"], ee["linear_acceleration_base"], ee["angular_acceleration_base"],
        frame["robot"]["gripper_width"], frame["robot"]["actuator_control"], frame["robot"]["actuator_force"],
    )).astype(np.float32)


def _sampling_times(cfg: AppConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_count = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
    sensor_count = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
    future_count = round(cfg.model_data.future_seconds * cfg.model_data.sensor_hz)
    rgb = (np.arange(rgb_count, dtype=np.float32) - (rgb_count - 1)) / cfg.model_data.rgb_hz
    sensor = (np.arange(sensor_count, dtype=np.float32) - (sensor_count - 1)) / cfg.model_data.sensor_hz
    future = np.arange(1, future_count + 1, dtype=np.float32) / cfg.model_data.sensor_hz
    return rgb, sensor, future


def _nearest_time_indices(frame_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(frame_times, target_times, side="left").clip(0, len(frame_times) - 1)
    left = (right - 1).clip(0, len(frame_times) - 1)
    choose_left = np.abs(frame_times[left] - target_times) <= np.abs(frame_times[right] - target_times)
    return np.where(choose_left, left, right)


class ByteDriveDataset(Dataset[PolicyBatch]):
    """以场景为epoch采样单位的在线重放数据集。"""

    def __init__(self, cfg: AppConfig, split: str, stats: NormalizationStats | None = None, normalize: bool = True, render_rgb: bool = True):
        root = Path(cfg.model_data.dataset)
        self.root = root.resolve() if root.is_absolute() else (PROJECT_ROOT / root).resolve()
        check_dataset_path(self.root)
        all_scenes = sorted(self.root.glob("scene_*.lmdb"))
        train_end = round(len(all_scenes) * cfg.model_data.train_fraction)
        validation_end = train_end + round(len(all_scenes) * cfg.model_data.validation_fraction)
        partitions = {"train": all_scenes[:train_end], "validation": all_scenes[train_end:validation_end], "test": all_scenes[validation_end:]}
        if split not in partitions:
            raise ValueError("split 必须是 train、validation 或 test")
        self.scenes, self.cfg, self.split = partitions[split], cfg, split
        self.windows = cfg.model_data.windows_per_scene if split == "train" else cfg.model_data.validation_windows_per_scene
        self.stats = _resolve_normalization_stats(cfg, stats, normalize)
        self.normalize, self.render_rgb = normalize, render_rgb
        self.tokenizer = ClosedLanguageTokenizer(cfg.model_data.language_length, cfg.model_data.language_vocabulary)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """更换训练窗口的确定性随机流。"""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.scenes) * self.windows

    def _indices(self, frame_times: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        check_frame_times(frame_times)
        rgb_time, sensor_time, future_time = _sampling_times(self.cfg)
        history_start = min(float(rgb_time[0]), float(sensor_time[0]))
        future_end = float(future_time[-1])
        low = int(np.searchsorted(frame_times, frame_times[0] - history_start, side="left"))
        high = int(np.searchsorted(frame_times, frame_times[-1] - future_end, side="right")) - 1
        if low > high:
            raise ValueError("场景长度不足以形成4秒窗口")
        seed = self.cfg.training.seed + self.epoch * max(len(self), 1) + index
        anchor = int(np.random.default_rng(seed).integers(low, high + 1)) if self.split == "train" else low + (high - low) * (index % self.windows + 1) // (self.windows + 1)
        anchor_time = frame_times[anchor]
        rgb = _nearest_time_indices(frame_times, anchor_time + rgb_time)
        sensor = _nearest_time_indices(frame_times, anchor_time + sensor_time)
        future = _nearest_time_indices(frame_times, anchor_time + future_time)
        return rgb, sensor, future

    @staticmethod
    def _read_scene(path: Path, cfg: AppConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_dbs=cfg.data_collector.storage.max_dbs)
        try:
            meta_db, frame_db, index_db = env.open_db(b"meta"), env.open_db(b"frames"), env.open_db(b"index")
            with env.begin() as transaction:
                metadata = decode_value(transaction.get(b"scene", db=meta_db))
                count = decode_value(transaction.get(b"summary", db=index_db))["frame_count"]
                frames_data = [decode_value(transaction.get(str(i).zfill(cfg.data_collector.storage.frame_key_width).encode(), db=frame_db)) for i in range(count)]
            return metadata, frames_data
        finally:
            env.close()

    @staticmethod
    def _patch_mask(segmentation: np.ndarray, patch: int) -> np.ndarray:
        height, width = segmentation.shape
        return segmentation.reshape(height // patch, patch, width // patch, patch).any(axis=(1, 3)).flatten()

    def __getitem__(self, index: int) -> PolicyBatch:
        scene = self.scenes[index // self.windows]
        metadata, frames_data = self._read_scene(scene, self.cfg)
        frame_times = np.asarray([frame["simulation_time"] for frame in frames_data], dtype=np.float64)
        rgb_indices, sensor_indices, future_indices = self._indices(frame_times, index)
        replay = _SceneReplay(metadata, self.cfg, self.render_rgb)
        try:
            camera_values = {
                name: [replay.camera(frames_data[i], name, self.render_rgb) for i in rgb_indices]
                for name in ("overview", "wrist")
            }
            tactile_values = [replay.tactile(frames_data[i]) for i in np.concatenate((sensor_indices, future_indices))]
        finally:
            replay.close()
        overview, wrist = camera_values["overview"], camera_values["wrist"]
        image_mean = np.asarray(self.cfg.model_data.image_mean, dtype=np.float32)[:, None, None]
        image_std = np.asarray(self.cfg.model_data.image_std, dtype=np.float32)[:, None, None]
        images = lambda values: torch.from_numpy(np.stack([(value[0].transpose(2, 0, 1) / 255.0 - image_mean) / image_std for value in values])).float()
        overview_rgb, wrist_rgb = images(overview), images(wrist)
        sensor_count = len(sensor_indices)
        tactile_history_raw = torch.from_numpy(np.stack([value[0] for value in tactile_values[:sensor_count]])).float()
        tactile_future = torch.from_numpy(np.stack([value[0] for value in tactile_values[sensor_count:]])).float()
        state_raw = np.stack([_state_vector(frames_data[i]) for i in sensor_indices])
        target_actions = np.stack([
            np.concatenate((frames_data[i]["action"]["joint_position_target"], [frames_data[i]["action"]["gripper_width_target"]]))
            for i in future_indices
        ]).astype(np.float32)
        midpoint = (self.cfg.data_collector.controller.gripper_open + self.cfg.data_collector.controller.gripper_closed) / 2
        binary = np.where(target_actions[:, -1:] >= midpoint, 1.0, -1.0).astype(np.float32)
        flow_raw = torch.cat((torch.from_numpy(np.concatenate((target_actions, binary), axis=-1)), tactile_summary(tactile_future).flatten(1)), dim=-1)
        state = torch.from_numpy(state_raw).float()
        if self.normalize:
            state = (state - torch.tensor(self.stats.state_mean)) / torch.tensor(self.stats.state_std)
            tactile_mean = torch.tensor(self.stats.tactile_map_mean).view(1, 1, 3, 1, 1)
            tactile_std = torch.tensor(self.stats.tactile_map_std).view(1, 1, 3, 1, 1)
            tactile_history = (tactile_history_raw - tactile_mean) / tactile_std
            flow_target = (flow_raw - torch.tensor(self.stats.flow_mean)) / torch.tensor(self.stats.flow_std)
            flow_target[:, 8] = flow_raw[:, 8]
        else:
            tactile_history = tactile_history_raw
            flow_target = flow_raw
        language_ids, language_valid = self.tokenizer.encode(metadata["spec"]["task"]["instruction"])
        counts = sensor_token_counts(self.cfg)
        required = torch.zeros(sum(counts), dtype=torch.bool)
        overview_required = torch.from_numpy(np.concatenate([
            self._patch_mask(value[1], self.cfg.model.image_patch) for value in overview
        ]))
        wrist_required = torch.from_numpy(np.concatenate([
            self._patch_mask(value[1], self.cfg.model.image_patch) for value in wrist
        ]))
        required[:len(overview_required)] = overview_required
        wrist_start = len(overview_required)
        required[wrist_start:wrist_start + len(wrist_required)] = wrist_required
        temporal_gradient = torch.cat((
            _temporal_patch_gradient(overview_rgb, self.cfg.model.image_patch),
            _temporal_patch_gradient(wrist_rgb, self.cfg.model.image_patch),
            _temporal_patch_gradient(tactile_history, self.cfg.model.tactile_patch),
            _temporal_state_gradient(state),
        ))
        sensor_mask = build_sensor_mask(
            required, temporal_gradient, counts, self.cfg,
            self.cfg.training.seed + self.epoch * max(len(self), 1) + index,
        )
        phases = [frame["phase"] for frame in frames_data]
        behavior = behavior_validity(phases)
        current_index = int(sensor_indices[-1])
        phase_name = canonical_phase(phases[current_index], self.cfg.model.phase_names)
        phase_target = self.cfg.model.phase_names.index(phase_name) if behavior[current_index] else -100
        return PolicyBatch(
            overview_rgb=overview_rgb, wrist_rgb=wrist_rgb, tactile=tactile_history, state=state,
            language_ids=language_ids, language_valid=language_valid,
            overview_intrinsics=torch.from_numpy(np.stack([value[2] for value in overview])),
            overview_transform=torch.from_numpy(np.stack([value[3] for value in overview])),
            wrist_intrinsics=torch.from_numpy(np.stack([value[2] for value in wrist])),
            wrist_transform=torch.from_numpy(np.stack([value[3] for value in wrist])),
            tactile_geometry=torch.from_numpy(np.stack([value[1] for value in tactile_values[:sensor_count]])),
            state_geometry=torch.from_numpy(np.stack([frames_data[i]["robot"]["frames"]["ee_site"]["position_base"] for i in sensor_indices])),
            coordinate_bounds=torch.tensor(self.stats.coordinate_bounds, dtype=torch.float32),
            rgb_time=torch.from_numpy(_sampling_times(self.cfg)[0]),
            sensor_time=torch.from_numpy(_sampling_times(self.cfg)[1]),
            future_time=torch.from_numpy(_sampling_times(self.cfg)[2]),
            sensor_mask=sensor_mask, task_patch_mask=required,
            behavior_valid=torch.from_numpy(behavior[future_indices]), phase_target=torch.tensor(phase_target), flow_target=flow_target.float(),
        )


def collate_policy_batches(samples: list[PolicyBatch]) -> PolicyBatch:
    """将固定序列样本堆叠为模型批次。"""
    return PolicyBatch(**{
        field.name: torch.stack([getattr(sample, field.name) for sample in samples])
        for field in fields(PolicyBatch)
    })


class _RunningMoments:
    def __init__(self, width: int):
        self.count, self.sum, self.square = 0, np.zeros(width, dtype=np.float64), np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        flattened = values.reshape(-1, values.shape[-1]).astype(np.float64)
        self.count += len(flattened)
        self.sum += flattened.sum(0)
        self.square += np.square(flattened).sum(0)

    def finish(self, epsilon: float) -> tuple[list[float], list[float]]:
        mean = self.sum / max(self.count, 1)
        variance = self.square / max(self.count, 1) - np.square(mean)
        return mean.tolist(), np.sqrt(np.maximum(variance, epsilon**2)).tolist()


def _shared_finger_statistics(moments: _RunningMoments, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    count = max(moments.count, 1)
    paired_sum = moments.sum[9:16] + moments.sum[16:23]
    paired_square = moments.square[9:16] + moments.square[16:23]
    mean = paired_sum / (2 * count)
    variance = paired_square / (2 * count) - np.square(mean)
    return mean, np.sqrt(np.maximum(variance, epsilon**2))


def fit_normalization_statistics(cfg: AppConfig) -> NormalizationStats:
    """遍历训练场景的单窗口并写入逐通道统计。"""
    dataset = ByteDriveDataset(cfg, "train", normalize=False, render_rgb=False)
    state_moments, tactile_moments, flow_moments = _RunningMoments(37), _RunningMoments(3), _RunningMoments(23)
    coordinate_low, coordinate_high = np.full(3, np.inf), np.full(3, -np.inf)
    for sample in dataset:
        state_moments.update(sample.state.numpy())
        tactile_channels_last = sample.tactile.permute(0, 1, 3, 4, 2).numpy()
        tactile_moments.update(tactile_channels_last)
        valid_flow = sample.flow_target[sample.behavior_valid]
        if len(valid_flow):
            flow_moments.update(valid_flow.numpy())
        visual_points = []
        for name, depth_range in (("overview", cfg.model.overview_depth_range), ("wrist", cfg.model.wrist_depth_range)):
            image = getattr(sample, f"{name}_rgb")
            centers = patch_centers(*image.shape[-2:], cfg.model.image_patch, image.device)
            depths = logarithmic_depths(*depth_range, cfg.model.petr_depth_samples, image.device)
            points = build_petr_points(
                centers, getattr(sample, f"{name}_intrinsics").unsqueeze(0),
                getattr(sample, f"{name}_transform").unsqueeze(0), depths,
            )
            visual_points.append(points.numpy().reshape(-1, 3))
        coordinates = np.concatenate((
            *visual_points, sample.tactile_geometry.numpy().reshape(-1, 3),
            sample.state_geometry.numpy().reshape(-1, 3),
        ))
        coordinate_low, coordinate_high = np.minimum(coordinate_low, coordinates.min(0)), np.maximum(coordinate_high, coordinates.max(0))
    state_mean, state_std = state_moments.finish(cfg.model_data.normalization_epsilon)
    tactile_mean, tactile_std = tactile_moments.finish(cfg.model_data.normalization_epsilon)
    flow_mean, flow_std = flow_moments.finish(cfg.model_data.normalization_epsilon)
    finger_mean, finger_std = _shared_finger_statistics(flow_moments, cfg.model_data.normalization_epsilon)
    flow_mean[9:16] = flow_mean[16:23] = finger_mean.tolist()
    flow_std[9:16] = flow_std[16:23] = finger_std.tolist()
    flow_mean[8], flow_std[8] = 0.0, 1.0
    margin = np.maximum((coordinate_high - coordinate_low) * 0.05, cfg.model_data.normalization_epsilon)
    stats = NormalizationStats(
        state_mean, state_std, tactile_mean, tactile_std, flow_mean, flow_std,
        np.stack((coordinate_low - margin, coordinate_high + margin), axis=-1).tolist(), "1.2.0",
    )
    output = Path(cfg.model_data.statistics)
    stats.save(output if output.is_absolute() else PROJECT_ROOT / output)
    return stats


__all__ = [
    "ByteDriveDataset", "ClosedLanguageTokenizer", "NormalizationStats", "behavior_validity",
    "build_sensor_mask", "canonical_phase", "collate_policy_batches", "fit_normalization_statistics",
    "tactile_summary",
]
