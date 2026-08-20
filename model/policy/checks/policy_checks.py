from __future__ import annotations

import torch

from config.schema import AppConfig


def check_policy_batch(batch: object, cfg: AppConfig) -> None:
    # 校验对象: ByteDrivePolicy.forward 的 PolicyBatch——固定采样率必须形成规划的 Token 数。
    rgb_frames = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
    sensor_frames = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
    future_frames = round(cfg.model_data.future_seconds * cfg.model_data.sensor_hz)
    cameras = {camera.name: camera for camera in cfg.data_collector.render.cameras}
    tactile_height, tactile_width = cfg.data_collector.sensors.tactile_resolution
    overview_tokens = rgb_frames * (cameras["overview"].height // cfg.model.image_patch) * (cameras["overview"].width // cfg.model.image_patch)
    wrist_tokens = rgb_frames * (cameras["wrist"].height // cfg.model.image_patch) * (cameras["wrist"].width // cfg.model.image_patch)
    tactile_tokens = sensor_frames * 2 * (tactile_height // cfg.model.tactile_patch) * (tactile_width // cfg.model.tactile_patch)
    sensor_tokens = overview_tokens + wrist_tokens + tactile_tokens + sensor_frames
    expected = {
        "overview_rgb": (rgb_frames, 3, cameras["overview"].height, cameras["overview"].width),
        "wrist_rgb": (rgb_frames, 3, cameras["wrist"].height, cameras["wrist"].width),
        "tactile": (sensor_frames, 2, 3, tactile_height, tactile_width), "state": (sensor_frames, 37),
        "language_ids": (cfg.model_data.language_length,), "flow_target": (future_frames, 23),
        "sensor_mask": (sensor_tokens,),
    }
    for name, trailing in expected.items():
        value = getattr(batch, name)
        if value is not None and tuple(value.shape[1:]) != trailing:
            raise ValueError(f"{name} 期望 (B,{','.join(map(str, trailing))})，实际 {tuple(value.shape)}")
    if batch.language_valid.dtype != torch.bool or batch.sensor_mask.dtype != torch.bool:
        raise ValueError("language_valid 与 sensor_mask 必须是 bool")


def check_teacher_force(probability: float) -> None:
    # 校验对象: teacher_force_probability——调度采样概率必须可解释。
    if not 0.0 <= probability <= 1.0:
        raise ValueError("teacher_force_probability 必须位于 [0,1]")


def check_flow_statistics(flow_statistics: tuple[list[float], list[float]] | None) -> None:
    # 校验对象: ByteDrivePolicy 构造参数 flow_statistics——显式统计必须符合23维流契约。
    if flow_statistics is None:
        return
    mean, std = map(lambda values: torch.as_tensor(values, dtype=torch.float32), flow_statistics)
    if mean.shape != (23,) or std.shape != (23,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("flow_statistics 必须是有限的23维均值和标准差")
    if torch.any(std <= 0) or mean[8] != 0 or std[8] != 1:
        raise ValueError("流标准差必须为正，且二值夹爪统计必须保持0/1")
    if not torch.allclose(mean[9:16], mean[16:23]) or not torch.allclose(std[9:16], std[16:23]):
        raise ValueError("左右指触觉摘要必须共享流归一化参数")


def check_predict_statistics(ready: torch.Tensor) -> None:
    # 校验对象: ByteDrivePolicy.predict 的流统计状态——物理量纲输出禁止使用静默单位统计。
    if not bool(ready.item()):
        raise RuntimeError("predict 需要真实流归一化统计；请从 stats 文件构造模型")
