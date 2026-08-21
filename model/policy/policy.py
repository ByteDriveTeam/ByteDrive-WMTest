"""组装多模态骨干、掩码 Predictor 与23维结构化流匹配策略。

模块: model/policy/policy.py
依赖: torch, config, model.position, model.transformer, model.policy.checks
读取配置: model.*, model_data.*, data_collector.render.cameras,
    data_collector.sensors.tactile_resolution
对外接口:
    - PolicyBatch
    - PolicyOutput
    - ByteDrivePolicy
    - sensor_token_counts(cfg) -> tuple[int, int, int, int]
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from config.schema import AppConfig
from model.policy.checks import (
    check_flow_statistics, check_policy_batch, check_predict_statistics, check_teacher_force,
)
from model.position import (
    MODALITY_CLS, MODALITY_LANGUAGE, MODALITY_OVERVIEW, MODALITY_PREDICT,
    MODALITY_REGISTER, MODALITY_STATE, MODALITY_TACTILE, MODALITY_WRIST,
    PositionInputs, SharedPositionEncoder, build_petr_geometry, far_dense_depths,
    logarithmic_depths, patch_centers,
)
from model.transformer import DenseResidualMixer, RMSNorm, TransformerBlock


@dataclass
class PolicyBatch:
    """保存模型前向需要的固定序列与训练目标。"""

    overview_rgb: torch.Tensor
    wrist_rgb: torch.Tensor
    tactile: torch.Tensor
    state: torch.Tensor
    language_ids: torch.Tensor
    language_valid: torch.Tensor
    overview_intrinsics: torch.Tensor
    overview_transform: torch.Tensor
    wrist_intrinsics: torch.Tensor
    wrist_transform: torch.Tensor
    tactile_geometry: torch.Tensor
    state_geometry: torch.Tensor
    coordinate_bounds: torch.Tensor
    rgb_time: torch.Tensor
    sensor_time: torch.Tensor
    future_time: torch.Tensor
    sensor_mask: torch.Tensor
    task_patch_mask: torch.Tensor
    behavior_valid: torch.Tensor
    phase_target: torch.Tensor
    cache_hits: torch.Tensor
    cache_misses: torch.Tensor
    flow_target: torch.Tensor | None = None

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "PolicyBatch":
        """将批次张量移动到指定设备，并允许Pinned Memory异步传输。"""
        return PolicyBatch(**{
            field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
            if isinstance(getattr(self, field.name), torch.Tensor)
            and field.name not in {"cache_hits", "cache_misses"} else getattr(self, field.name)
            for field in fields(self)
        })


@dataclass
class PolicyOutput:
    """保存逐层速度、最终积分轨迹和感知重建结果。"""

    velocities: torch.Tensor
    final_flow: torch.Tensor
    phase_logits: torch.Tensor
    predictor_features: torch.Tensor
    observation_features: torch.Tensor
    flow_noise: torch.Tensor


class LayerConditionedFlowDecoder(nn.Module):
    """使用层身份AdaLN共享12层速度场参数。"""

    def __init__(self, cfg: AppConfig):
        super().__init__()
        model = cfg.model
        self.norm = nn.LayerNorm(model.width, eps=model.norm_epsilon, elementwise_affine=False)
        self.modulation = nn.Linear(model.backbone_layers, 2 * model.width)
        self.first = nn.Linear(model.width, model.flow_hidden)
        self.second = nn.Linear(model.flow_hidden, 23)
        self.layers = model.backbone_layers
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, features: torch.Tensor, layer_index: int) -> torch.Tensor:
        one_hot = F.one_hot(torch.tensor(layer_index, device=features.device), self.layers).float()
        scale, shift = self.modulation(one_hot).chunk(2)
        normalized = self.norm(features.float()) * (1.0 + scale) + shift
        return self.second(F.silu(self.first(normalized)))


def sensor_token_counts(cfg: AppConfig) -> tuple[int, int, int, int]:
    """按集中配置推导Overview、Wrist、触觉和状态Token数。"""
    cameras = {camera.name: camera for camera in cfg.data_collector.render.cameras}
    rgb_frames = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
    sensor_frames = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
    overview = rgb_frames * (cameras["overview"].height // cfg.model.image_patch) * (cameras["overview"].width // cfg.model.image_patch)
    wrist = rgb_frames * (cameras["wrist"].height // cfg.model.image_patch) * (cameras["wrist"].width // cfg.model.image_patch)
    tactile_height, tactile_width = cfg.data_collector.sensors.tactile_resolution
    tactile = sensor_frames * 2 * (tactile_height // cfg.model.tactile_patch) * (tactile_width // cfg.model.tactile_patch)
    return overview, wrist, tactile, sensor_frames


class ByteDrivePolicy(nn.Module):
    """统一编码异构观测并生成未来动作与触觉摘要。"""

    SENSOR_MODES = (MODALITY_OVERVIEW, MODALITY_WRIST, MODALITY_TACTILE, MODALITY_STATE)

    def __init__(self, cfg: AppConfig, flow_statistics: tuple[list[float], list[float]] | None = None):
        super().__init__()
        check_flow_statistics(flow_statistics)
        self.cfg = cfg
        self.sensor_counts = sensor_token_counts(cfg)
        model = cfg.model
        self.overview_embed = nn.Conv2d(3, model.width, model.image_patch, model.image_patch)
        self.wrist_embed = nn.Conv2d(3, model.width, model.image_patch, model.image_patch)
        self.tactile_embed = nn.Conv2d(3, model.width, model.tactile_patch, model.tactile_patch, bias=True)
        self.state_embed = nn.Linear(37, model.width)
        self.language_embed = nn.Embedding(len(cfg.model_data.language_vocabulary) + 4, model.width, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model.width))
        self.register_token = nn.Parameter(torch.zeros(1, model.register_tokens, model.width))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.width))
        self.flow_embed = nn.Linear(23, model.width)
        self.register_buffer(
            "image_mean", torch.tensor(cfg.model_data.image_mean, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor(cfg.model_data.image_std, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.position = SharedPositionEncoder(cfg)
        self.backbone = nn.ModuleList([TransformerBlock(cfg, True) for _ in range(model.backbone_layers)])
        self.backbone_mixer = DenseResidualMixer(model.backbone_layers)
        self.backbone_output_norm = RMSNorm(model.width, model.norm_epsilon)
        self.predictor = nn.ModuleList([TransformerBlock(cfg, False) for _ in range(model.predictor_layers)])
        self.predictor_mixer = DenseResidualMixer(model.predictor_layers)
        self.predictor_output_norm = RMSNorm(model.width, model.norm_epsilon)
        self.velocity_decoder = LayerConditionedFlowDecoder(cfg)
        self.phase_head = nn.Linear(model.width, len(model.phase_names))
        flow_mean, flow_std = flow_statistics or ([0.0] * 23, [1.0] * 23)
        self.register_buffer("flow_mean", torch.tensor(flow_mean, dtype=torch.float32))
        self.register_buffer("flow_std", torch.tensor(flow_std, dtype=torch.float32))
        self.register_buffer("flow_statistics_ready", torch.tensor(flow_statistics is not None, dtype=torch.bool))
        for token in (self.cls_token, self.register_token, self.mask_token):
            nn.init.normal_(token, std=model.token_init_std)

    @property
    def sensor_tokens(self) -> int:
        return sum(self.sensor_counts)

    def _embed_images(self, images: torch.Tensor, embedding: nn.Conv2d) -> torch.Tensor:
        batch, frames, channels, height, width = images.shape
        normalized = (images.float() / 255.0 - self.image_mean) / self.image_std
        patches = embedding(normalized.reshape(batch * frames, channels, height, width))
        return patches.flatten(2).transpose(1, 2).reshape(batch, frames, -1, patches.shape[1])

    def _embed_sensors(self, batch: PolicyBatch) -> tuple[torch.Tensor, ...]:
        with torch.autocast(device_type=batch.state.device.type, enabled=False):
            overview = self._embed_images(batch.overview_rgb, self.overview_embed).flatten(1, 2)
            wrist = self._embed_images(batch.wrist_rgb, self.wrist_embed).flatten(1, 2)
            b, time, sides, channels, height, width = batch.tactile.shape
            tactile = self.tactile_embed(batch.tactile.float().reshape(b * time * sides, channels, height, width))
            tactile = tactile.flatten(2).transpose(1, 2).reshape(b, time * sides, -1, tactile.shape[1]).flatten(1, 2)
            return overview, wrist, tactile, self.state_embed(batch.state.float())

    @staticmethod
    def _normalize_points(points: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        low, high = bounds[:, 0], bounds[:, 1]
        return (2.0 * (points.float() - low) / (high - low) - 1.0).clamp(-1.0, 1.0)

    def _observation_conditions(self, batch: PolicyBatch) -> tuple[PositionInputs, torch.Tensor]:
        model, b = self.cfg.model, batch.state.shape[0]
        history_offset = self.cfg.model_data.history_seconds
        language, registers = self.cfg.model_data.language_length, model.register_tokens
        sensor_start, total = language + 1 + registers, language + 1 + registers + self.sensor_tokens
        device = batch.state.device
        modality = torch.empty((b, total), dtype=torch.long, device=device)
        modality[:, :language], modality[:, language] = MODALITY_LANGUAGE, MODALITY_CLS
        modality[:, language + 1:sensor_start] = MODALITY_REGISTER
        cursor = sensor_start
        for count, mode in zip(self.sensor_counts, self.SENSOR_MODES):
            modality[:, cursor:cursor + count], cursor = mode, cursor + count
        physical_time = torch.zeros((b, total), device=device)
        physical_valid = torch.zeros((b, total), dtype=torch.bool, device=device)
        language_index = torch.zeros((b, total), device=device)
        language_valid = torch.zeros((b, total), dtype=torch.bool, device=device)
        language_index[:, :language] = torch.arange(language, device=device)
        language_valid[:, :language] = batch.language_valid
        geometry = torch.zeros((b, total, model.petr_depth_samples, 3), device=device)
        geometry_valid = torch.zeros((b, total, model.petr_depth_samples), dtype=torch.bool, device=device)
        side = torch.full((b, total), -1, dtype=torch.long, device=device)
        position_enabled = torch.ones((b, total), dtype=torch.bool, device=device)
        position_enabled[:, language + 1:sensor_start] = False
        bounds = batch.coordinate_bounds[0] if batch.coordinate_bounds.ndim == 3 else batch.coordinate_bounds
        overview_height, overview_width = batch.overview_rgb.shape[-2:]
        wrist_height, wrist_width = batch.wrist_rgb.shape[-2:]
        overview_geometry = build_petr_geometry(
            patch_centers(overview_height, overview_width, model.image_patch, device), batch.overview_intrinsics,
            batch.overview_transform, far_dense_depths(
                *model.overview_depth_range, model.petr_depth_samples, device,
            ), bounds,
        ).flatten(1, 2)
        wrist_geometry = build_petr_geometry(
            patch_centers(wrist_height, wrist_width, model.image_patch, device), batch.wrist_intrinsics,
            batch.wrist_transform, logarithmic_depths(*model.wrist_depth_range, model.petr_depth_samples, device), bounds,
        ).flatten(1, 2)
        overview_count, wrist_count = overview_geometry.shape[1], wrist_geometry.shape[1]
        overview_patches = overview_count // batch.rgb_time.shape[1]
        wrist_patches = wrist_count // batch.rgb_time.shape[1]
        cursor = sensor_start
        geometry[:, cursor:cursor + overview_count], geometry_valid[:, cursor:cursor + overview_count] = overview_geometry, True
        physical_time[:, cursor:cursor + overview_count], physical_valid[:, cursor:cursor + overview_count] = (
            batch.rgb_time + history_offset
        ).repeat_interleave(overview_patches, 1), True
        cursor += overview_count
        geometry[:, cursor:cursor + wrist_count], geometry_valid[:, cursor:cursor + wrist_count] = wrist_geometry, True
        physical_time[:, cursor:cursor + wrist_count], physical_valid[:, cursor:cursor + wrist_count] = (
            batch.rgb_time + history_offset
        ).repeat_interleave(wrist_patches, 1), True
        cursor += wrist_count
        tactile_time, tactile_sides, tactile_patches = batch.tactile_geometry.shape[1:4]
        tactile_count = tactile_time * tactile_sides * tactile_patches
        geometry[:, cursor:cursor + tactile_count, 0] = self._normalize_points(batch.tactile_geometry, bounds).reshape(b, tactile_count, 3)
        geometry_valid[:, cursor:cursor + tactile_count, 0] = True
        physical_time[:, cursor:cursor + tactile_count], physical_valid[:, cursor:cursor + tactile_count] = (
            batch.sensor_time + history_offset
        ).repeat_interleave(tactile_sides * tactile_patches, 1), True
        side[:, cursor:cursor + tactile_count] = torch.arange(tactile_sides, device=device).repeat_interleave(tactile_patches).repeat(tactile_time)
        cursor += tactile_count
        state_count = batch.state_geometry.shape[1]
        geometry[:, cursor:cursor + state_count, 0] = self._normalize_points(batch.state_geometry, bounds)
        geometry_valid[:, cursor:cursor + state_count, 0] = True
        physical_time[:, cursor:cursor + state_count], physical_valid[:, cursor:cursor + state_count] = (
            batch.sensor_time + history_offset
        ), True
        return PositionInputs(
            modality, physical_time, physical_valid, language_index, language_valid,
            geometry, geometry_valid, side, position_enabled,
        ), modality

    def _predict_conditions(self, batch: PolicyBatch) -> PositionInputs:
        b, steps, device = batch.state.shape[0], batch.future_time.shape[1], batch.state.device
        depth = self.cfg.model.petr_depth_samples
        future_window_time = batch.future_time + self.cfg.model_data.history_seconds
        return PositionInputs(
            torch.full((b, steps), MODALITY_PREDICT, dtype=torch.long, device=device), future_window_time,
            torch.ones((b, steps), dtype=torch.bool, device=device), torch.zeros((b, steps), device=device),
            torch.zeros((b, steps), dtype=torch.bool, device=device), torch.zeros((b, steps, depth, 3), device=device),
            torch.zeros((b, steps, depth), dtype=torch.bool, device=device),
            torch.full((b, steps), -1, dtype=torch.long, device=device),
            torch.ones((b, steps), dtype=torch.bool, device=device),
        )

    def _embed_observations(self, batch: PolicyBatch) -> tuple[torch.Tensor, PositionInputs, torch.Tensor]:
        overview, wrist, tactile, state = self._embed_sensors(batch)
        b = state.shape[0]
        tokens = torch.cat((
            self.language_embed(batch.language_ids), self.cls_token.expand(b, -1, -1),
            self.register_token.expand(b, -1, -1), overview, wrist, tactile, state,
        ), dim=1)
        positions, modality = self._observation_conditions(batch)
        return tokens, positions, modality

    def _observation_attention(self, batch: PolicyBatch, sensor_mask: torch.Tensor) -> torch.Tensor:
        b, language = batch.language_ids.shape
        sensor_start = language + 1 + self.cfg.model.register_tokens
        total = sensor_start + self.sensor_tokens
        allowed = torch.zeros((b, total, total), dtype=torch.bool, device=batch.state.device)
        causal = torch.tril(torch.ones((language, language), dtype=torch.bool, device=batch.state.device))
        allowed[:, :language, :language] = causal.unsqueeze(0) & batch.language_valid.unsqueeze(1)
        diagonal = torch.arange(language, device=batch.state.device)
        allowed[:, diagonal, diagonal] = True
        visible = torch.cat((batch.language_valid, torch.ones((b, 1 + self.cfg.model.register_tokens), dtype=torch.bool, device=batch.state.device), ~sensor_mask), 1)
        allowed[:, language:, :] = visible.unsqueeze(1)
        return allowed

    @staticmethod
    def _full_attention(observation_allowed: torch.Tensor, visible: torch.Tensor, predict_steps: int) -> torch.Tensor:
        b, observation_count = visible.shape
        allowed = torch.zeros((b, observation_count + predict_steps, observation_count + predict_steps), dtype=torch.bool, device=visible.device)
        allowed[:, :observation_count, :observation_count] = observation_allowed
        predict_keys = torch.cat((visible, torch.ones((b, predict_steps), dtype=torch.bool, device=visible.device)), 1)
        allowed[:, observation_count:, :] = predict_keys.unsqueeze(1)
        return allowed

    def _encode_observations(self, tokens: torch.Tensor, positions: PositionInputs, modality: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        states, last_input = [tokens.to(torch.bfloat16)], tokens.to(torch.bfloat16)
        for layer_index, block in enumerate(self.backbone):
            layer_input = self.backbone_mixer(states, layer_index)
            layer_position = self.position(positions, layer_index)
            with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16, enabled=True):
                layer_output = block(layer_input, layer_position, modality, allowed)
            states.append(layer_output)
            last_input = layer_input
        return self.backbone_output_norm(states[-1] + last_input)

    def encode_teacher(self, batch: PolicyBatch) -> torch.Tensor:
        """用完整无掩码观测生成 Predictor 的 EMA 目标特征。"""
        tokens, positions, modality = self._embed_observations(batch)
        allowed = self._observation_attention(batch, torch.zeros_like(batch.sensor_mask))
        encoded = self._encode_observations(tokens, positions, modality, allowed)
        sensor_start = batch.language_ids.shape[1] + 1 + self.cfg.model.register_tokens
        return encoded[:, sensor_start:].float()

    def _run_predictor(self, features: torch.Tensor, positions: PositionInputs, modality: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_token = self.mask_token.expand(features.shape[0], features.shape[1], -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_token.to(features.dtype), features)
        states, last_input = [tokens.to(torch.bfloat16)], tokens.to(torch.bfloat16)
        allowed = torch.ones((tokens.shape[0], tokens.shape[1], tokens.shape[1]), dtype=torch.bool, device=tokens.device)
        offset = self.cfg.model.backbone_layers
        for index, block in enumerate(self.predictor):
            layer_input = self.predictor_mixer(states, index)
            layer_position = self.position(positions, offset + index)
            with torch.autocast(device_type=tokens.device.type, dtype=torch.bfloat16, enabled=True):
                layer_output = block(layer_input, layer_position, modality, allowed)
            states.append(layer_output)
            last_input = layer_input
        return self.predictor_output_norm(states[-1] + last_input).float()

    def forward(self, batch: PolicyBatch, teacher_force_probability: float = 0.0, flow_noise: torch.Tensor | None = None) -> PolicyOutput:
        """执行感知编码、逐层流积分、阶段分类和掩码特征重建。"""
        check_policy_batch(batch, self.cfg)
        check_teacher_force(teacher_force_probability)
        observation_tokens, observation_positions, observation_modality = self._embed_observations(batch)
        b, observation_count = observation_tokens.shape[:2]
        steps = batch.future_time.shape[1]
        flow_noise = torch.randn((b, steps, 23), device=batch.state.device) if flow_noise is None else flow_noise.float()
        flow_state = flow_noise
        predict_positions = self._predict_conditions(batch)
        positions = PositionInputs(**{
            name: torch.cat((getattr(observation_positions, name), getattr(predict_positions, name)), 1)
            for name in observation_positions.__dict__
        })
        modality = torch.cat((observation_modality, predict_positions.modality), 1)
        observation_allowed = self._observation_attention(batch, batch.sensor_mask)
        visible = torch.cat((batch.language_valid, torch.ones((b, 1 + self.cfg.model.register_tokens), dtype=torch.bool, device=batch.state.device), ~batch.sensor_mask), 1)
        allowed = self._full_attention(observation_allowed, visible, steps)
        observation_states = [observation_tokens.to(torch.bfloat16)]
        velocities, last_observation_input = [], observation_states[0]
        for layer_index, block in enumerate(self.backbone):
            observation_input = self.backbone_mixer(observation_states, layer_index)
            with torch.autocast(device_type=batch.state.device.type, enabled=False):
                predict_input = self.flow_embed(flow_state.float())
                layer_tokens = torch.cat((observation_input, predict_input.to(torch.bfloat16)), 1)
                layer_position = self.position(positions, layer_index)
            with torch.autocast(device_type=batch.state.device.type, dtype=torch.bfloat16, enabled=True):
                layer_output = block(layer_tokens, layer_position, modality, allowed)
            observation_output, predict_output = layer_output[:, :observation_count], layer_output[:, observation_count:]
            observation_states.append(observation_output)
            last_observation_input = observation_input
            with torch.autocast(device_type=batch.state.device.type, enabled=False):
                velocity = self.velocity_decoder(predict_output.float(), layer_index)
                model_next = flow_state.float() + velocity / len(self.backbone)
                if batch.flow_target is not None and teacher_force_probability > 0 and layer_index + 1 < len(self.backbone):
                    next_t = (layer_index + 1) / len(self.backbone)
                    teacher_next = flow_noise + next_t * (batch.flow_target.float() - flow_noise)
                    choose = torch.rand((b, 1, 1), device=flow_state.device) < teacher_force_probability
                    flow_state = torch.where(choose, teacher_next, model_next)
                else:
                    flow_state = model_next
            velocities.append(velocity)
        final_observation = self.backbone_output_norm(observation_states[-1] + last_observation_input)
        language = batch.language_ids.shape[1]
        phase_logits = self.phase_head(final_observation[:, language].float())
        sensor_start = language + 1 + self.cfg.model.register_tokens
        sensor_features = final_observation[:, sensor_start:].float()
        predictor_features = self._run_predictor(
            sensor_features, observation_positions.index(slice(sensor_start, None)),
            observation_modality[:, sensor_start:], batch.sensor_mask,
        )
        return PolicyOutput(torch.stack(velocities, 1), flow_state.float(), phase_logits.float(), predictor_features, sensor_features, flow_noise.float())

    @torch.no_grad()
    def predict(self, batch: PolicyBatch, flow_noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """返回物理量纲动作、触觉摘要和阶段分类结果。"""
        check_predict_statistics(self.flow_statistics_ready)
        output = self.forward(batch, 0.0, flow_noise)
        physical = output.final_flow * self.flow_std + self.flow_mean
        actions = physical[..., :9].clone()
        actions[..., 8] = torch.where(actions[..., 8] >= 0, 1.0, -1.0)
        return {"actions": actions, "tactile_summary": physical[..., 9:].reshape(*physical.shape[:2], 2, 7), "phase": output.phase_logits}


__all__ = ["ByteDrivePolicy", "PolicyBatch", "PolicyOutput", "sensor_token_counts"]
