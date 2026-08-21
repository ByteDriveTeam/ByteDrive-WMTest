"""构造多模态时间、PETR 几何与逐层 Q/K 位置条件。

模块: model/position/position.py
依赖: torch, config, model.position.checks
读取配置: model.width, model.backbone_layers, model.predictor_layers,
    model.position_hidden, model.time_frequencies, model.petr_depth_samples,
    model.norm_epsilon, model_data.history_seconds, model_data.future_seconds,
    model_data.language_length
对外接口:
    - PositionInputs
    - SharedPositionEncoder
    - patch_centers(height, width, patch, device) -> Tensor
    - build_petr_points(patch_centers, intrinsics, transforms, depths) -> Tensor
    - build_petr_geometry(patch_centers, intrinsics, transforms, depths, bounds) -> Tensor
    - far_dense_depths(near, far, count, device) -> Tensor
    - logarithmic_depths(near, far, count, device) -> Tensor
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from config.schema import AppConfig
from model.position.checks import check_camera_geometry, check_position_inputs


MODALITY_OVERVIEW = 0
MODALITY_WRIST = 1
MODALITY_TACTILE = 2
MODALITY_STATE = 3
MODALITY_LANGUAGE = 4
MODALITY_CLS = 5
MODALITY_REGISTER = 6
MODALITY_PREDICT = 7


@dataclass
class PositionInputs:
    """保存每个 Token 进入共享位置网络的原始条件。"""

    modality: torch.Tensor
    physical_time: torch.Tensor
    physical_valid: torch.Tensor
    language_index: torch.Tensor
    language_valid: torch.Tensor
    geometry: torch.Tensor
    geometry_valid: torch.Tensor
    side: torch.Tensor
    position_enabled: torch.Tensor

    def index(self, indices: torch.Tensor | slice) -> "PositionInputs":
        """在 Token 轴上选取条件，保持所有字段同步。"""
        return PositionInputs(**{
            name: value[:, indices]
            for name, value in self.__dict__.items()
        })


class AdaLayerNorm(nn.Module):
    """使用显式层 one-hot 调制 LayerNorm。"""

    def __init__(self, width: int, layer_count: int, epsilon: float):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=epsilon, elementwise_affine=False)
        self.modulation = nn.Linear(layer_count, 2 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.layer_count = layer_count

    def forward(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        condition = F.one_hot(
            torch.tensor(layer_index, device=x.device), self.layer_count
        ).to(dtype=torch.float32)
        scale, shift = self.modulation(condition).chunk(2)
        return self.norm(x) * (1.0 + scale) + shift


class SharedPositionEncoder(nn.Module):
    """为骨干和 Predictor 的15层生成仅作用于 Q/K 的位置向量。"""

    def __init__(self, cfg: AppConfig):
        super().__init__()
        model = cfg.model
        self.depth_samples = model.petr_depth_samples
        self.frequencies = model.time_frequencies
        self.history_seconds = cfg.model_data.history_seconds
        self.future_seconds = cfg.model_data.future_seconds
        self.language_length = cfg.model_data.language_length
        condition_width = 8 + 2 * self.frequencies + 1 + 2 * self.frequencies + 1
        condition_width += 4 * self.depth_samples + 2
        layer_count = model.backbone_layers + model.predictor_layers
        self.input = nn.Linear(condition_width, model.position_hidden)
        self.adaln = AdaLayerNorm(model.position_hidden, layer_count, model.norm_epsilon)
        self.output = nn.Linear(model.position_hidden, model.width)

    def _sinusoid(self, theta: torch.Tensor) -> torch.Tensor:
        exponent = torch.arange(self.frequencies, device=theta.device, dtype=torch.float32)
        divisor = 10000.0 ** (exponent / self.frequencies)
        angles = theta.unsqueeze(-1) / divisor
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def _conditions(self, values: PositionInputs) -> torch.Tensor:
        check_position_inputs(values.modality, values.geometry, values.geometry_valid, self.depth_samples)
        # physical_time 使用完整窗口坐标：历史观测位于 [0, history]，
        # PredictToken 位于 (history, history + future]。流匹配噪声时间不进入这里。
        physical_unit = values.physical_time / (self.history_seconds + self.future_seconds)
        physical_theta = physical_unit.clamp(0.0, 1.0) * (math.pi / 2.0)
        language_unit = values.language_index / max(self.language_length - 1, 1)
        language_theta = language_unit.clamp(0.0, 1.0) * (math.pi / 2.0)
        physical = self._sinusoid(physical_theta) * values.physical_valid.unsqueeze(-1)
        language = self._sinusoid(language_theta) * values.language_valid.unsqueeze(-1)
        modality = F.one_hot(values.modality, 8).to(dtype=torch.float32)
        geometry = values.geometry.to(dtype=torch.float32).flatten(-2)
        geometry_valid = values.geometry_valid.to(dtype=torch.float32)
        side = F.one_hot(values.side.clamp_min(0), 2).to(dtype=torch.float32)
        side = side * (values.side >= 0).unsqueeze(-1)
        return torch.cat((
            modality,
            physical,
            values.physical_valid.unsqueeze(-1).to(dtype=torch.float32),
            language,
            values.language_valid.unsqueeze(-1).to(dtype=torch.float32),
            geometry,
            geometry_valid,
            side,
        ), dim=-1)

    def forward(self, values: PositionInputs, layer_index: int) -> torch.Tensor:
        """返回形状 (B,N,D) 的 FP32 位置向量。"""
        device_type = values.modality.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            hidden = self.input(self._conditions(values))
            hidden = F.silu(self.adaln(hidden, layer_index))
            position = self.output(hidden)
            return position * values.position_enabled.unsqueeze(-1)


def patch_centers(height: int, width: int, patch: int, device: torch.device | str) -> torch.Tensor:
    """返回按行优先排列的图像 Patch 中心像素坐标。"""
    y = torch.arange(patch / 2, height, patch, device=device, dtype=torch.float32)
    x = torch.arange(patch / 2, width, patch, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)


def logarithmic_depths(near: float, far: float, count: int, device: torch.device | str) -> torch.Tensor:
    """在明确工作区间内生成近处更密的深度采样。"""
    return torch.logspace(math.log10(near), math.log10(far), count, device=device)


def far_dense_depths(near: float, far: float, count: int, device: torch.device | str) -> torch.Tensor:
    """生成近处稀疏、远处密集的镜像对数深度采样。"""
    near_dense = logarithmic_depths(near, far, count, device)
    return near + far - near_dense.flip(0)


def build_petr_points(
    centers: torch.Tensor,
    intrinsics: torch.Tensor,
    transforms: torch.Tensor,
    depths: torch.Tensor,
) -> torch.Tensor:
    """将每帧Patch射线上的深度点反投影到基座坐标。"""
    check_camera_geometry(centers, intrinsics, transforms)
    batch, frames = intrinsics.shape[:2]
    pixels = torch.cat((centers, torch.ones_like(centers[:, :1])), dim=-1)
    rays = torch.einsum("btij,pj->btpi", torch.linalg.inv(intrinsics.float()), pixels.float())
    camera_points = rays.unsqueeze(-2) * depths.float().view(1, 1, 1, -1, 1)
    rotation = transforms[..., :3, :3].float()
    translation = transforms[..., :3, 3].float()
    base_points = torch.einsum("btij,btpdj->btpdi", rotation, camera_points)
    return base_points + translation.view(batch, frames, 1, 1, 3)


def build_petr_geometry(
    centers: torch.Tensor,
    intrinsics: torch.Tensor,
    transforms: torch.Tensor,
    depths: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    """将每帧Patch射线上的基座坐标按训练集范围归一化。"""
    check_camera_geometry(centers, intrinsics, transforms, bounds)
    base_points = build_petr_points(centers, intrinsics, transforms, depths)
    low, high = bounds[:, 0], bounds[:, 1]
    return (2.0 * (base_points - low) / (high - low) - 1.0).clamp(-1.0, 1.0)


__all__ = [
    "MODALITY_CLS", "MODALITY_LANGUAGE", "MODALITY_OVERVIEW", "MODALITY_PREDICT",
    "MODALITY_REGISTER", "MODALITY_STATE", "MODALITY_TACTILE", "MODALITY_WRIST",
    "PositionInputs", "SharedPositionEncoder", "build_petr_geometry", "build_petr_points",
    "far_dense_depths", "logarithmic_depths", "patch_centers",
]
