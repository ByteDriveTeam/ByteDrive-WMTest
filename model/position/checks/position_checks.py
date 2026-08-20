from __future__ import annotations

import torch


def check_position_inputs(
    modality: torch.Tensor,
    geometry: torch.Tensor,
    geometry_valid: torch.Tensor,
    depth_samples: int,
) -> None:
    # 校验对象: PositionInputs 的模态与几何张量——Token 轴必须对齐。
    if modality.ndim != 2 or geometry.shape[:2] != modality.shape:
        raise ValueError("位置条件期望 modality=(B,N) 且 geometry=(B,N,D,3)")
    if geometry.shape[-2:] != (depth_samples, 3):
        raise ValueError(f"geometry 期望末两维 ({depth_samples},3)")
    if geometry_valid.shape != geometry.shape[:-1]:
        raise ValueError("geometry_valid 必须与 geometry 的前三维一致")
    if torch.any((modality < 0) | (modality >= 8)):
        raise ValueError("modality 必须位于 [0,7]")


def check_camera_geometry(
    patch_centers: torch.Tensor,
    intrinsics: torch.Tensor,
    transforms: torch.Tensor,
    bounds: torch.Tensor | None = None,
) -> None:
    # 校验对象: build_petr_geometry 入参——相机几何必须可批量反投影。
    if patch_centers.ndim != 2 or patch_centers.shape[-1] != 2:
        raise ValueError("patch_centers 期望 (P,2)")
    if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics 期望 (B,T,3,3)")
    if transforms.shape != (*intrinsics.shape[:2], 4, 4):
        raise ValueError("transforms 期望 (B,T,4,4)")
    if bounds is not None and (bounds.shape != (3, 2) or torch.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("bounds 期望每轴递增的 (3,2)")
