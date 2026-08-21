from __future__ import annotations

import torch


def check_loss_shapes(
    velocities: torch.Tensor,
    final_flow: torch.Tensor,
    target: torch.Tensor,
    backbone_features: torch.Tensor | None,
    cls_index: int,
) -> None:
    # 校验对象: compute_policy_losses 的流张量——逐层和积分目标必须保持23维。
    if velocities.ndim != 4 or velocities.shape[-1] != 23:
        raise ValueError("velocities 期望 (B,L,40,23)")
    if final_flow.shape != target.shape or target.shape[-1] != 23:
        raise ValueError("final_flow 与 target 必须同为 (B,40,23)")
    # 校验对象: VISReg的CLS来源——必须来自骨干最终完整观测序列，不能误用Predictor或Patch均值。
    if backbone_features is None or backbone_features.ndim != 3 or cls_index >= backbone_features.shape[1]:
        raise ValueError("VISReg需要包含CLS位置的骨干末端特征")
