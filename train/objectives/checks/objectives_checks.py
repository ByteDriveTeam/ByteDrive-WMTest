from __future__ import annotations

import torch


def check_loss_shapes(velocities: torch.Tensor, final_flow: torch.Tensor, target: torch.Tensor) -> None:
    # 校验对象: compute_policy_losses 的流张量——逐层和积分目标必须保持23维。
    if velocities.ndim != 4 or velocities.shape[-1] != 23:
        raise ValueError("velocities 期望 (B,L,40,23)")
    if final_flow.shape != target.shape or target.shape[-1] != 23:
        raise ValueError("final_flow 与 target 必须同为 (B,40,23)")
