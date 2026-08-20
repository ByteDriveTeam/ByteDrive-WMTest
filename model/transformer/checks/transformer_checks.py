from __future__ import annotations

import torch


def check_transformer_inputs(
    tokens: torch.Tensor,
    position: torch.Tensor,
    modality: torch.Tensor,
    allowed: torch.Tensor,
) -> None:
    # 校验对象: TransformerBlock.forward 入参——批次与 Token 轴必须完全对齐。
    if tokens.ndim != 3 or position.shape != tokens.shape:
        raise ValueError("tokens 与 position 必须同为 (B,N,D)")
    if modality.shape != tokens.shape[:2]:
        raise ValueError("modality 必须为 (B,N)")
    if allowed.shape != (tokens.shape[0], tokens.shape[1], tokens.shape[1]):
        raise ValueError("allowed 必须为 (B,N,N)")
    if allowed.dtype != torch.bool:
        raise ValueError("allowed 必须是 bool 张量")


def check_dense_states(states: list[torch.Tensor], expected: int) -> None:
    # 校验对象: DenseResidualMixer 的历史状态——必须只包含观测残差流。
    if len(states) != expected or not states:
        raise ValueError(f"稠密残差期望 {expected} 个历史状态")
    if any(state.shape != states[0].shape for state in states):
        raise ValueError("稠密残差历史的形状必须一致")
