"""定义 BF16 Pre-Norm Transformer、模态 LoRA 与独立稠密残差流。

模块: model/transformer/transformer.py
依赖: torch, config, model.transformer.checks
读取配置: model.width, model.heads, model.ffn_width, model.lora_rank, model.norm_epsilon
对外接口:
    - RMSNorm
    - TransformerBlock
    - DenseResidualMixer
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from config.schema import AppConfig
from model.transformer.checks import check_dense_states, check_transformer_inputs


class RMSNorm(nn.Module):
    """使用FP32累积保持BF16残差流稳定的RMSNorm。"""

    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + self.epsilon)
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class QKAttention(nn.Module):
    """仅将逐层位置向量加入 Q/K 的 SDPA。"""

    def __init__(self, width: int, heads: int, epsilon: float):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.q_norm = RMSNorm(self.head_width, epsilon)
        self.k_norm = RMSNorm(self.head_width, epsilon)

    def forward(self, x: torch.Tensor, position: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        positional = position.to(dtype=query.dtype)
        query, key = query + positional, key + positional
        reshape = lambda tensor: tensor.view(batch, length, self.heads, self.head_width).transpose(1, 2)
        query, key, value = reshape(query), reshape(key), reshape(value)
        query, key = self.q_norm(query), self.k_norm(key)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed.unsqueeze(1), dropout_p=0.0
        )
        merged = attended.transpose(1, 2).reshape(batch, length, width)
        return self.output(merged)


class ModalityLoRAFFN(nn.Module):
    """在 SwiGLU 第一层注入逐Token门控的模态独立LoRA。"""

    def __init__(self, width: int, hidden: int, rank: int, enabled: bool):
        super().__init__()
        self.first = nn.Linear(width, hidden)
        self.second = nn.Linear(hidden // 2, width)
        self.enabled = enabled
        if enabled:
            self.lora_a = nn.Parameter(torch.empty(8, rank, width))
            self.lora_b = nn.Parameter(torch.zeros(8, hidden, rank))
            self.gate = nn.Linear(width, 1)
            nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        projected = self.first(x)
        if self.enabled:
            selected_a = self.lora_a[modality]
            selected_b = self.lora_b[modality]
            low_rank = torch.einsum("bnw,bnrw->bnr", x, selected_a)
            delta = torch.einsum("bnr,bnhr->bnh", low_rank, selected_b)
            projected = projected + torch.sigmoid(self.gate(x)) * delta
        gate, value = projected.chunk(2, dim=-1)
        return self.second(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    """执行一层 Pre-Norm SDPA 和 SwiGLU FFN。"""

    def __init__(self, cfg: AppConfig, lora_enabled: bool):
        super().__init__()
        model = cfg.model
        self.attention_norm = RMSNorm(model.width, model.norm_epsilon)
        self.attention = QKAttention(model.width, model.heads, model.norm_epsilon)
        self.ffn_norm = RMSNorm(model.width, model.norm_epsilon)
        self.ffn = ModalityLoRAFFN(model.width, model.ffn_width, model.lora_rank, lora_enabled)

    def forward(
        self,
        tokens: torch.Tensor,
        position: torch.Tensor,
        modality: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        check_transformer_inputs(tokens, position, modality, allowed)
        tokens = tokens + self.attention(self.attention_norm(tokens), position, allowed)
        return tokens + self.ffn(self.ffn_norm(tokens), modality)


class DenseResidualMixer(nn.Module):
    """对初始状态和先前各层输出学习尺度不变的凸组合。"""

    def __init__(self, layers: int):
        super().__init__()
        self.logits = nn.ParameterList([
            nn.Parameter(torch.zeros(index + 1)) for index in range(layers)
        ])

    def forward(self, states: list[torch.Tensor], layer_index: int) -> torch.Tensor:
        check_dense_states(states, layer_index + 1)
        weights = torch.softmax(self.logits[layer_index].float(), dim=0)
        stacked = torch.stack(states, dim=0)
        mixed = torch.einsum("l,lbnw->bnw", weights, stacked.float())
        return mixed.to(dtype=states[0].dtype)


__all__ = ["DenseResidualMixer", "RMSNorm", "TransformerBlock"]
