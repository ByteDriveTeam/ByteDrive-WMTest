from __future__ import annotations

from pathlib import Path

import torch

from config import PROJECT_ROOT
from config.schema import AppConfig


def check_training_environment(cfg: AppConfig) -> torch.device:
    # 校验对象: training.device——CUDA正式训练必须真实支持BF16。
    device = torch.device(cfg.training.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("正式训练需要支持 BF16 的 CUDA GPU")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("training.device 只支持 cuda 或测试用 cpu")
    return device


def check_project_output(path: Path) -> None:
    # 校验对象: 训练检查点和指标输出——不得写出项目边界。
    resolved = path.resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"训练输出必须位于项目内: {resolved}")
