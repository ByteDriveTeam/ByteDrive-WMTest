"""提供 ByteDrive 训练、评估与损失公开接口。

模块: train/__init__.py
依赖: train.engine, train.objectives
读取配置: 无
对外接口:
    - train_model
    - evaluate_checkpoint
    - compute_policy_losses
"""

from train.engine import evaluate_checkpoint, train_model
from train.objectives import compute_policy_losses

__all__ = ["compute_policy_losses", "evaluate_checkpoint", "train_model"]
