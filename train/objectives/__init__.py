"""重导出 ByteDrive 多目标损失与epoch调度。

模块: train/objectives/__init__.py
依赖: train.objectives.objectives
读取配置: 无
对外接口:
    - LossOutput
    - compute_policy_losses
    - endpoint_weight
    - teacher_force_probability
    - visible_reconstruction_weight
"""

from train.objectives.objectives import (
    LossOutput, compute_policy_losses, endpoint_weight, teacher_force_probability,
    visible_reconstruction_weight,
)

__all__ = [
    "LossOutput", "compute_policy_losses", "endpoint_weight", "teacher_force_probability",
    "visible_reconstruction_weight",
]
