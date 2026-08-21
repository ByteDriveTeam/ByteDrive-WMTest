"""提供 ByteDrive 训练、评估与损失公开接口。

模块: train/__init__.py
依赖: importlib；按需加载 train.engine 与 train.objectives
读取配置: 无
对外接口:
    - train_model
    - evaluate_checkpoint
    - compute_policy_losses
"""

from importlib import import_module


def __getattr__(name: str):
    """延迟导入训练模块，使CLI能在MuJoCo导入前选择Linux渲染后端。"""
    modules = {
        "compute_policy_losses": "train.objectives",
        "evaluate_checkpoint": "train.engine",
        "train_model": "train.engine",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name]), name)

__all__ = ["compute_policy_losses", "evaluate_checkpoint", "train_model"]
