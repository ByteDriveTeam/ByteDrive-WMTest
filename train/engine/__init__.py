"""重导出单GPU训练、EMA、评估和检查点接口。

模块: train/engine/__init__.py
依赖: train.engine.engine
读取配置: 无
对外接口:
    - create_ema_teacher
    - evaluate_checkpoint
    - evaluate_model
    - load_checkpoint
    - save_checkpoint
    - train_model
    - update_ema
"""

from train.engine.engine import (
    create_ema_teacher, evaluate_checkpoint, evaluate_model, load_checkpoint,
    save_checkpoint, train_model, update_ema,
)

__all__ = [
    "create_ema_teacher", "evaluate_checkpoint", "evaluate_model", "load_checkpoint",
    "save_checkpoint", "train_model", "update_ema",
]
