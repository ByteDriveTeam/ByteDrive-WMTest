"""重导出完整观测行为后训练与测试接口。

模块: train/post_training/__init__.py
依赖: train.post_training.post_training
读取配置: post_training.*, training.*, loss.*, validation_vis.*
对外接口:
    - evaluate_post_training_checkpoint
    - post_train_model
"""

from train.post_training.post_training import evaluate_post_training_checkpoint, post_train_model

__all__ = ["evaluate_post_training_checkpoint", "post_train_model"]
