from __future__ import annotations

from pathlib import Path


def check_post_training_checkpoint(path: Path, checkpoint: dict, resume: bool) -> None:
    # 校验对象: 后训练源检查点 —— 必须含Student；恢复后训练时还必须含阶段与优化状态。
    if not path.is_file():
        raise FileNotFoundError(path)
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise ValueError("后训练源检查点缺少Student model状态")
    if resume:
        required = {"stage", "optimizer", "scheduler", "epoch"}
        if checkpoint.get("stage") != "post_training" or not required.issubset(checkpoint):
            raise ValueError("--resume只能恢复完整的后训练检查点")


def check_post_training_load(missing: list[str], unexpected: list[str], excluded: set[str]) -> None:
    # 校验对象: 后训练Student加载结果 —— 只允许Predictor专属参数缺失。
    if unexpected or set(missing) != excluded:
        raise RuntimeError(f"后训练权重不兼容: missing={missing}, unexpected={unexpected}")


__all__ = ["check_post_training_checkpoint", "check_post_training_load"]
