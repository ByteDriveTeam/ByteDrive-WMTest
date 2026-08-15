from pathlib import Path

from config import PROJECT_ROOT


def check_visualization_inputs(dataset: Path, output: Path, modality: str) -> None:
    # 校验对象: 数据集目录 —— 必须是项目内已存在的目录。
    root = PROJECT_ROOT.resolve()
    if not dataset.is_dir() or dataset == root or root not in dataset.parents:
        raise ValueError("可视化数据集必须是项目目录内已存在的目录")
    # 校验对象: 输出目录 —— 禁止向项目外或项目根目录写入。
    if output == root or root not in output.parents:
        raise ValueError("可视化输出必须位于项目目录内")
    # 校验对象: 视觉模态 —— 仅接受采集 schema 定义的三种模态。
    if modality not in {"rgb", "depth", "segmentation"}:
        raise ValueError(f"未知视觉模态: {modality}")

