from dataclasses import fields
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT
from config.schema import SensorSettings


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


def check_tactile_snapshot(snapshot: Any) -> None:
    # 校验对象: meta/scene.config_snapshot.data_collector.sensors —— 触觉重算必须使用完整且兼容的场景快照。
    expected = {field.name for field in fields(SensorSettings)}
    if not isinstance(snapshot, dict) or set(snapshot) != expected:
        raise ValueError("场景元数据缺少完整且兼容的 data_collector.sensors 配置快照，无法重算触觉")
