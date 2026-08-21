from pathlib import Path

import torch

from config import PROJECT_ROOT


def check_model_visualization_arrays(
    features,
    predicted_actions,
    target_actions,
    future_time,
    token_groups: list[tuple[str, int]],
    output: Path,
) -> None:
    # 校验对象: 骨干特征和动作绘图数组——Token、特征维和50步动作必须一一对应。
    if features.ndim != 2 or sum(count for _, count in token_groups) != features.shape[0]:
        raise ValueError("骨干特征必须是与Token分组一致的二维数组")
    if predicted_actions.shape != target_actions.shape or predicted_actions.ndim != 2 or predicted_actions.shape[1] != 9:
        raise ValueError("预测和真值动作必须是相同的[T,9]数组")
    if future_time.ndim != 1 or len(future_time) != len(predicted_actions):
        raise ValueError("未来时间轴必须与动作步数一致")
    # 校验对象: 模型可视化输出——任何图像和原始数组都不得写出项目。
    root = PROJECT_ROOT.resolve()
    resolved_output = output.resolve()
    if resolved_output == root or root not in resolved_output.parents:
        raise ValueError("模型可视化输出必须位于项目目录内")


def check_model_visualization_inputs(
    checkpoint: Path,
    statistics: Path,
    output: Path,
    sample_index: int,
    sample_count: int,
    device: torch.device,
) -> None:
    # 校验对象: 检查点与归一化统计——模型推理必须同时拥有权重和物理量缩放参数。
    if not checkpoint.is_file():
        raise FileNotFoundError(f"检查点不存在: {checkpoint}")
    if not statistics.is_file():
        raise FileNotFoundError(f"归一化统计不存在: {statistics}")
    # 校验对象: 模型可视化根输出——运行推理前先拦截项目外路径。
    root = PROJECT_ROOT.resolve()
    if output.resolve() == root or root not in output.resolve().parents:
        raise ValueError("模型可视化输出必须位于项目目录内")
    # 校验对象: 可视化样本索引——必须指向所选划分中的真实滑窗。
    if not 0 <= sample_index < sample_count:
        raise IndexError(f"样本索引 {sample_index} 超出 [0,{sample_count})")
    # 校验对象: CUDA BF16 可视化推理——与训练保持相同精度边界。
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("模型可视化的CUDA设备必须支持BF16")
