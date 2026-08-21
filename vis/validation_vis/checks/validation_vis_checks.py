from pathlib import Path

from config import PROJECT_ROOT


def check_validation_visualization_output(output: Path) -> None:
    # 校验对象: 验证数据与曲线输出——公开绘图接口也不得绕过项目边界。
    root = PROJECT_ROOT.resolve()
    resolved = output.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError("自动验证可视化输出必须位于项目目录内")


def check_validation_visualization_inputs(output: Path, sample_index: int, sample_count: int) -> None:
    # 校验对象: 自动验证可视化输出——每个epoch的派生产物只能写入项目。
    check_validation_visualization_output(output)
    # 校验对象: 固定验证样本——必须在验证滑窗范围内，才能跨epoch对比。
    if not 0 <= sample_index < sample_count:
        raise IndexError(f"验证可视化样本 {sample_index} 超出 [0,{sample_count})")
