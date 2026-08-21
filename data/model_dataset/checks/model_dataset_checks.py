from __future__ import annotations

from pathlib import Path

import numpy as np

from config import PROJECT_ROOT


def check_dataset_path(path: Path) -> None:
    # 校验对象: ByteDriveDataset 数据根目录——数据集是只读输入，可以位于项目外。
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"数据集必须是已存在目录: {resolved}")


def check_statistics_output(path: Path) -> None:
    # 校验对象: fit_normalization_statistics 输出——派生统计不得写出项目。
    resolved = path.resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents:
        raise ValueError(f"统计输出必须位于项目内: {resolved}")


def check_statistics_input(path: Path) -> None:
    # 校验对象: ByteDriveDataset 的归一化统计输入——禁止用静默单位统计替代缺失文件。
    resolved = path.resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents or not resolved.is_file():
        raise FileNotFoundError(f"归一化统计不存在或不在项目内，请先运行 stats: {resolved}")


def check_statistics_values(values: dict) -> None:
    # 校验对象: NormalizationStats.load 的 JSON——维度必须匹配37维状态、23维流和三维坐标。
    expected = {
        "state_mean": 37, "state_std": 37, "tactile_map_mean": 3, "tactile_map_std": 3,
        "flow_mean": 23, "flow_std": 23, "coordinate_bounds": 3,
    }
    if any(len(values.get(name, ())) != length for name, length in expected.items()):
        raise ValueError("归一化统计维度与模型契约不一致")
    if any(value <= 0 for value in values["tactile_map_std"]):
        raise ValueError("共享触觉图标准差必须为正")
    if any(len(axis) != 2 or axis[0] >= axis[1] for axis in values["coordinate_bounds"]):
        raise ValueError("coordinate_bounds 必须是三个递增轴区间")
    flow_mean, flow_std = np.asarray(values["flow_mean"]), np.asarray(values["flow_std"])
    if flow_mean[8] != 0 or flow_std[8] != 1:
        raise ValueError("二值夹爪状态不得做均值方差归一化")
    if not np.allclose(flow_mean[9:16], flow_mean[16:23]) or not np.allclose(flow_std[9:16], flow_std[16:23]):
        raise ValueError("左右指触觉摘要必须共享归一化参数")


def check_frame_times(frame_times: np.ndarray) -> None:
    # 校验对象: ByteDriveDataset._indices 的 simulation_time——真实时间采样要求严格递增。
    if frame_times.ndim != 1 or len(frame_times) < 2 or np.any(np.diff(frame_times) <= 0):
        raise ValueError("simulation_time 必须是一维严格递增序列")
