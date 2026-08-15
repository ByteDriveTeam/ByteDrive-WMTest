from pathlib import Path

from config import PROJECT_ROOT


def check_dataset_path(path) -> None:
    # 校验对象: 所有数据集写入路径 —— 必须位于项目目录内且不能等于项目根。
    resolved = Path(path).resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"数据集路径必须位于项目目录内: {resolved}")


def check_scene_record(record) -> None:
    # 校验对象: DatasetStore.publish 的 SceneRecord —— 只允许持久化成功且非空的场景。
    if not record.success_evidence.get("success") or not record.frames:
        raise ValueError("禁止持久化失败或空场景")
    # 校验对象: DatasetStore.publish 的 FrameRecord 序号 —— 必须连续且从零开始。
    if [frame.frame_index for frame in record.frames] != list(range(len(record.frames))):
        raise ValueError("场景帧序号不连续")

