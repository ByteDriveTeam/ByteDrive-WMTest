from __future__ import annotations

from pathlib import Path

from config import PROJECT_ROOT


def check_cache_directory(path: Path) -> None:
    # 校验对象: ReplayDiskCache 输出目录——派生缓存不得写出项目边界。
    resolved = path.resolve()
    project = PROJECT_ROOT.resolve()
    if resolved == project or project not in resolved.parents:
        raise ValueError(f"重放缓存必须位于项目子目录: {resolved}")

