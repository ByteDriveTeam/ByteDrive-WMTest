"""加载集中配置并返回不可变配置对象。

模块: config/__init__.py
依赖: config.schema, yaml
读取配置: 全部配置由调用方指定的 YAML 提供
对外接口:
    - AppConfig
    - ConfigError
    - load_config(path=None) -> AppConfig
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from config.schema import AppConfig, ConfigError, build_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载默认配置，并可用另一个 YAML 递归覆盖。"""
    default_path = PROJECT_ROOT / "config" / "default.yaml"
    base = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    if path is None or Path(path).resolve() == default_path.resolve():
        return build_config(base)
    override = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return build_config(_merge(base, override))


__all__ = ["AppConfig", "ConfigError", "PROJECT_ROOT", "load_config"]

