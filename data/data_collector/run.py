"""提供采集、续采、检查、验证、compact 和二次渲染 CLI。

模块: data/data_collector/run.py
依赖: argparse, config, data.data_collector.collection, data.data_collector.replay,
    data.data_collector.storage
读取配置: CLI 只覆盖 data_collector.collector.*, data_collector.render.enabled,
    data_collector.render.viewer, data_collector.sensors.contact_enabled
对外接口:
    - main(argv=None) -> int
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

from config import PROJECT_ROOT, load_config
from data.data_collector.collection import collect_scenes
from data.data_collector.replay import rerender_scene
from data.data_collector.storage import compact_scene, validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo 单臂具身智能成功场景采集器")
    parser.add_argument("--config")
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="采集到指定成功场景总数")
    collect.add_argument("--scenes", type=int)
    collect.add_argument("--seed", type=int)
    collect.add_argument("--output")
    collect.add_argument("--task", action="append")
    collect.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--render", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=None)
    collect.add_argument("--contact-sensors", action=argparse.BooleanOptionalAction, default=None)
    for name in ("inspect", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True)
        if name == "validate":
            command.add_argument("--deep", action="store_true")
    compact = commands.add_parser("compact")
    compact.add_argument("--dataset", required=True)
    compact.add_argument("--scene")
    rerender = commands.add_parser("rerender")
    rerender.add_argument("--dataset", required=True)
    rerender.add_argument("--scene", required=True)
    rerender.add_argument("--frames", required=True)
    rerender.add_argument("--camera", required=True)
    rerender.add_argument("--output", required=True)
    return parser


def _override(cfg, args):
    dc = cfg.data_collector
    collector = replace(
        dc.collector,
        scene_count=args.scenes if args.scenes is not None else dc.collector.scene_count,
        master_seed=args.seed if args.seed is not None else dc.collector.master_seed,
        output=args.output if args.output is not None else dc.collector.output,
        resume=args.resume if args.resume is not None else dc.collector.resume,
    )
    render = replace(
        dc.render,
        enabled=args.render if args.render is not None else dc.render.enabled,
        viewer=args.viewer if args.viewer is not None else dc.render.viewer,
    )
    sensors = replace(dc.sensors, contact_enabled=args.contact_sensors if args.contact_sensors is not None else dc.sensors.contact_enabled)
    return replace(cfg, data_collector=replace(dc, collector=collector, render=render, sensors=sensors))


def _frames(text: str) -> list[int]:
    output: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = (int(value) for value in part.split("-", maxsplit=1))
            output.extend(range(start, end + 1))
        else:
            output.append(int(part))
    return sorted(set(output))


def _dataset_path(text: str) -> Path:
    path = Path(text)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令并执行对应的采集或维护操作。"""
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "collect":
        result = collect_scenes(_override(cfg, args), args.task)
    elif args.command in {"inspect", "validate"}:
        result = validate_dataset(_dataset_path(args.dataset), cfg, deep=getattr(args, "deep", False))
    elif args.command == "compact":
        dataset = _dataset_path(args.dataset)
        scenes = list(dataset.glob(f"scene_{int(args.scene):06d}_*.lmdb")) if args.scene is not None and args.scene.isdigit() else (
            list(dataset.glob(f"*{args.scene}*.lmdb")) if args.scene else sorted(dataset.glob("scene_*.lmdb"))
        )
        result = {"compacted": [compact_scene(scene, cfg) for scene in scenes]}
    else:
        result = rerender_scene(_dataset_path(args.dataset), args.scene, _frames(args.frames), args.camera, _dataset_path(args.output), cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

