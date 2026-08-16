"""提供成功场景 LMDB 可视化命令行入口。

模块: vis/data_vis/run.py
依赖: argparse, json, config, vis.data_vis
读取配置: data_vis.*；CLI 参数仅在显式给出时覆盖配置
对外接口:
    - main(argv=None) -> int
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from config import load_config
from vis.data_vis import visualize_scene


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可视化单场景具身采集 LMDB")
    parser.add_argument("--config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--camera")
    parser.add_argument("--modality", choices=("rgb", "depth", "segmentation"))
    parser.add_argument("--output")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-replay", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-tactile-replay", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析 CLI 覆盖项并生成可视化帧、GIF 与来源汇总。"""
    args = _parser().parse_args(argv)
    result = visualize_scene(
        args.dataset,
        args.scene,
        load_config(args.config),
        camera=args.camera,
        modality=args.modality,
        output=args.output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=args.stride,
        max_frames=args.max_frames,
        gif_enabled=args.gif,
        force_replay=args.force_replay,
        force_tactile_replay=args.force_tactile_replay,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
