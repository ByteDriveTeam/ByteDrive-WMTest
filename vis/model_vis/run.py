"""提供骨干末端特征与动作预测可视化CLI。

模块: vis/model_vis/run.py
依赖: argparse, json, config, vis.model_vis
读取配置: model_vis.*；CLI参数仅在显式给出时覆盖配置
对外接口:
    - main(argv=None) -> int
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from config import configure_mujoco_rendering, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="可视化ByteDrive检查点特征与动作")
    parser.add_argument(
        "checkpoint", nargs="?",
        help="可选检查点；省略时使用随机初始化模型",
    )
    parser.add_argument("--config")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析检查点和可选覆盖项，生成项目内可视化产物。"""
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    configure_mujoco_rendering(cfg)
    from vis.model_vis import visualize_model_checkpoint

    result = visualize_model_checkpoint(
        args.checkpoint, cfg, split=args.split, sample_index=args.sample_index,
        output=args.output, device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
