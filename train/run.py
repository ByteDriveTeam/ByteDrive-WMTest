"""提供归一化统计、训练与检查点评估 CLI。

模块: train/run.py
依赖: argparse, json, config, data.model_dataset, train.engine
读取配置: 由 --config 加载全部配置
对外接口:
    - main() -> None
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from config import configure_mujoco_rendering, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ByteDrive 结构化流匹配训练")
    parser.add_argument("--config", help="可选 YAML 覆盖")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stats", help="扫描训练集并生成归一化统计")
    train = commands.add_parser("train", help="训练模型")
    train.add_argument("--resume", help="继续训练的检查点")
    evaluate = commands.add_parser("evaluate", help="评估检查点")
    evaluate.add_argument("checkpoint")
    return parser


def main() -> None:
    """解析命令并执行对应训练工作流。"""
    args = _parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "stats":
        from data.model_dataset import fit_normalization_statistics

        result = asdict(fit_normalization_statistics(cfg))
    elif args.command == "train":
        configure_mujoco_rendering(cfg)
        from train.engine import train_model

        result = train_model(cfg, args.resume)
    else:
        configure_mujoco_rendering(cfg)
        from train.engine import evaluate_checkpoint

        result = evaluate_checkpoint(cfg, args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
