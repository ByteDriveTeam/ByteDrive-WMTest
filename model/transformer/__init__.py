"""重导出 Transformer 内核与稠密残差公开接口。

模块: model/transformer/__init__.py
依赖: model.transformer.transformer
读取配置: 无
对外接口:
    - RMSNorm
    - TransformerBlock
    - DenseResidualMixer
"""

from model.transformer.transformer import DenseResidualMixer, RMSNorm, TransformerBlock

__all__ = ["DenseResidualMixer", "RMSNorm", "TransformerBlock"]
