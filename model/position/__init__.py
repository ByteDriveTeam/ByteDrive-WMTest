"""重导出共享 Q/K 位置编码公开接口。

模块: model/position/__init__.py
依赖: model.position.position
读取配置: 无
对外接口:
    - PositionInputs
    - SharedPositionEncoder
    - build_petr_geometry
    - build_petr_points
    - far_dense_depths
    - logarithmic_depths
    - patch_centers
"""

from model.position.position import (
    MODALITY_CLS,
    MODALITY_LANGUAGE,
    MODALITY_OVERVIEW,
    MODALITY_PREDICT,
    MODALITY_REGISTER,
    MODALITY_STATE,
    MODALITY_TACTILE,
    MODALITY_WRIST,
    PositionInputs,
    SharedPositionEncoder,
    build_petr_geometry,
    build_petr_points,
    far_dense_depths,
    logarithmic_depths,
    patch_centers,
)

__all__ = [
    "MODALITY_CLS", "MODALITY_LANGUAGE", "MODALITY_OVERVIEW", "MODALITY_PREDICT",
    "MODALITY_REGISTER", "MODALITY_STATE", "MODALITY_TACTILE", "MODALITY_WRIST",
    "PositionInputs", "SharedPositionEncoder", "build_petr_geometry", "build_petr_points",
    "far_dense_depths", "logarithmic_depths", "patch_centers",
]
