"""重导出每次验证后的自动可视化接口。

模块: vis/validation_vis/__init__.py
依赖: vis.validation_vis.validation_vis
读取配置: validation_vis.*, model_vis.*, model_data.*, model.*
对外接口:
    - generate_validation_visualizations
    - render_training_history
    - render_validation_data
"""

from vis.validation_vis.validation_vis import (
    generate_validation_visualizations, render_training_history, render_validation_data,
)

__all__ = ["generate_validation_visualizations", "render_training_history", "render_validation_data"]

