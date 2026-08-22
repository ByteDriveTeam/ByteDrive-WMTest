"""重导出固定闭环策略验证接口。

模块: vis/closed_loop_validation/__init__.py
依赖: vis.closed_loop_validation.closed_loop_validation
读取配置: validation_vis.closed_loop_*, validation_vis.output
对外接口:
    - render_sensor_archive_to_mp4(archive, output, cfg) -> dict
    - run_fixed_closed_loop_validation(model, stats, cfg, epoch) -> dict
"""

from vis.closed_loop_validation.closed_loop_validation import (
    render_sensor_archive_to_mp4, run_fixed_closed_loop_validation,
)

__all__ = ["render_sensor_archive_to_mp4", "run_fixed_closed_loop_validation"]
