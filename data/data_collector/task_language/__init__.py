"""重导出受控任务语言接口。

模块: data/data_collector/task_language/__init__.py
依赖: data.data_collector.task_language.task_language
读取配置: 无
对外接口:
    - format_instruction(steps, objects) -> str
    - parse_instruction(text) -> list[ActionStep]
    - describe_scene(objects, relations) -> str
"""

from data.data_collector.task_language.task_language import describe_scene, format_instruction, parse_instruction

__all__ = ["describe_scene", "format_instruction", "parse_instruction"]

