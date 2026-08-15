"""按配置生成确定性的多类型具身任务。

模块: data/data_collector/tasks/tasks.py
依赖: numpy, config, data.data_collector.records, data.data_collector.task_language
读取配置: data_collector.tasks.*
对外接口:
    - choose_task_type(rng, cfg) -> str
    - object_count_for_task(task_type, cfg) -> int
    - build_task(task_type, objects) -> TaskSpec
"""

from __future__ import annotations

import numpy as np

from config.schema import AppConfig
from data.data_collector.records import ActionStep, ObjectSpec, TaskSpec
from data.data_collector.task_language import format_instruction
from data.data_collector.tasks.checks.tasks_checks import check_task_inputs


def choose_task_type(rng: np.random.Generator, cfg: AppConfig) -> str:
    """按集中配置权重选择任务类型。"""
    weights = cfg.data_collector.tasks.weights
    names = list(weights)
    probabilities = np.asarray(list(weights.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def object_count_for_task(task_type: str, cfg: AppConfig) -> int:
    """返回任务需要的对象数。"""
    settings = cfg.data_collector.tasks
    counts = {
        "SORT": settings.sort_object_count,
        "STACK": settings.stack_object_count,
        "SEQUENTIAL_REARRANGE": settings.sequential_object_count,
    }
    return counts.get(task_type, 1)


def _steps(task_type: str, objects: list[ObjectSpec]) -> list[ActionStep]:
    first = objects[0].name
    if task_type == "PICK_PLACE":
        return [ActionStep("PICK", first), ActionStep("PLACE", first, "center_zone", "IN")]
    if task_type == "SORT":
        return [step for index, obj in enumerate(objects) for step in (
            ActionStep("PICK", obj.name), ActionStep("PLACE", obj.name, "bin_left" if index % 2 == 0 else "bin_right", "IN"))]
    if task_type == "SLIDE_REGRASP":
        return [
            ActionStep("PICK", first), ActionStep("PLACE", first, "slope", "ON"),
            ActionStep("WAIT", first, "slope", event="SLIDES_OFF"), ActionStep("PICK", first, qualifier="AGAIN"),
            ActionStep("PLACE", first, "recovery_bin", "IN"),
        ]
    if task_type == "STACK":
        return [ActionStep("PICK", first), ActionStep("PLACE", first, objects[1].name, "ON")]
    if task_type == "SEQUENTIAL_REARRANGE":
        return [step for index, obj in enumerate(objects) for step in (
            ActionStep("PICK", obj.name), ActionStep("PLACE", obj.name, "bin_right" if index % 2 == 0 else "bin_left", "IN"))]
    if task_type == "ORIENT_AND_PLACE":
        return [ActionStep("PICK", first), ActionStep("PLACE", first, "center_zone", "IN", qualifier="UPRIGHT")]
    raise ValueError(f"未知任务类型: {task_type}")


def build_task(task_type: str, objects: list[ObjectSpec]) -> TaskSpec:
    """从对象和任务类型生成 AST 与规范指令。"""
    check_task_inputs(task_type, objects)
    steps = _steps(task_type, objects)
    return TaskSpec(task_type, steps, format_instruction(steps, objects))


__all__ = ["build_task", "choose_task_type", "object_count_for_task"]

