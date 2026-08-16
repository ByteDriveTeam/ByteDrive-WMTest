"""按配置生成确定性的多类型具身任务。

模块: data/data_collector/tasks/tasks.py
依赖: numpy, config, data.data_collector.records, data.data_collector.task_language
读取配置: data_collector.tasks.weights, data_collector.scene.object_count_min,
    data_collector.scene.object_count_max
对外接口:
    - choose_task_type(rng, cfg) -> str
    - object_count_for_task(task_type, rng, cfg) -> int
    - build_task(task_type, objects) -> TaskSpec
"""

from __future__ import annotations

import numpy as np

from config.schema import AppConfig
from data.data_collector.records import ActionStep, ObjectSpec, TaskSpec
from data.data_collector.task_language import format_instruction
from data.data_collector.tasks.checks.tasks_checks import check_task_inputs, minimum_object_count


def choose_task_type(rng: np.random.Generator, cfg: AppConfig) -> str:
    """按集中配置权重选择任务类型。"""
    weights = cfg.data_collector.tasks.weights
    names = list(weights)
    probabilities = np.asarray(list(weights.values()), dtype=np.float64)
    probabilities /= probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def object_count_for_task(task_type: str, rng: np.random.Generator, cfg: AppConfig) -> int:
    """在场景配置区间内采样满足任务语义最低要求的对象数。"""
    scene = cfg.data_collector.scene
    lower = max(scene.object_count_min, minimum_object_count(task_type))
    return int(rng.integers(lower, scene.object_count_max + 1))


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

