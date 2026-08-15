"""生成和解析无特权坐标的受控任务语言。

模块: data/data_collector/task_language/task_language.py
依赖: data.data_collector.records
读取配置: 无
对外接口:
    - format_instruction(steps, objects) -> str
    - parse_instruction(text) -> list[ActionStep]
    - describe_scene(objects, relations) -> str
"""

from __future__ import annotations

from data.data_collector.records import ActionStep, ObjectSpec
from data.data_collector.task_language.checks.task_language_checks import check_instruction


def _selector(obj: ObjectSpec) -> str:
    return f"{obj.color} {obj.shape}"


def format_instruction(steps: list[ActionStep], objects: list[ObjectSpec]) -> str:
    """把动作 AST 格式化为固定词表的受控祈使句。"""
    object_map = {obj.name: obj for obj in objects}
    clauses: list[str] = []
    for step in steps:
        if step.verb == "PICK":
            suffix = " AGAIN" if step.qualifier == "AGAIN" else ""
            selector = _selector(object_map[step.object_ref]) if step.object_ref in object_map else step.object_ref
            alias = f" AS {step.object_ref}" if step.object_ref in object_map and not suffix else ""
            clauses.append(f"PICK {selector}{alias}{suffix}")
        elif step.verb == "PLACE":
            qualifier = f" {step.qualifier}" if step.qualifier else ""
            clauses.append(f"PLACE {step.object_ref} {step.relation} {step.target_ref}{qualifier}")
        elif step.verb == "WAIT":
            clauses.append(f"WAIT UNTIL {step.object_ref} {step.event} {step.target_ref}".rstrip())
        else:
            raise ValueError(f"未知动作词: {step.verb}")
    text = "; ".join(clauses) + "."
    check_instruction(text)
    return text


def parse_instruction(text: str) -> list[ActionStep]:
    """解析由本模块生成的规范指令，供数据校验和外部消费。"""
    check_instruction(text)
    steps: list[ActionStep] = []
    for clause in text.rstrip(".").split("; "):
        tokens = clause.split()
        if tokens[0] == "WAIT":
            steps.append(ActionStep("WAIT", tokens[2], target_ref=tokens[4] if len(tokens) > 4 else "", event=tokens[3]))
        elif tokens[0] == "PLACE":
            steps.append(ActionStep("PLACE", tokens[1], tokens[3], tokens[2], qualifier=" ".join(tokens[4:])))
        elif tokens[0] == "PICK":
            again = tokens[-1] == "AGAIN"
            alias = tokens[tokens.index("AS") + 1] if "AS" in tokens else tokens[1]
            steps.append(ActionStep("PICK", alias, qualifier="AGAIN" if again else ""))
    return steps


def describe_scene(objects: list[ObjectSpec], relations: dict[str, str]) -> str:
    """生成只包含公开属性和定性关系的逐帧场景描述。"""
    descriptions = [f"{obj.name} IS {obj.color} {obj.shape} AND {relations.get(obj.name, 'ON table')}" for obj in objects]
    return "SCENE HAS " + "; ".join(descriptions) + "."


__all__ = ["describe_scene", "format_instruction", "parse_instruction"]

