import re


def check_instruction(text: str) -> None:
    # 校验对象: 规范任务指令 —— 禁止出现连续坐标或浮点特权信息。
    if re.search(r"[-+]?\d+\.\d+", text):
        raise ValueError("规范任务指令不得包含连续坐标或精确浮点值")
    # 校验对象: 规范任务指令 —— 只接受固定动作词开头的子句。
    clauses = text.rstrip(".").split("; ")
    if not clauses or any(clause.split()[0] not in {"PICK", "PLACE", "WAIT"} for clause in clauses):
        raise ValueError("规范任务指令包含未知动作词")

