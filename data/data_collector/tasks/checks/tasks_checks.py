SUPPORTED_TASKS = {"PICK_PLACE", "SORT", "SLIDE_REGRASP", "STACK", "SEQUENTIAL_REARRANGE", "ORIENT_AND_PLACE"}
MULTI_OBJECT_TASKS = {"SORT", "STACK", "SEQUENTIAL_REARRANGE"}


def minimum_object_count(task_type: str) -> int:
    return 2 if task_type in MULTI_OBJECT_TASKS else 1


def check_task_inputs(task_type, objects) -> None:
    # 校验对象: build_task 的 task_type —— 任务类型必须受支持。
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"不支持的任务类型: {task_type}")
    # 校验对象: build_task 的 objects —— 多物体任务至少两个对象，其余任务至少一个。
    if len(objects) < minimum_object_count(task_type):
        raise ValueError("任务对象数量不足")
