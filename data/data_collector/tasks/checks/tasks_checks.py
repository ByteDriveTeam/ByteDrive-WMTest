def check_task_inputs(task_type, objects) -> None:
    # 校验对象: build_task 的 task_type —— 任务类型必须受支持。
    supported = {"PICK_PLACE", "SORT", "SLIDE_REGRASP", "STACK", "SEQUENTIAL_REARRANGE", "ORIENT_AND_PLACE"}
    if task_type not in supported:
        raise ValueError(f"不支持的任务类型: {task_type}")
    # 校验对象: build_task 的 objects —— 堆叠需要至少两个对象，其余任务至少一个。
    if not objects or (task_type == "STACK" and len(objects) < 2):
        raise ValueError("任务对象数量不足")

