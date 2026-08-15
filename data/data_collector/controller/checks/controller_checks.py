def check_controller_inputs(simulator, spec) -> None:
    # 校验对象: ScriptedExpert 的 simulator/spec —— 模型对象名必须覆盖任务对象。
    model_names = {simulator.model.body(body_id).name for body_id in range(simulator.model.nbody)}
    expected = {obj.name for obj in spec.objects}
    if not expected.issubset(model_names):
        raise ValueError("仿真模型缺少任务对象")
    # 校验对象: ScriptedExpert 的任务步骤 —— 只接受脚本专家已实现的动词。
    if any(step.verb not in {"PICK", "PLACE", "WAIT"} for step in spec.task.steps):
        raise ValueError("任务包含脚本专家不支持的动作")

