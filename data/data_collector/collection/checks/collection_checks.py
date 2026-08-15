def check_collection_inputs(cfg, task_types) -> None:
    # 校验对象: collect_scenes 的目标成功场景数 —— 必须为正。
    if cfg.data_collector.collector.scene_count <= 0:
        raise ValueError("成功场景总数必须 > 0")
    # 校验对象: collect_scenes 的 task_types —— 显式任务必须已配置。
    configured = set(cfg.data_collector.tasks.weights)
    if any(task not in configured for task in task_types):
        raise ValueError("显式任务列表包含未配置任务")

