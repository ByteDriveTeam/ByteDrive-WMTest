def check_simulator_inputs(spec, mjcf_xml) -> None:
    # 校验对象: EmbodiedSimulator 的 spec —— 场景必须包含任务和对象。
    if spec.task is None or not spec.objects:
        raise ValueError("仿真场景缺少任务或对象")
    # 校验对象: EmbodiedSimulator 的 mjcf_xml —— 必须是 MuJoCo XML。
    if "<mujoco" not in mjcf_xml or "__PANDA_ASSET_DIR__" not in mjcf_xml:
        raise ValueError("MJCF 缺少根元素或资产占位符")


def check_tactile_geometry_inputs(settings, patch: int) -> None:
    # 校验对象: compute_tactile_geometry 的 patch —— 必须完整划分触觉网格。
    if patch <= 0 or any(size % patch for size in settings.tactile_resolution):
        raise ValueError("触觉Patch必须为正且完整划分触觉分辨率")
