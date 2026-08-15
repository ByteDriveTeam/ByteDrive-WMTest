from config import PROJECT_ROOT


def check_replay_inputs(dataset, output, camera, cfg) -> None:
    # 校验对象: rerender_scene 的 dataset —— 必须存在且位于项目目录。
    root = PROJECT_ROOT.resolve()
    if not dataset.exists() or root not in dataset.parents:
        raise ValueError("二次渲染数据集必须存在于项目目录内")
    # 校验对象: rerender_scene 的 output —— 所有输出必须留在项目目录。
    if output == root or root not in output.parents:
        raise ValueError("二次渲染输出必须位于项目目录内")
    # 校验对象: rerender_scene 的 camera —— 必须对应配置中的相机。
    if camera not in {item.name for item in cfg.data_collector.render.cameras}:
        raise ValueError(f"未知相机: {camera}")

