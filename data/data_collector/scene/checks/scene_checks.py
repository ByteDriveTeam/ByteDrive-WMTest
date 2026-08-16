import numpy as np


def check_scene_spec(spec, cfg) -> None:
    # 校验对象: generate_scene_spec 的场景对象 —— 名称必须唯一。
    names = [obj.name for obj in spec.objects]
    if len(names) != len(set(names)):
        raise ValueError("场景对象名称重复")
    # 校验对象: generate_scene_spec 的场景对象 —— 颜色必须一物一色。
    colors = [obj.color for obj in spec.objects]
    if len(colors) != len(set(colors)):
        raise ValueError("场景对象颜色重复")
    # 校验对象: generate_scene_spec 的初始布局 —— 对象间距必须满足配置。
    xy = np.asarray([obj.initial_position[:2] for obj in spec.objects], dtype=np.float64)
    if len(xy) > 1:
        distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        distances += np.eye(len(xy)) * cfg.data_collector.scene.minimum_object_spacing
        if distances.min() < cfg.data_collector.scene.minimum_object_spacing:
            raise ValueError("场景对象初始位置重叠")
