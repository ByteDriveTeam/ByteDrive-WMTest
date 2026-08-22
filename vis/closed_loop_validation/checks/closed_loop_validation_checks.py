import os
from pathlib import Path
import sys

from config import PROJECT_ROOT


def check_closed_loop_output(output: Path) -> None:
    # 校验对象: run_fixed_closed_loop_validation 的输出目录 —— 所有产物必须位于项目内。
    try:
        output.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError("闭环验证输出必须位于项目目录内") from error


def check_closed_loop_history(frames: list, first_index: int) -> None:
    # 校验对象: 在线PolicyBatch的历史帧 —— 必须覆盖完整窗口并包含两路RGB和触觉几何。
    if first_index < 0:
        raise ValueError("闭环观测历史不足")
    required_cameras = {"overview", "wrist"}
    if any(not required_cameras.issubset(frame.cameras) for frame in frames[first_index:]):
        raise ValueError("闭环观测缺少overview或wrist相机")
    if any("patch_geometry_base" not in frame.tactile for frame in frames[first_index:]):
        raise ValueError("闭环观测缺少触觉Patch几何")


def check_closed_loop_rendering(cfg) -> None:
    # 校验对象: Linux闭环Renderer —— 后端和EGL设备必须在导入MuJoCo前由CLI显式设置。
    if not sys.platform.startswith("linux"):
        return
    settings = cfg.model_data.replay_cache
    if os.environ.get("MUJOCO_GL") != settings.linux_render_backend:
        raise RuntimeError("Linux闭环验证的MUJOCO_GL未按配置显式设置；请在导入MuJoCo前调用configure_mujoco_rendering")
    if settings.linux_render_backend == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(settings.linux_egl_device_id):
        raise RuntimeError("Linux闭环验证的MUJOCO_EGL_DEVICE_ID未按配置显式设置")


def check_sensor_archive_replay(source: Path, output: Path, arrays) -> None:
    # 校验对象: render_sensor_archive_to_mp4 的NPZ与输出 —— 路径须在项目内且逐帧控制字段完整。
    check_closed_loop_output(source)
    check_closed_loop_output(output)
    required = {"time", "joint_position", "gripper_width", "model_action"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"传感器归档缺少字段: {sorted(missing)}")
    frame_count = len(arrays["time"])
    if frame_count == 0:
        raise ValueError("传感器归档没有可重放帧")
    expected = {
        "joint_position": (frame_count, 7),
        "gripper_width": (frame_count, 2),
        "model_action": (frame_count, 9),
    }
    invalid = {name: arrays[name].shape for name, shape in expected.items() if arrays[name].shape != shape}
    if invalid:
        raise ValueError(f"传感器归档字段形状不合法: {invalid}")


__all__ = [
    "check_closed_loop_history", "check_closed_loop_output", "check_closed_loop_rendering",
    "check_sensor_archive_replay",
]
