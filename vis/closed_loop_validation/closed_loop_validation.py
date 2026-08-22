"""运行固定抓取放置闭环验证，并保存传感器记录与MP4。

模块: vis/closed_loop_validation/closed_loop_validation.py
依赖: ffmpeg, mujoco, numpy, pillow, torch, config, data.data_collector,
    data.model_dataset, model.policy, vis.closed_loop_validation.checks
读取配置: validation_vis.closed_loop_*, validation_vis.output, model_vis.flow_noise_seed,
    model_data.*, model.tactile_patch, data_collector.simulation.*,
    data_collector.controller.gripper_*, data_collector.render.*, data_collector.sensors.*,
    data_collector.scene.*, data_collector.tasks.*, data_vis.*, training.device
对外接口:
    - run_fixed_closed_loop_validation(model, stats, cfg, epoch) -> dict
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch

from config import PROJECT_ROOT
from config.schema import AppConfig
from data.data_collector.scene import build_mjcf, generate_scene_spec
from data.data_collector.simulation import EmbodiedSimulator
from data.model_dataset import ClosedLanguageTokenizer, NormalizationStats, sampling_times, state_vector
from model.policy import ByteDrivePolicy, PolicyBatch, sensor_token_counts
from vis.closed_loop_validation.checks import (
    check_closed_loop_history, check_closed_loop_output, check_closed_loop_rendering,
)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _rollout_config(cfg: AppConfig) -> AppConfig:
    cameras = [replace(camera, modalities=["rgb"]) for camera in cfg.data_collector.render.cameras]
    render = replace(cfg.data_collector.render, enabled=True, viewer=False, cameras=cameras)
    sensors = replace(cfg.data_collector.sensors, contact_enabled=True)
    collector = replace(cfg.data_collector, render=render, sensors=sensors)
    return replace(cfg, data_collector=collector)


def _source_hz(cfg: AppConfig) -> int:
    simulation = cfg.data_collector.simulation
    return round(1.0 / (simulation.timestep * simulation.control_substeps))


def _history_indices(frame_count: int, cfg: AppConfig) -> tuple[np.ndarray, np.ndarray]:
    source_hz = _source_hz(cfg)
    sensor_step = source_hz // cfg.model_data.sensor_hz
    sensor_count = round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz)
    rgb_step = source_hz // cfg.model_data.rgb_hz
    rgb_count = round(cfg.model_data.history_seconds * cfg.model_data.rgb_hz)
    first = frame_count - 1 - (sensor_count - 1) * sensor_step
    sensor = first + np.arange(sensor_count) * sensor_step
    rgb = first + np.arange(rgb_count) * rgb_step
    return rgb, sensor


def _build_policy_batch(frames: list, instruction: str, stats: NormalizationStats, cfg: AppConfig) -> PolicyBatch:
    rgb_indices, sensor_indices = _history_indices(len(frames), cfg)
    check_closed_loop_history(frames, int(sensor_indices[0]))
    selected_rgb = [frames[int(index)] for index in rgb_indices]
    selected_sensor = [frames[int(index)] for index in sensor_indices]

    def camera_tensor(name: str, key: str) -> torch.Tensor:
        values = [frame.cameras[name][key] for frame in selected_rgb]
        tensor = torch.from_numpy(np.stack(values))
        return tensor.permute(0, 3, 1, 2) if key == "rgb" else tensor

    force_maps = torch.from_numpy(np.stack([
        np.stack([frame.tactile["force_maps"][side].transpose(2, 0, 1) for side in ("left", "right")])
        for frame in selected_sensor
    ])).float()
    tactile_mean = torch.tensor(stats.tactile_map_mean).view(1, 1, 3, 1, 1)
    tactile_std = torch.tensor(stats.tactile_map_std).view(1, 1, 3, 1, 1)
    raw_state = torch.from_numpy(np.stack([state_vector(frame.to_dict()) for frame in selected_sensor])).float()
    state = (raw_state - torch.tensor(stats.state_mean)) / torch.tensor(stats.state_std)
    language_ids, language_valid = ClosedLanguageTokenizer(
        cfg.model_data.language_length, cfg.model_data.language_vocabulary,
    ).encode(instruction)
    counts = sensor_token_counts(cfg)
    _, _, future_time = sampling_times(cfg)

    def batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0)

    return PolicyBatch(
        overview_rgb=batch(camera_tensor("overview", "rgb")),
        wrist_rgb=batch(camera_tensor("wrist", "rgb")),
        tactile=batch((force_maps - tactile_mean) / tactile_std),
        state=batch(state),
        language_ids=batch(language_ids), language_valid=batch(language_valid),
        overview_intrinsics=batch(camera_tensor("overview", "K")),
        overview_transform=batch(camera_tensor("overview", "T_base_camera")),
        wrist_intrinsics=batch(camera_tensor("wrist", "K")),
        wrist_transform=batch(camera_tensor("wrist", "T_base_camera")),
        tactile_geometry=batch(torch.from_numpy(np.stack([
            frame.tactile["patch_geometry_base"] for frame in selected_sensor
        ])).float()),
        state_geometry=batch(torch.from_numpy(np.stack([
            frame.robot["frames"]["ee_site"]["position_base"] for frame in selected_sensor
        ])).float()),
        coordinate_bounds=batch(torch.tensor(stats.coordinate_bounds, dtype=torch.float32)),
        rgb_time=batch(torch.from_numpy(sampling_times(cfg)[0])),
        sensor_time=batch(torch.from_numpy(sampling_times(cfg)[1])),
        future_time=batch(torch.from_numpy(future_time)),
        sensor_mask=torch.zeros((1, sum(counts)), dtype=torch.bool),
        task_patch_mask=torch.zeros((1, sum(counts)), dtype=torch.bool),
        behavior_valid=torch.ones((1, len(future_time)), dtype=torch.bool),
        phase_target=torch.full((1,), -100, dtype=torch.long),
        cache_hits=torch.zeros(1, dtype=torch.long), cache_misses=torch.zeros(1, dtype=torch.long),
        flow_target=None,
    )


def _capture(simulator: EmbodiedSimulator, joint_targets: np.ndarray, gripper_width: float, phase: str, action: dict[str, Any]) -> None:
    simulator.set_controls(joint_targets, gripper_width)
    simulator.step()
    simulator.capture(phase, action)


def _task_status(simulator: EmbodiedSimulator, stable_streak: int, cfg: AppConfig) -> tuple[str | None, int, dict[str, float | bool]]:
    target = simulator.spec.task.steps[0].object_ref
    position = simulator.object_position(target)
    target_position = np.asarray(simulator.spec.target_positions["center_zone"])
    distance = float(np.linalg.norm(position[:2] - target_position[:2]))
    speed = simulator.object_effective_speed(target)
    gripper = float(np.mean(simulator.gripper_width))
    settings = cfg.data_collector.scene
    on_table = 0.0 <= position[0] <= settings.table_size[0] and abs(position[1]) <= settings.table_size[1] / 2
    finite = bool(np.isfinite(simulator.data.qpos).all() and np.isfinite(simulator.data.qvel).all())
    released = gripper >= (
        cfg.data_collector.controller.gripper_open + cfg.data_collector.controller.gripper_closed
    ) / 2
    stable = distance <= cfg.data_collector.tasks.region_tolerance and speed <= cfg.data_collector.tasks.velocity_tolerance and released
    stable_streak = stable_streak + 1 if stable else 0
    metrics = {"target_distance": distance, "target_speed": speed, "released": released}
    if not finite:
        return "nonfinite_physics", stable_streak, metrics
    if position[2] < 0.0 or not on_table:
        return "object_out_of_workspace", stable_streak, metrics
    if stable_streak >= cfg.data_collector.tasks.stable_frames:
        return "success", stable_streak, metrics
    return None, stable_streak, metrics


def _tactile_image(values: np.ndarray, cfg: AppConfig) -> Image.Image:
    force = np.power(
        np.clip(values[..., 0] / cfg.data_vis.tactile_force_max, 0.0, 1.0),
        cfg.data_vis.tactile_force_gamma,
    )
    rgb = np.stack((np.clip(2 * force, 0, 1), np.clip(2 * force - 0.5, 0, 1), np.clip(1.5 * force, 0, 1)), -1)
    return Image.fromarray((255 * rgb).astype(np.uint8), mode="RGB")


def _compose_video_frame(frame, status: str, cfg: AppConfig) -> np.ndarray:
    overview = Image.fromarray(frame.cameras["overview"]["rgb"], mode="RGB")
    wrist = Image.fromarray(frame.cameras["wrist"]["rgb"], mode="RGB")
    camera_width = max(overview.width, wrist.width)
    width = camera_width + cfg.data_vis.panel_width
    height = max(overview.height + wrist.height, cfg.data_vis.panel_min_height)
    width, height = width + width % 2, height + height % 2
    canvas = Image.new("RGB", (width, height), tuple(cfg.data_vis.background_rgb))
    canvas.paste(overview, (0, 0))
    canvas.paste(wrist, (0, overview.height))
    draw = ImageDraw.Draw(canvas)
    x = camera_width + cfg.data_vis.padding
    lines = (
        f"Fixed PICK_PLACE | {status}",
        f"frame={frame.frame_index} time={frame.simulation_time:.2f}s",
        f"phase={frame.phase}",
        f"gripper={float(np.mean(frame.robot['gripper_width'])):.4f}m",
    )
    for index, line in enumerate(lines):
        draw.text((x, cfg.data_vis.padding + index * cfg.data_vis.text_line_height), line, fill=tuple(cfg.data_vis.text_rgb))
    maps = frame.tactile["force_maps"]
    map_size = min(cfg.data_vis.tactile_map_size, (cfg.data_vis.panel_width - 3 * cfg.data_vis.padding) // 2)
    map_top = cfg.data_vis.padding + (len(lines) + 1) * cfg.data_vis.text_line_height
    for index, side in enumerate(("left", "right")):
        map_x = x + index * (map_size + cfg.data_vis.padding)
        tactile = _tactile_image(maps[side], cfg).resize((map_size, map_size), Image.Resampling.NEAREST)
        canvas.paste(tactile, (map_x, map_top))
        draw.text((map_x, map_top + map_size), side, fill=tuple(cfg.data_vis.text_rgb))
    qpos = np.asarray(frame.robot["joint_position"])
    joint_top = map_top + map_size + 2 * cfg.data_vis.text_line_height
    for index, value in enumerate(qpos):
        draw.text((x, joint_top + index * cfg.data_vis.text_line_height), f"J{index + 1}: {value:+.3f}", fill=tuple(cfg.data_vis.text_rgb))
    return np.asarray(canvas)


def _write_sensor_archive(frames: list, path: Path) -> None:
    action = np.full((len(frames), 9), np.nan, dtype=np.float32)
    for index, frame in enumerate(frames):
        if "model_action" in frame.action:
            action[index] = frame.action["model_action"]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            time=np.asarray([frame.simulation_time for frame in frames]),
            joint_position=np.stack([frame.robot["joint_position"] for frame in frames]),
            joint_velocity=np.stack([frame.robot["joint_velocity"] for frame in frames]),
            gripper_width=np.stack([frame.robot["gripper_width"] for frame in frames]),
            tactile=np.stack([
                np.stack([frame.tactile["force_maps"][side] for side in ("left", "right")])
                for frame in frames
            ]),
            model_action=action,
        )
    temporary.replace(path)


def _write_mp4(frames: list, path: Path, status: str, cfg: AppConfig) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("闭环验证生成MP4需要ffmpeg")
    selected = frames[::cfg.validation_vis.closed_loop_video_stride]
    first = _compose_video_frame(selected[0], status, cfg)
    temporary = path.with_suffix(".tmp.mp4")
    command = [
        executable, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{first.shape[1]}x{first.shape[0]}", "-r", str(cfg.validation_vis.closed_loop_video_fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        process.stdin.write(first.tobytes())
        for frame in selected[1:]:
            process.stdin.write(_compose_video_frame(frame, status, cfg).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg编码失败: {stderr.strip()}")
        temporary.replace(path)
    except Exception:
        process.kill()
        temporary.unlink(missing_ok=True)
        raise


@torch.no_grad()
def run_fixed_closed_loop_validation(
    model: ByteDrivePolicy,
    stats: NormalizationStats,
    cfg: AppConfig,
    epoch: int,
) -> dict[str, Any]:
    """在固定PICK_PLACE场景中滚动策略，失败或超时后保存完整审计产物。"""
    output = _project_path(cfg.validation_vis.output) / f"epoch_{epoch:04d}" / "closed_loop"
    check_closed_loop_rendering(cfg)
    check_closed_loop_output(output)
    output.mkdir(parents=True, exist_ok=True)
    rollout_cfg = _rollout_config(cfg)
    settings = cfg.validation_vis
    spec = generate_scene_spec(settings.closed_loop_scene_index, settings.closed_loop_attempt, rollout_cfg, settings.closed_loop_task)
    simulator = EmbodiedSimulator(spec, build_mjcf(spec, rollout_cfg), rollout_cfg)
    device = next(model.parameters()).device
    future_steps = round(cfg.model_data.future_seconds * cfg.model_data.sensor_hz)
    generator = torch.Generator().manual_seed(cfg.model_vis.flow_noise_seed)
    flow_noise = torch.randn((1, future_steps, 23), generator=generator).to(device)
    source_per_action = _source_hz(cfg) // cfg.model_data.sensor_hz
    history_frames = (round(cfg.model_data.history_seconds * cfg.model_data.sensor_hz) - 1) * source_per_action + 1
    status, stable_streak, rollout_frames = "timeout", 0, 0
    terminal_metrics: dict[str, float | bool] = {}
    joint_ranges = simulator.arm_joint_ranges
    try:
        for _ in range(history_frames):
            _capture(simulator, simulator.home_arm_qpos, cfg.data_collector.controller.gripper_open, "WARMUP", {})
        while rollout_frames < settings.closed_loop_max_control_frames:
            batch = _build_policy_batch(simulator.frames, spec.task.instruction, stats, cfg).to(device)
            actions = model.predict(batch, flow_noise=flow_noise)["actions"][0].cpu().numpy()
            horizon = min(settings.closed_loop_replan_action_steps, len(actions))
            terminal = None
            for action in actions[:horizon]:
                if not np.isfinite(action).all():
                    terminal = "nonfinite_action"
                    break
                joint_targets = action[:7]
                if np.any(joint_targets < joint_ranges[:, 0]) or np.any(joint_targets > joint_ranges[:, 1]):
                    terminal = "joint_limit_action"
                    break
                gripper = cfg.data_collector.controller.gripper_open if action[8] >= 0 else cfg.data_collector.controller.gripper_closed
                for _ in range(source_per_action):
                    _capture(simulator, joint_targets, gripper, "MODEL_ROLLOUT", {"model_action": action.copy()})
                    rollout_frames += 1
                    terminal, stable_streak, terminal_metrics = _task_status(simulator, stable_streak, cfg)
                    if terminal is not None or rollout_frames >= settings.closed_loop_max_control_frames:
                        break
                if terminal is not None or rollout_frames >= settings.closed_loop_max_control_frames:
                    break
            if terminal is not None:
                status = terminal
                break
        video = output / "fixed_pick_place.mp4"
        sensors = output / "sensors.npz"
        _write_sensor_archive(simulator.frames, sensors)
        _write_mp4(simulator.frames, video, status, cfg)
        result = {
            "task": settings.closed_loop_task, "scene_index": settings.closed_loop_scene_index,
            "attempt": settings.closed_loop_attempt, "instruction": spec.task.instruction,
            "status": status, "success": status == "success", "control_frames": rollout_frames,
            "simulation_seconds": rollout_frames / _source_hz(cfg), **terminal_metrics,
            "policy_device": str(device),
            "render_backend": cfg.model_data.replay_cache.linux_render_backend,
            "render_device": cfg.model_data.replay_cache.linux_egl_device_id,
            "video": str(video), "sensors": str(sensors),
        }
        summary = output / "summary.json"
        temporary = summary.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(summary)
        result["summary"] = str(summary)
        return result
    finally:
        simulator.close()


__all__ = ["run_fixed_closed_loop_validation"]
