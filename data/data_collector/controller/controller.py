"""使用 Jacobian IK 和夹爪状态机执行受控任务 AST。

模块: data/data_collector/controller/controller.py
依赖: numpy, config, data.data_collector.records, data.data_collector.simulation
读取配置: data_collector.controller.*, data_collector.scene.slope_size,
    data_collector.tasks.stable_frames, data_collector.tasks.region_tolerance,
    data_collector.tasks.velocity_tolerance, data_collector.tasks.slide_distance
对外接口:
    - ScriptedExpert
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config.schema import AppConfig
from data.data_collector.controller.checks.controller_checks import check_controller_inputs
from data.data_collector.records import ActionStep, SceneSpec
from data.data_collector.simulation import EmbodiedSimulator


class ScriptedExpert:
    """把任务 AST 编译为可复现的末端运动和夹爪动作。"""

    def __init__(self, simulator: EmbodiedSimulator, spec: SceneSpec, cfg: AppConfig):
        check_controller_inputs(simulator, spec)
        self.sim = simulator
        self.spec = spec
        self.cfg = cfg
        self.settings = cfg.data_collector.controller
        self.home_rotation = simulator.ee_rotation
        self.gripper_width = self.settings.gripper_open
        self.slide_start: dict[str, np.ndarray] = {}
        self.completed_places: dict[str, str] = {}

    @staticmethod
    def _orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 0.5 * sum((np.cross(current[:, axis], target[:, axis]) for axis in range(3)), start=np.zeros(3))

    def _move(
        self,
        target_position: np.ndarray,
        phase: str,
        target_rotation: np.ndarray | None = None,
        position_tolerance: float | None = None,
        require_orientation: bool = True,
    ) -> bool:
        rotation = self.home_rotation if target_rotation is None else target_rotation
        arrival_tolerance = self.settings.position_tolerance if position_tolerance is None else position_tolerance
        grasp_loss_streak = 0
        for _ in range(self.settings.phase_frames):
            position_error = np.asarray(target_position) - self.sim.ee_position
            orientation_error = self._orientation_error(self.sim.ee_rotation, rotation)
            error = np.concatenate([position_error, self.settings.orientation_weight * orientation_error])
            jacobian = self.sim.jacobian()[:, self.sim.arm_dofs]
            regularizer = (self.settings.ik_damping**2) * np.eye(jacobian.shape[0])
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + regularizer, error)
            step_limit = self.settings.held_max_joint_step if self.sim.held_object is not None else self.settings.max_joint_step
            delta = np.clip(self.settings.ik_gain * delta, -step_limit, step_limit)
            targets = self.sim.arm_qpos + delta
            self.sim.set_controls(targets, self.gripper_width)
            self.sim.step()
            held_object = self.sim.held_object
            retained = held_object is None or self.sim.physical_grasp_is_retained(held_object)
            forces = self.sim.grasp_contact_forces(held_object) if held_object is not None else {"left": 0.0, "right": 0.0}
            self.sim.capture(phase, {
                "delta_pose_base": error.astype(np.float32),
                "joint_position_target": targets.astype(np.float32),
                "gripper_width_target": self.gripper_width,
                "physical_grasp_retained": retained,
                "finger_normal_force": forces,
            })
            if held_object is not None:
                grasp_loss_streak = 0 if retained else grasp_loss_streak + 1
                if grasp_loss_streak >= self.settings.grasp_loss_frames:
                    self.sim.release()
                    return False
            position_error_after_step = np.asarray(target_position) - self.sim.ee_position
            orientation_error_after_step = self._orientation_error(self.sim.ee_rotation, rotation)
            orientation_reached = not require_orientation or np.linalg.norm(orientation_error_after_step) <= self.settings.orientation_tolerance
            if np.linalg.norm(position_error_after_step) <= arrival_tolerance and orientation_reached:
                return True
        return False

    def _hold_gripper(self, width: float, phase: str) -> None:
        self.gripper_width = width
        for _ in range(self.settings.gripper_frames):
            targets = self.sim.arm_qpos
            self.sim.set_controls(targets, width)
            self.sim.step()
            self.sim.capture(phase, {
                "delta_pose_base": np.zeros(6, dtype=np.float32),
                "joint_position_target": targets.astype(np.float32),
                "gripper_width_target": width,
            })

    def _close_on_object(self, object_name: str, phase: str) -> bool:
        """闭合夹爪，并要求双指法向接触连续稳定后登记抓取。"""
        self.gripper_width = self.settings.gripper_closed
        contact_streak = 0
        for _ in range(self.settings.gripper_frames):
            targets = self.sim.arm_qpos
            self.sim.set_controls(targets, self.gripper_width)
            self.sim.step()
            forces = self.sim.grasp_contact_forces(object_name)
            bilateral = all(force >= self.settings.grasp_min_normal_force for force in forces.values())
            geometry_valid = self.sim.object_in_grasp_capture_region(object_name)
            contact_streak = contact_streak + 1 if bilateral and geometry_valid else 0
            self.sim.capture(phase, {
                "delta_pose_base": np.zeros(6, dtype=np.float32),
                "joint_position_target": targets.astype(np.float32),
                "gripper_width_target": self.gripper_width,
                "finger_normal_force": forces,
                "bilateral_contact": bilateral,
                "grasp_validation_streak": contact_streak,
            })
        return contact_streak >= self.settings.grasp_validation_frames and self.sim.claim_physical_grasp(object_name)

    def _verify_lifted_grasp(self, object_name: str, initial_height: float) -> bool:
        """以实际物体抬升量和连续双指接触验证抓取，而不施加绑定。"""
        validation_streak = 0
        for _ in range(self.settings.grasp_validation_frames):
            targets = self.sim.arm_qpos
            self.sim.set_controls(targets, self.gripper_width)
            self.sim.step()
            forces = self.sim.grasp_contact_forces(object_name)
            lifted_height = float(self.sim.object_position(object_name)[2] - initial_height)
            bilateral = all(force >= self.settings.grasp_min_normal_force for force in forces.values())
            retained = self.sim.physical_grasp_is_retained(object_name)
            valid = bilateral and retained and lifted_height >= self.settings.grasp_lift_min_height
            validation_streak = validation_streak + 1 if valid else 0
            self.sim.capture(f"VERIFY_PHYSICAL_GRASP_{object_name}", {
                "delta_pose_base": np.zeros(6, dtype=np.float32),
                "joint_position_target": targets.astype(np.float32),
                "gripper_width_target": self.gripper_width,
                "finger_normal_force": forces,
                "bilateral_contact": bilateral,
                "physical_grasp_retained": retained,
            }, {
                "physical_grasp": valid,
                "object_lifted_height": lifted_height,
                "validation_streak": validation_streak,
            })
        if validation_streak < self.settings.grasp_validation_frames:
            self.sim.release()
            return False
        return True

    def _home(self, phase: str) -> bool:
        """以关节空间轨迹脱离上一次任务可能留下的奇异位形。"""
        target = self.sim.home_arm_qpos
        for _ in range(self.settings.phase_frames):
            self.sim.set_controls(target, self.gripper_width)
            self.sim.step()
            self.sim.capture(phase, {
                "delta_pose_base": np.zeros(6, dtype=np.float32),
                "joint_position_target": target.astype(np.float32),
                "gripper_width_target": self.gripper_width,
            })
            if np.linalg.norm(self.sim.arm_qpos - target) <= self.settings.joint_home_tolerance:
                return True
        return False

    def _pick(self, step: ActionStep) -> bool:
        if not self._home(f"HOME_BEFORE_{step.object_ref}"):
            return False
        object_position = self.sim.object_position(step.object_ref)
        initial_object_height = float(object_position[2])
        grasp_position = object_position + np.asarray([0.0, 0.0, self.settings.grasp_height_offset])
        approach = grasp_position + np.asarray([0.0, 0.0, self.settings.approach_height])
        self._move(approach, f"APPROACH_{step.object_ref}")
        self._move(
            grasp_position,
            f"DESCEND_{step.object_ref}",
            position_tolerance=self.settings.grasp_position_tolerance,
        )
        # 抓取成立与否只由双指接触、摩擦和后续实际抬升决定。
        if not self._close_on_object(step.object_ref, f"CLOSE_{step.object_ref}"):
            return False
        lift = self.sim.ee_position + np.asarray([0.0, 0.0, self.settings.lift_height])
        self._move(lift, f"LIFT_{step.object_ref}")
        if self.sim.held_object != step.object_ref:
            return False
        return self._verify_lifted_grasp(step.object_ref, initial_object_height)

    def _target_object_position(self, step: ActionStep) -> np.ndarray:
        held_spec = next(obj for obj in self.spec.objects if obj.name == step.object_ref)
        if step.target_ref in {obj.name for obj in self.spec.objects}:
            base_spec = next(obj for obj in self.spec.objects if obj.name == step.target_ref)
            target = self.sim.object_position(step.target_ref)
            return target + np.asarray([0.0, 0.0, base_spec.size + held_spec.size])
        target = np.asarray(self.spec.target_positions[step.target_ref], dtype=np.float64)
        if step.target_ref == "slope":
            target[2] += self.cfg.data_collector.scene.slope_size[2] + held_spec.size
        return target

    def _place(self, step: ActionStep) -> bool:
        if self.sim.held_object != step.object_ref:
            return False
        object_target = self._target_object_position(step)
        target_rotation = self.home_rotation.copy()
        if step.qualifier == "UPRIGHT":
            target_rotation = self.home_rotation
        ee_target = self.sim.object_target_to_ee(object_target, target_rotation)
        approach = ee_target + np.asarray([0.0, 0.0, self.settings.transport_clearance])
        approach[2] = min(approach[2], self.settings.transport_max_height)
        transfer_start = self.sim.ee_position
        for waypoint_index in range(1, self.settings.transport_waypoints + 1):
            fraction = waypoint_index / self.settings.transport_waypoints
            waypoint = transfer_start + fraction * (approach - transfer_start)
            if not self._move(
                waypoint,
                f"TRANSFER_{step.object_ref}_TO_{step.target_ref}_{waypoint_index}",
                target_rotation,
                position_tolerance=self.settings.transport_position_tolerance,
                require_orientation=False,
            ):
                return False
        if not self._move(
            ee_target,
            f"PLACE_{step.object_ref}_ON_{step.target_ref}",
            target_rotation,
            require_orientation=step.qualifier == "UPRIGHT",
        ):
            return False
        # 先张开夹爪再清除逻辑状态；物体释放过程仍完全由接触动力学决定。
        self._hold_gripper(self.settings.gripper_open, f"OPEN_{step.object_ref}")
        self.sim.release()
        self.completed_places[step.object_ref] = step.target_ref
        self._move(approach, f"RETREAT_{step.object_ref}", target_rotation)
        return True

    def _wait(self, step: ActionStep) -> bool:
        self.slide_start[step.object_ref] = self.sim.object_position(step.object_ref)
        direction = np.asarray(self.settings.slide_direction, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        slope_center = np.asarray(self.spec.target_positions["slope"], dtype=np.float64)
        object_spec = next(obj for obj in self.spec.objects if obj.name == step.object_ref)
        slide_axis = int(np.argmax(np.abs(direction[:2])))
        off_slope_distance = self.cfg.data_collector.scene.slope_size[slide_axis] + object_spec.size
        forcing_slide = True
        stable_streak = 0
        reached = False
        for _ in range(self.settings.wait_frames):
            if forcing_slide:
                self.sim.set_object_linear_velocity(step.object_ref, self.settings.slide_initial_speed * direction)
            targets = self.sim.arm_qpos
            self.sim.set_controls(targets, self.gripper_width)
            self.sim.step()
            object_position = self.sim.object_position(step.object_ref)
            distance = float(np.linalg.norm(object_position - self.slide_start[step.object_ref]))
            travel_from_slope_center = float(np.dot(object_position - slope_center, direction))
            off_slope = travel_from_slope_center >= off_slope_distance
            if forcing_slide and off_slope:
                # 仅在斜面阶段提供确定性初始滑动；越过边缘后保留瞬时速度并完全交还物理引擎。
                self.sim.clear_object_linear_velocity(step.object_ref, zero_velocity=False)
                forcing_slide = False
            effective_speed = self.sim.object_effective_speed(step.object_ref)
            stable = effective_speed <= self.cfg.data_collector.tasks.velocity_tolerance
            stable_streak = stable_streak + 1 if off_slope and stable else 0
            self.sim.capture(f"WAIT_{step.event}", {
                "delta_pose_base": np.zeros(6, dtype=np.float32),
                "joint_position_target": targets.astype(np.float32),
                "gripper_width_target": self.gripper_width,
            }, {
                "slide_distance": distance,
                "off_slope": off_slope,
                "effective_speed": effective_speed,
                "stable_streak": stable_streak,
            })
            if distance >= self.cfg.data_collector.tasks.slide_distance and stable_streak >= self.cfg.data_collector.tasks.stable_frames:
                reached = True
                break
        self.sim.clear_object_linear_velocity(step.object_ref)
        return reached

    def _settle(self) -> None:
        for _ in range(self.settings.settle_control_frames):
            targets = self.sim.arm_qpos
            self.sim.set_controls(targets, self.gripper_width)
            self.sim.step()
            self.sim.capture("SETTLE", {
                "delta_pose_base": np.zeros(6, dtype=np.float32),
                "joint_position_target": targets.astype(np.float32),
                "gripper_width_target": self.gripper_width,
            })

    def _success_evidence(self) -> dict[str, Any]:
        tolerance = self.cfg.data_collector.tasks.region_tolerance
        velocity_tolerance = self.cfg.data_collector.tasks.velocity_tolerance
        placements: dict[str, Any] = {}
        for object_name, target_name in self.completed_places.items():
            position = self.sim.object_position(object_name)
            if target_name in {obj.name for obj in self.spec.objects}:
                target_position = self.sim.object_position(target_name)
                reached = position[2] > target_position[2]
            else:
                target_position = np.asarray(self.spec.target_positions[target_name])
                reached = np.linalg.norm(position[:2] - target_position[:2]) <= tolerance
            stable = self.sim.object_effective_speed(object_name) <= velocity_tolerance
            placements[object_name] = {"target": target_name, "reached": bool(reached), "stable": bool(stable)}
        slide_ok = True
        if self.spec.task.task_type == "SLIDE_REGRASP":
            object_name = self.spec.objects[0].name
            slide_ok = object_name in self.slide_start and np.linalg.norm(self.sim.object_position(object_name) - self.slide_start[object_name]) >= self.cfg.data_collector.tasks.slide_distance
        success = bool(placements and all(item["reached"] and item["stable"] for item in placements.values()) and slide_ok)
        return {"success": success, "placements": placements, "slide_event": bool(slide_ok), "task_type": self.spec.task.task_type}

    def run(self) -> dict[str, Any] | None:
        """执行整个任务；失败返回 None，成功返回可审计证据。"""
        dispatch = {"PICK": self._pick, "PLACE": self._place, "WAIT": self._wait}
        for step in self.spec.task.steps:
            if not dispatch[step.verb](step):
                return None
        self._settle()
        evidence = self._success_evidence()
        if not evidence["success"]:
            return None
        self.sim.frames[-1].success_state = evidence
        return evidence


__all__ = ["ScriptedExpert"]
