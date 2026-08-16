"""封装 MuJoCo 步进、状态读取、多相机渲染及可复用的 32×32 三轴触觉计算。

模块: data/data_collector/simulation/simulation.py
依赖: mujoco, numpy, config, data.data_collector.records, data.data_collector.scene,
    data.data_collector.task_language
读取配置: data_collector.simulation.*, data_collector.controller.gripper_*,
    data_collector.controller.grasp_*, data_collector.render.*,
    data_collector.sensors.*
对外接口:
    - EmbodiedSimulator
    - compute_tactile_state(model, data, settings) -> tuple[list, dict]
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import mujoco
import numpy as np

from config.schema import AppConfig, SensorSettings
from data.data_collector.records import FrameRecord, SceneSpec
from data.data_collector.scene import materialize_mjcf
from data.data_collector.simulation.checks.simulation_checks import check_simulator_inputs
from data.data_collector.task_language import describe_scene


def compute_tactile_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    settings: SensorSettings,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """根据当前 MuJoCo 求解结果生成接触记录和双指三轴触觉力图。"""
    resolution = tuple(settings.tactile_resolution)
    finger_bodies = {side: model.body(f"{side}_finger").id for side in ("left", "right")}
    tactile_sites = {side: model.site(f"{side}_tactile_site").id for side in finger_bodies}
    tactile = {side: np.zeros((*resolution, 3), dtype=np.float32) for side in finger_bodies}
    height, width = resolution
    yy, xx = np.mgrid[0:height, 0:width]
    extent_x, extent_y = settings.tactile_extent
    sigma = settings.tactile_sigma_pixels
    contacts: list[dict[str, Any]] = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        side = next((name for name, body in finger_bodies.items() if body in {body1, body2}), None)
        if side is None:
            continue
        force_contact = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force_contact)
        contact_rotation = np.asarray(contact.frame).reshape(3, 3)
        world_force = contact_rotation.T @ force_contact[:3]
        finger_body = finger_bodies[side]
        tactile_site = tactile_sites[side]
        tactile_rotation = data.site_xmat[tactile_site].reshape(3, 3)
        local_position = tactile_rotation.T @ (np.asarray(contact.pos) - data.site_xpos[tactile_site])
        local_force = (1.0 if finger_body == body2 else -1.0) * tactile_rotation.T @ world_force
        pixel_x = (local_position[0] / extent_x + 0.5) * (width - 1)
        pixel_y = (local_position[2] / extent_y + 0.5) * (height - 1)
        weights = np.exp(-((xx - pixel_x) ** 2 + (yy - pixel_y) ** 2) / (2 * sigma**2)).astype(np.float32)
        peak = float(weights.max())
        if peak > 0:
            weights /= peak
        components = np.asarray([abs(local_force[1]), local_force[0], local_force[2]], dtype=np.float32)
        tactile[side] += weights[..., None] * components
        contacts.append({
            "finger": side,
            "geom1": model.geom(contact.geom1).name,
            "geom2": model.geom(contact.geom2).name,
            "position_world": np.asarray(contact.pos, dtype=np.float32),
            "normal_world": contact_rotation[0].astype(np.float32),
            "force_contact_nt1t2": force_contact[:3].astype(np.float32),
        })
    force_clip = settings.tactile_force_clip
    return contacts, {side: np.clip(grid, -force_clip, force_clip) for side, grid in tactile.items()}


class EmbodiedSimulator:
    """管理一个场景对应的独立 MuJoCo 实例。"""

    def __init__(self, spec: SceneSpec, mjcf_xml: str, cfg: AppConfig):
        check_simulator_inputs(spec, mjcf_xml)
        self.spec = spec
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(materialize_mjcf(mjcf_xml))
        self.data = mujoco.MjData(self.model)
        self.frames: list[FrameRecord] = []
        self.held_object: str | None = None
        self._held_offset = np.zeros(3, dtype=np.float64)
        self._object_velocity_overrides: dict[str, np.ndarray] = {}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._viewer: Any = None
        self._arm_joint_ids = np.asarray([self.model.joint(f"joint{index}").id for index in range(1, 8)], dtype=np.int32)
        self._arm_qpos = self.model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dofs = self.model.jnt_dofadr[self._arm_joint_ids]
        self._finger_joint_ids = np.asarray([self.model.joint("finger_joint1").id, self.model.joint("finger_joint2").id])
        self._ee_site = self.model.site("ee_site").id
        self._base_body = self.model.body("link0").id
        self._object_bodies = {obj.name: self.model.body(obj.name).id for obj in spec.objects}
        self._object_joints = {obj.name: self.model.joint(f"{obj.name}_free").id for obj in spec.objects}
        self._finger_bodies = {side: self.model.body(f"{side}_finger").id for side in ("left", "right")}
        self._sensor_sites = [f"joint{index}_imu" for index in range(1, 8)] + ["ee_site"]
        self._reset()

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Panda 资产的 keyframe 不包含运行期追加的自由物体，因此只恢复机器人部分。
        robot_qpos_count = int(self.model.jnt_qposadr[self._object_joints[self.spec.objects[0].name]])
        self.data.qpos[:robot_qpos_count] = self.model.key_qpos[0, :robot_qpos_count]
        self.data.ctrl[:] = self.model.key_ctrl[0]
        mujoco.mj_forward(self.model, self.data)
        if self.cfg.data_collector.render.viewer:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
        for _ in range(self.cfg.data_collector.simulation.settle_steps):
            mujoco.mj_step(self.model, self.data)

    @property
    def ee_position(self) -> np.ndarray:
        """返回末端 site 的世界坐标。"""
        return self.data.site_xpos[self._ee_site].copy()

    @property
    def ee_rotation(self) -> np.ndarray:
        """返回末端 site 的世界旋转矩阵。"""
        return self.data.site_xmat[self._ee_site].reshape(3, 3).copy()

    @property
    def arm_qpos(self) -> np.ndarray:
        """返回七个机械臂关节位置。"""
        return self.data.qpos[self._arm_qpos].copy()

    @property
    def arm_dofs(self) -> np.ndarray:
        """返回机械臂在广义速度向量中的索引。"""
        return self._arm_dofs.copy()

    @property
    def home_arm_qpos(self) -> np.ndarray:
        """返回固定 Panda 资产 keyframe 中的七关节 home 姿态。"""
        return self.model.key_qpos[0, self._arm_qpos].copy()

    def object_position(self, name: str) -> np.ndarray:
        """返回对象的世界坐标。"""
        return self.data.xpos[self._object_bodies[name]].copy()

    def object_rotation(self, name: str) -> np.ndarray:
        """返回对象的世界旋转矩阵。"""
        return self.data.xmat[self._object_bodies[name]].reshape(3, 3).copy()

    def object_effective_speed(self, name: str) -> float:
        """返回兼顾质心平移与表面转动的对象有效速度。"""
        dof_address = self.model.jnt_dofadr[self._object_joints[name]]
        velocity = self.data.qvel[dof_address : dof_address + 6]
        object_size = next(obj.size for obj in self.spec.objects if obj.name == name)
        return float(max(np.linalg.norm(velocity[:3]), object_size * np.linalg.norm(velocity[3:])))

    def object_target_to_ee(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """把抓持对象的目标位置转换为末端位置目标。"""
        return np.asarray(position, dtype=np.float64) - rotation @ self._held_offset

    def set_object_linear_velocity(self, name: str, velocity: np.ndarray) -> None:
        """持续覆盖自由对象的世界线速度，并清零角速度。"""
        dof_address = self.model.jnt_dofadr[self._object_joints[name]]
        value = np.asarray(velocity, dtype=np.float64).copy()
        self._object_velocity_overrides[name] = value
        self.data.qvel[dof_address : dof_address + 3] = value
        self.data.qvel[dof_address + 3 : dof_address + 6] = 0.0

    def clear_object_linear_velocity(self, name: str, *, zero_velocity: bool = True) -> None:
        """结束对象速度覆盖；可保留瞬时速度交还给自由物理运动。"""
        self._object_velocity_overrides.pop(name, None)
        if zero_velocity:
            dof_address = self.model.jnt_dofadr[self._object_joints[name]]
            self.data.qvel[dof_address : dof_address + 6] = 0.0

    def set_controls(self, joint_targets: np.ndarray, gripper_width: float) -> None:
        """设置七关节位置目标与夹爪开度。"""
        self.data.ctrl[:7] = joint_targets
        actuator = self.model.actuator("actuator8")
        joint = self.model.joint("finger_joint1")
        width_ratio = np.clip(gripper_width / joint.range[1], 0.0, 1.0)
        self.data.ctrl[7] = actuator.ctrlrange[0] + width_ratio * (actuator.ctrlrange[1] - actuator.ctrlrange[0])

    def step(self) -> None:
        """推进控制周期；抓取运动只由 MuJoCo 接触、摩擦和执行器决定。"""
        for _ in range(self.cfg.data_collector.simulation.control_substeps):
            for object_name, velocity in self._object_velocity_overrides.items():
                dof_address = self.model.jnt_dofadr[self._object_joints[object_name]]
                self.data.qvel[dof_address : dof_address + 3] = velocity
                self.data.qvel[dof_address + 3 : dof_address + 6] = 0.0
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync(state_only=True)

    def grasp_contact_forces(self, object_name: str) -> dict[str, float]:
        """返回指定对象与左右手指之间的法向接触力总和。"""
        object_body = self._object_bodies[object_name]
        forces = {"left": 0.0, "right": 0.0}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            bodies = {int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])}
            if object_body not in bodies:
                continue
            side = next((name for name, body in self._finger_bodies.items() if body in bodies), None)
            if side is None:
                continue
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            forces[side] += abs(float(contact_force[0]))
        return forces

    def object_in_contact_with_geom(self, object_name: str, geom_name: str) -> bool:
        """判断对象是否与指定场景几何体保持真实物理接触。"""
        object_body = self._object_bodies[object_name]
        target_geom = self.model.geom(geom_name).id
        return any(
            target_geom in (int(contact.geom1), int(contact.geom2))
            and object_body in (
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            )
            for contact in self.data.contact
        )

    def _grasp_relative_position(self, object_name: str) -> np.ndarray:
        return self.ee_rotation.T @ (self.object_position(object_name) - self.ee_position)

    def object_in_grasp_capture_region(self, object_name: str) -> bool:
        """检查对象中心是否位于两指可形成物理夹持的局部区域。"""
        position = self.object_position(object_name)
        settings = self.cfg.data_collector.controller
        relative_position = self._grasp_relative_position(object_name)
        if np.linalg.norm(position - self.ee_position) > settings.grasp_distance:
            return False
        if np.linalg.norm(relative_position[:2]) > settings.grasp_lateral_tolerance:
            return False
        if abs(relative_position[2] - settings.grasp_height_offset) > settings.grasp_axial_tolerance:
            return False
        return True

    def claim_physical_grasp(self, object_name: str) -> bool:
        """仅在捕获区内且双指法向力达标时登记物理抓取状态。"""
        settings = self.cfg.data_collector.controller
        forces = self.grasp_contact_forces(object_name)
        if not self.object_in_grasp_capture_region(object_name):
            return False
        if any(force < settings.grasp_min_normal_force for force in forces.values()):
            return False
        self._held_offset = self._grasp_relative_position(object_name)
        self.held_object = object_name
        return True

    def physical_grasp_is_retained(self, object_name: str) -> bool:
        """依据夹爪邻域与持续双指法向接触判断物理抓取是否保持。"""
        if self.held_object != object_name:
            return False
        distance = np.linalg.norm(self.object_position(object_name) - self.ee_position)
        settings = self.cfg.data_collector.controller
        forces = self.grasp_contact_forces(object_name)
        return bool(
            distance <= settings.grasp_hold_distance
            and all(force >= settings.grasp_min_normal_force for force in forces.values())
        )

    def release(self) -> str | None:
        """清除逻辑抓取状态；对象运动始终由物理引擎决定。"""
        released = self.held_object
        self.held_object = None
        return released

    def jacobian(self) -> np.ndarray:
        """返回末端的 6×nv 空间 Jacobian。"""
        jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self._ee_site)
        return np.vstack([jac_pos, jac_rot])

    def _sensor(self, name: str) -> np.ndarray:
        return np.asarray(self.data.sensor(name).data, dtype=np.float32).copy()

    def _robot_state(self) -> dict[str, Any]:
        base_rotation = self.data.xmat[self._base_body].reshape(3, 3)
        frames = {
            site: {
                "position_base": self._sensor(f"{site}_pos"),
                "quaternion_base_wxyz": self._sensor(f"{site}_quat"),
                "imu_acceleration_local": self._sensor(f"{site}_accel"),
                "imu_angular_velocity_local": self._sensor(f"{site}_gyro"),
                "linear_velocity_base": (base_rotation.T @ self._sensor(f"{site}_linvel")).astype(np.float32),
                "angular_velocity_base": (base_rotation.T @ self._sensor(f"{site}_angvel")).astype(np.float32),
                "linear_acceleration_base": (base_rotation.T @ self._sensor(f"{site}_linacc")).astype(np.float32),
                "angular_acceleration_base": (base_rotation.T @ self._sensor(f"{site}_angacc")).astype(np.float32),
            }
            for site in self._sensor_sites
        }
        qvel = self.data.qvel[self._arm_dofs].astype(np.float32).copy()
        qacc = self.data.qacc[self._arm_dofs].astype(np.float32).copy()
        return {
            "joint_names": [f"joint{index}" for index in range(1, 8)],
            "joint_position": self.arm_qpos.astype(np.float32),
            "joint_velocity": qvel,
            "joint_acceleration": qacc,
            "actuator_control": self.data.ctrl.copy().astype(np.float32),
            "actuator_force": self.data.actuator_force.copy().astype(np.float32),
            "gripper_width": np.asarray(self.data.qpos[self.model.jnt_qposadr[self._finger_joint_ids]], dtype=np.float32),
            "frames": frames,
        }

    def _object_state(self) -> dict[str, Any]:
        return {
            obj.name: {
                "position_world": self.data.xpos[self._object_bodies[obj.name]].copy().astype(np.float32),
                "quaternion_world_wxyz": self.data.xquat[self._object_bodies[obj.name]].copy().astype(np.float32),
                "spatial_velocity_world": self.data.cvel[self._object_bodies[obj.name]].copy().astype(np.float32),
                "shape": obj.shape,
                "color": obj.color,
                "size": obj.size,
                "mass": obj.mass,
                "friction": obj.friction,
            }
            for obj in self.spec.objects
        }

    def _contact_and_tactile(self) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        resolution = self.cfg.data_collector.sensors.tactile_resolution
        tactile = {side: np.zeros((*resolution, 3), dtype=np.float32) for side in self._finger_bodies}
        if not self.cfg.data_collector.sensors.contact_enabled:
            return [], tactile
        return compute_tactile_state(self.model, self.data, self.cfg.data_collector.sensors)

    def _camera_state(self) -> dict[str, Any]:
        if not self.cfg.data_collector.render.enabled:
            return {}
        base_position = self.data.xpos[self._base_body]
        base_rotation = self.data.xmat[self._base_body].reshape(3, 3)
        output: dict[str, Any] = {}
        for camera in self.cfg.data_collector.render.cameras:
            camera_id = self.model.camera(camera.name).id
            world_rotation = self.data.cam_xmat[camera_id].reshape(3, 3).copy()
            world_position = self.data.cam_xpos[camera_id].copy()
            transform_world = np.eye(4, dtype=np.float32)
            transform_world[:3, :3] = world_rotation
            transform_world[:3, 3] = world_position
            transform_base = np.eye(4, dtype=np.float32)
            transform_base[:3, :3] = base_rotation.T @ world_rotation
            transform_base[:3, 3] = base_rotation.T @ (world_position - base_position)
            focal_x, focal_y, principal_x, principal_y = self.model.cam_intrinsic[camera_id]
            center_x = camera.width / 2.0 + principal_x
            center_y = camera.height / 2.0 + principal_y
            intrinsics = np.asarray([
                [focal_x, 0.0, center_x],
                [0.0, focal_y, center_y],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            camera_data: dict[str, Any] = {"K": intrinsics, "T_world_camera": transform_world, "T_base_camera": transform_base}
            renderer = self._renderers.setdefault((camera.height, camera.width), mujoco.Renderer(self.model, camera.height, camera.width))
            if "rgb" in camera.modalities:
                renderer.update_scene(self.data, camera=camera.name)
                camera_data["rgb"] = renderer.render().copy()
            if "depth" in camera.modalities:
                renderer.enable_depth_rendering()
                renderer.update_scene(self.data, camera=camera.name)
                camera_data["depth"] = renderer.render().copy().astype(np.float32)
                renderer.disable_depth_rendering()
            if "segmentation" in camera.modalities:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(self.data, camera=camera.name)
                camera_data["segmentation"] = renderer.render().copy().astype(np.int32)
                renderer.disable_segmentation_rendering()
            output[camera.name] = camera_data
        return output

    def _full_physics_state(self) -> np.ndarray:
        size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return state

    def capture(self, phase: str, action: dict[str, Any], success_state: dict[str, Any] | None = None) -> None:
        """采集一个包含状态、动作、视觉和触觉的控制帧。"""
        if len(self.frames) >= self.cfg.data_collector.simulation.max_frames:
            raise RuntimeError("场景超过最大记录帧数")
        relations = {
            obj.name: "HELD_BY gripper" if obj.name == self.held_object else (
                "AT_REST ON table" if np.linalg.norm(self.data.cvel[self._object_bodies[obj.name]]) < self.cfg.data_collector.tasks.velocity_tolerance else "MOVING_ON workspace"
            )
            for obj in self.spec.objects
        }
        contacts, tactile = self._contact_and_tactile()
        self.frames.append(FrameRecord(
            frame_index=len(self.frames), simulation_time=float(self.data.time), phase=phase, action=action,
            robot=self._robot_state(), objects=self._object_state(),
            scene_description=describe_scene(self.spec.objects, relations), physics_state=self._full_physics_state(),
            cameras=self._camera_state(), contacts=contacts,
            tactile={"channel_order": ["normal", "tangent_x", "tangent_y"], "force_maps": tactile} if self.cfg.data_collector.sensors.contact_enabled else {},
            success_state=success_state or {},
        ))

    def close(self) -> None:
        """释放渲染器和可选 GUI。"""
        for renderer in self._renderers.values():
            renderer.close()
        if self._viewer is not None:
            with suppress(Exception):
                self._viewer.close()


__all__ = ["EmbodiedSimulator", "compute_tactile_state"]
