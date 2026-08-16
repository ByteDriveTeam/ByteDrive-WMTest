"""验证任务、传感器、LMDB compact 和断点续采核心契约。

模块: data/data_collector/tests/test_data_collector.py
依赖: pytest, mujoco, config, data.data_collector
读取配置: data_collector.*
对外接口: 无
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from config import load_config
from data.data_collector.controller import ScriptedExpert
from data.data_collector.records import FrameRecord, SceneRecord
from data.data_collector.scene import (
    add_virtual_tactile_sites,
    asset_fingerprint,
    build_mjcf,
    generate_scene_spec,
    materialize_mjcf,
    scene_identifier,
)
from data.data_collector.simulation import EmbodiedSimulator, compute_tactile_state
from data.data_collector.storage import DatasetStore, config_fingerprint, validate_dataset
from data.data_collector.task_language import parse_instruction


RUNTIME = Path(__file__).resolve().parent / "_runtime"


@pytest.fixture(autouse=True)
def clean_runtime():
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True)
    yield
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)


def test_scene_generation_is_deterministic_and_language_has_no_privileged_float():
    cfg = load_config()
    first = generate_scene_spec(3, 2, cfg, "SLIDE_REGRASP")
    second = generate_scene_spec(3, 2, cfg, "SLIDE_REGRASP")
    assert first == second
    assert parse_instruction(first.task.instruction)
    assert not any(character == "." and index + 1 < len(first.task.instruction) and first.task.instruction[index + 1].isdigit() for index, character in enumerate(first.task.instruction))


def test_object_count_is_reproducibly_sampled_in_configured_interval():
    cfg = load_config()
    counts = [len(generate_scene_spec(index, 0, cfg, "PICK_PLACE").objects) for index in range(12)]
    repeated = [len(generate_scene_spec(index, 0, cfg, "PICK_PLACE").objects) for index in range(12)]
    assert counts == repeated
    assert set(counts) == set(range(cfg.data_collector.scene.object_count_min, cfg.data_collector.scene.object_count_max + 1))

    for task_type in ("SORT", "STACK", "SEQUENTIAL_REARRANGE"):
        assert len(generate_scene_spec(2, 0, cfg, task_type).objects) >= 2

    fixed_scene = replace(cfg.data_collector.scene, object_count_min=3, object_count_max=3)
    fixed_cfg = replace(cfg, data_collector=replace(cfg.data_collector, scene=fixed_scene))
    sort_spec = generate_scene_spec(0, 0, fixed_cfg, "SORT")
    pick_place_spec = generate_scene_spec(0, 0, fixed_cfg, "PICK_PLACE")
    assert len(sort_spec.objects) == 3
    assert sum(step.verb == "PICK" for step in sort_spec.task.steps) == 3
    assert len(pick_place_spec.objects) == 3
    assert sum(step.verb == "PICK" for step in pick_place_spec.task.steps) == 1

    task_types = ("PICK_PLACE", "SORT", "SLIDE_REGRASP", "STACK", "SEQUENTIAL_REARRANGE", "ORIENT_AND_PLACE")
    for scene_index, task_type in enumerate(task_types):
        spec = generate_scene_spec(scene_index, 0, cfg, task_type)
        assert len({obj.color for obj in spec.objects}) == len(spec.objects)
        if task_type == "SLIDE_REGRASP":
            assert spec.objects[0].shape in cfg.data_collector.scene.slide_target_shapes


def test_tactile_force_map_has_required_resolution_and_components():
    cfg = load_config()
    sensors = replace(cfg.data_collector.sensors, contact_enabled=True)
    render = replace(cfg.data_collector.render, enabled=False)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, sensors=sensors, render=render))
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    simulator = EmbodiedSimulator(spec, build_mjcf(spec, cfg), cfg)
    try:
        simulator.capture("TEST", {})
        tactile = simulator.frames[0].tactile
        assert tactile["channel_order"] == ["normal", "tangent_x", "tangent_y"]
        assert tactile["force_maps"]["left"].shape == (32, 32, 3)
        assert tactile["force_maps"]["right"].shape == (32, 32, 3)
    finally:
        simulator.close()


def test_successful_grasp_records_nonzero_fingertip_contact_force():
    cfg = load_config()
    sensors = replace(cfg.data_collector.sensors, contact_enabled=True)
    render = replace(cfg.data_collector.render, enabled=False)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, sensors=sensors, render=render))
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    simulator = EmbodiedSimulator(spec, build_mjcf(spec, cfg), cfg)
    try:
        evidence = ScriptedExpert(simulator, spec, cfg).run()
        assert evidence is not None
        contact_frames = [frame for frame in simulator.frames if frame.contacts]
        assert contact_frames
        maps = [frame.tactile["force_maps"][side][..., 0] for frame in contact_frames for side in ("left", "right")]
        strongest = max(maps, key=lambda value: float(np.max(value)))
        assert float(np.max(strongest)) > 0.0
        peak_y, peak_x = np.unravel_index(int(np.argmax(strongest)), strongest.shape)
        assert 1 <= peak_x < 31 and 1 <= peak_y < 31
    finally:
        simulator.close()


def test_sensor_sites_are_invisible_and_target_marker_is_a_ground_decal():
    cfg = load_config()
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    root = ET.fromstring(build_mjcf(spec, cfg))
    ee_site = root.find(".//site[@name='ee_site']")
    target = root.find(".//geom[@name='target_center_zone']")
    assert ee_site is not None and ee_site.attrib["rgba"] == "0 0 0 0"
    assert target is not None and target.attrib["contype"] == "0" and target.attrib["conaffinity"] == "0"
    target_z = float(target.attrib["pos"].split()[2])
    assert target_z == pytest.approx(cfg.data_collector.scene.table_height + cfg.data_collector.scene.target_size[2])


def test_overview_camera_uses_manual_ypr_and_xy_fov_without_forced_target():
    cfg = load_config()
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    root = ET.fromstring(build_mjcf(spec, cfg))
    camera_cfg = next(camera for camera in cfg.data_collector.render.cameras if camera.name == "overview")
    camera = root.find(".//camera[@name='overview']")
    assert camera is not None
    assert "mode" not in camera.attrib and "target" not in camera.attrib
    assert not root.findall(".//camera[@quat]")
    assert root.find(".//body[@name='overview_target']") is None
    np.testing.assert_allclose([float(value) for value in camera.attrib["pos"].split()], camera_cfg.position)
    np.testing.assert_allclose(
        [float(value) for value in camera.attrib["euler"].split()],
        np.deg2rad([camera_cfg.roll, camera_cfg.pitch, camera_cfg.yaw]),
    )
    expected_focal = [
        camera_cfg.width / (2.0 * np.tan(np.deg2rad(camera_cfg.fov_x) / 2.0)),
        camera_cfg.height / (2.0 * np.tan(np.deg2rad(camera_cfg.fov_y) / 2.0)),
    ]
    assert "fovy" not in camera.attrib
    np.testing.assert_allclose([float(value) for value in camera.attrib["focalpixel"].split()], expected_focal)
    np.testing.assert_allclose([float(value) for value in camera.attrib["principalpixel"].split()], [0.0, 0.0])


def test_camera_principal_point_and_saved_intrinsics_are_image_centered():
    cfg = load_config()
    render = replace(cfg.data_collector.render, enabled=True)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, render=render))
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    simulator = EmbodiedSimulator(spec, build_mjcf(spec, cfg), cfg)
    try:
        simulator.capture("TEST_CAMERA_INTRINSICS", {})
        for camera in cfg.data_collector.render.cameras:
            intrinsics = simulator.frames[0].cameras[camera.name]["K"]
            np.testing.assert_allclose(intrinsics[:2, 2], [camera.width / 2.0, camera.height / 2.0])
    finally:
        simulator.close()


def test_fullphysics_replay_recomputes_tactile_for_dataset_without_saved_sites():
    cfg = load_config()
    sensors = replace(cfg.data_collector.sensors, contact_enabled=True)
    render = replace(cfg.data_collector.render, enabled=False)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, sensors=sensors, render=render))
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    simulator = EmbodiedSimulator(spec, build_mjcf(spec, cfg), cfg)
    try:
        assert ScriptedExpert(simulator, spec, cfg).run() is not None
        contact_frame = max(
            (frame for frame in simulator.frames if frame.contacts),
            key=lambda frame: max(
                float(np.max(frame.tactile["force_maps"][side][..., 0])) for side in ("left", "right")
            ),
        )
        disabled = replace(cfg.data_collector.sensors, contact_enabled=False)
        old_cfg = replace(cfg, data_collector=replace(cfg.data_collector, sensors=disabled))
        old_xml = build_mjcf(spec, old_cfg)
        assert "left_tactile_site" not in old_xml
        replay_model = mujoco.MjModel.from_xml_string(materialize_mjcf(add_virtual_tactile_sites(old_xml)))
        replay_data = mujoco.MjData(replay_model)
        assert mujoco.mj_stateSize(replay_model, mujoco.mjtState.mjSTATE_FULLPHYSICS) == len(contact_frame.physics_state)
        mujoco.mj_setState(
            replay_model, replay_data, contact_frame.physics_state, mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        mujoco.mj_forward(replay_model, replay_data)

        contacts, force_maps = compute_tactile_state(replay_model, replay_data, sensors)

        assert contacts
        assert max(float(np.max(force_maps[side][..., 0])) for side in ("left", "right")) > 0.0
    finally:
        simulator.close()


def test_physical_grasp_requires_bilateral_contact_and_never_snaps_object_back():
    cfg = load_config()
    sensors = replace(cfg.data_collector.sensors, contact_enabled=True)
    render = replace(cfg.data_collector.render, enabled=False)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, sensors=sensors, render=render))
    spec = generate_scene_spec(0, 0, cfg, "PICK_PLACE")
    simulator = EmbodiedSimulator(spec, build_mjcf(spec, cfg), cfg)
    object_name = spec.objects[0].name
    try:
        settings = cfg.data_collector.controller
        initial_height = float(simulator.object_position(object_name)[2])

        # 物体即使位于几何捕获区，没有左右指接触力也不能登记为抓住。
        joint_id = simulator._object_joints[object_name]
        qpos_address = simulator.model.jnt_qposadr[joint_id]
        centered = np.asarray([0.0, 0.0, settings.grasp_height_offset])
        simulator.data.qpos[qpos_address : qpos_address + 3] = simulator.ee_position + simulator.ee_rotation @ centered
        simulator.data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(simulator.model, simulator.data)
        assert not simulator.claim_physical_grasp(object_name)

        # 重置后由专家通过真实闭指接触和实际抬升建立物理抓取。
        simulator._reset()
        initial_height = float(simulator.object_position(object_name)[2])
        expert = ScriptedExpert(simulator, spec, cfg)
        assert expert._pick(spec.task.steps[0])
        forces = simulator.grasp_contact_forces(object_name)
        assert all(force >= settings.grasp_min_normal_force for force in forces.values())
        assert simulator.object_position(object_name)[2] - initial_height >= settings.grasp_lift_min_height
        assert any(frame.phase.startswith("VERIFY_PHYSICAL_GRASP") for frame in simulator.frames)

        # 人为把自由物体移出夹爪后，下一物理步不得像绑定约束那样把它拉回末端。
        joint_id = simulator._object_joints[object_name]
        qpos_address = simulator.model.jnt_qposadr[joint_id]
        simulator.data.qpos[qpos_address : qpos_address + 3] = simulator.ee_position + np.asarray([0.20, 0.0, 0.0])
        simulator.data.qvel[simulator.model.jnt_dofadr[joint_id] : simulator.model.jnt_dofadr[joint_id] + 6] = 0.0
        mujoco.mj_forward(simulator.model, simulator.data)
        simulator.step()
        assert np.linalg.norm(simulator.object_position(object_name) - simulator.ee_position) > 0.10
        assert not simulator.physical_grasp_is_retained(object_name)
    finally:
        simulator.close()


def _successful_record(cfg, scene_index):
    spec = generate_scene_spec(scene_index, 0, cfg, "PICK_PLACE")
    config_hash = config_fingerprint(cfg)
    frame = FrameRecord(
        frame_index=0, simulation_time=0.0, phase="TEST", action={}, robot={}, objects={},
        scene_description="SCENE HAS object_0 ON table.", physics_state=np.zeros(4, dtype=np.float64),
        success_state={"success": True},
    )
    return SceneRecord(
        scene_id=scene_identifier(spec, config_hash), spec=spec, mjcf_xml=build_mjcf(spec, cfg), frames=[frame],
        success_evidence={"success": True}, asset_hash=asset_fingerprint(), config_hash=config_hash,
        config_snapshot=asdict(cfg), versions={"test": "true"},
    )


def test_one_scene_one_compacted_lmdb_and_resume_from_success_count():
    cfg = load_config()
    root = RUNTIME / "dataset"
    store = DatasetStore(root, cfg, asset_fingerprint())
    store.initialize()
    first_path = store.publish(_successful_record(cfg, 0))
    assert first_path.is_dir()
    assert (first_path / "data.mdb").stat().st_size < cfg.data_collector.storage.map_size_mb * 1024 * 1024
    assert len(store.scan_completed()) == 1
    store.publish(_successful_record(cfg, 1))
    report = validate_dataset(root, cfg, deep=True)
    assert report["scene_count"] == 2
    assert len(store.scan_completed()) == 2


def test_checkpoint_atomic_replace_retries_transient_windows_lock(monkeypatch):
    cfg = load_config()
    storage_cfg = replace(cfg.data_collector.storage, atomic_replace_attempts=4, atomic_replace_retry_seconds=0.0)
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, storage=storage_cfg))
    root = RUNTIME / "checkpoint_retry_dataset"
    store = DatasetStore(root, cfg, asset_fingerprint())
    store.initialize()
    store._repair_checkpoint(0)

    import data.data_collector.storage.storage as storage_module

    replace_calls = 0
    real_replace = storage_module.os.replace

    def flaky_replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls < 3:
            raise PermissionError(5, "模拟 Windows 临时文件占用")
        real_replace(source, destination)

    monkeypatch.setattr(storage_module.os, "replace", flaky_replace)
    store._repair_checkpoint(1)
    checkpoint = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    assert replace_calls == 3
    assert checkpoint["completed_scene_count"] == 1
    assert not list(root.glob(".checkpoint.json.*.tmp"))


def test_atomic_replace_retry_policy_does_not_change_dataset_fingerprint():
    cfg = load_config()
    changed_storage = replace(
        cfg.data_collector.storage,
        atomic_replace_attempts=cfg.data_collector.storage.atomic_replace_attempts + 1,
        atomic_replace_retry_seconds=cfg.data_collector.storage.atomic_replace_retry_seconds + 0.1,
    )
    changed = replace(cfg, data_collector=replace(cfg.data_collector, storage=changed_storage))
    assert config_fingerprint(changed) == config_fingerprint(cfg)


def test_failed_scene_is_rejected_without_creating_lmdb():
    cfg = load_config()
    root = RUNTIME / "failure_dataset"
    store = DatasetStore(root, cfg, asset_fingerprint())
    store.initialize()
    record = _successful_record(cfg, 0)
    record.success_evidence = {"success": False}
    with pytest.raises(ValueError):
        store.publish(record)
    assert not list(root.glob("scene_*.lmdb"))


@pytest.mark.parametrize("task_type", ["PICK_PLACE", "SORT", "SLIDE_REGRASP", "STACK", "SEQUENTIAL_REARRANGE", "ORIENT_AND_PLACE"])
def test_scripted_expert_can_produce_success(task_type):
    cfg = load_config()
    cfg = replace(cfg, data_collector=replace(cfg.data_collector, render=replace(cfg.data_collector.render, enabled=False)))
    success = False
    for attempt in range(3):
        spec = generate_scene_spec(5, attempt, cfg, task_type)
        simulator = EmbodiedSimulator(spec, build_mjcf(spec, cfg), cfg)
        try:
            success = ScriptedExpert(simulator, spec, cfg).run() is not None
        finally:
            simulator.close()
        if success:
            break
    assert success
