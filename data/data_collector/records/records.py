"""定义采集系统跨模块使用的不可变记录类型。

模块: data/data_collector/records/records.py
依赖: dataclasses, typing
读取配置: 无
对外接口:
    - ObjectSpec
    - ActionStep
    - TaskSpec
    - SceneSpec
    - FrameRecord
    - SceneRecord
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObjectSpec:
    """描述一个可动物体的公开属性和初始物理参数。"""

    name: str
    shape: str
    color: str
    size: float
    mass: float
    friction: float
    initial_position: list[float]
    initial_quaternion: list[float]


@dataclass(frozen=True)
class ActionStep:
    """描述一个受控语言动作步骤。"""

    verb: str
    object_ref: str = ""
    target_ref: str = ""
    relation: str = ""
    event: str = ""
    qualifier: str = ""


@dataclass(frozen=True)
class TaskSpec:
    """保存任务类型、动作 AST 与规范指令。"""

    task_type: str
    steps: list[ActionStep]
    instruction: str

    def to_dict(self) -> dict[str, Any]:
        """返回可序列化的任务结构。"""
        return asdict(self)


@dataclass(frozen=True)
class SceneSpec:
    """保存随机场景的静态定义。"""

    scene_index: int
    seed: int
    task: TaskSpec
    objects: list[ObjectSpec]
    slope_angle: float
    target_positions: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        """返回可序列化的场景结构。"""
        return asdict(self)


@dataclass
class FrameRecord:
    """保存一个控制帧的完整观测、动作和复现状态。"""

    frame_index: int
    simulation_time: float
    phase: str
    action: dict[str, Any]
    robot: dict[str, Any]
    objects: dict[str, Any]
    scene_description: str
    physics_state: Any
    cameras: dict[str, Any] = field(default_factory=dict)
    contacts: list[dict[str, Any]] = field(default_factory=list)
    tactile: dict[str, Any] = field(default_factory=dict)
    success_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """返回保留 NumPy 数组的序列化映射。"""
        return dict(self.__dict__)


@dataclass
class SceneRecord:
    """保存一条已判定成功的场景轨迹。"""

    scene_id: str
    spec: SceneSpec
    mjcf_xml: str
    frames: list[FrameRecord]
    success_evidence: dict[str, Any]
    asset_hash: str
    config_hash: str
    config_snapshot: dict[str, Any]
    versions: dict[str, str]


__all__ = ["ActionStep", "FrameRecord", "ObjectSpec", "SceneRecord", "SceneSpec", "TaskSpec"]
