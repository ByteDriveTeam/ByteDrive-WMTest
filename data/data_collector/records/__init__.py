"""重导出采集记录类型。

模块: data/data_collector/records/__init__.py
依赖: data.data_collector.records.records
读取配置: 无
对外接口:
    - ObjectSpec
    - ActionStep
    - TaskSpec
    - SceneSpec
    - FrameRecord
    - SceneRecord
"""

from data.data_collector.records.records import ActionStep, FrameRecord, ObjectSpec, SceneRecord, SceneSpec, TaskSpec

__all__ = ["ActionStep", "FrameRecord", "ObjectSpec", "SceneRecord", "SceneSpec", "TaskSpec"]

