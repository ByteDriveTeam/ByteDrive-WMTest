# ByteDrive 文档与源文件索引

## 配置

- `config/default.yaml` — 数据采集与可视化全部可调参数的唯一默认值来源，并逐项说明单位和语义。
- `config/schema.py` — 定义并校验项目集中配置。
- `config/__init__.py` — 加载集中配置并返回不可变配置对象。

## 数据采集

- `data/__init__.py` — 提供项目数据处理与采集包。
- `data/data_collector/__init__.py` — 重导出单臂具身数据采集系统公共 API。
- `data/data_collector/run.py` — 提供采集、续采、检查、验证、compact 和二次渲染 CLI。
- `data/data_collector/README.md` — 说明采集器安装、采集、续采、维护及视觉触觉配置。
- `data/data_collector/records/records.py` — 定义采集系统跨模块使用的不可变记录类型。
- `data/data_collector/records/__init__.py` — 重导出采集记录类型。
- `data/data_collector/task_language/task_language.py` — 生成和解析无特权坐标的受控任务语言。
- `data/data_collector/task_language/__init__.py` — 重导出受控任务语言接口。
- `data/data_collector/task_language/checks/task_language_checks.py` — 校验受控指令词表并阻止连续坐标泄漏。
- `data/data_collector/tasks/tasks.py` — 按配置生成确定性的多类型具身任务。
- `data/data_collector/tasks/__init__.py` — 重导出任务生成接口。
- `data/data_collector/tasks/checks/tasks_checks.py` — 校验任务类型与对象数量。
- `data/data_collector/scene/scene.py` — 生成确定性任务场景并构造包含 Panda 的 MJCF。
- `data/data_collector/scene/__init__.py` — 重导出随机场景与 MJCF 构建接口。
- `data/data_collector/scene/checks/scene_checks.py` — 校验场景对象命名与初始布局间距。
- `data/data_collector/simulation/simulation.py` — 封装 MuJoCo 步进、状态读取、多相机渲染和 32×32 三轴触觉力图。
- `data/data_collector/simulation/__init__.py` — 重导出 MuJoCo 仿真接口。
- `data/data_collector/simulation/checks/simulation_checks.py` — 校验仿真场景与 MJCF 输入。
- `data/data_collector/controller/controller.py` — 使用 Jacobian IK 和夹爪状态机执行受控任务 AST。
- `data/data_collector/controller/__init__.py` — 重导出脚本专家控制接口。
- `data/data_collector/controller/checks/controller_checks.py` — 校验控制模型对象与任务动作支持范围。
- `data/data_collector/storage/storage.py` — 实现成功场景独立 LMDB、内容校验、compact 发布和可修复 checkpoint。
- `data/data_collector/storage/__init__.py` — 重导出成功场景 LMDB 和断点续采接口。
- `data/data_collector/storage/checks/storage_checks.py` — 校验项目内路径、成功证据和连续帧序号。
- `data/data_collector/collection/collection.py` — 编排成功场景生成、脚本执行、LMDB 发布和断点续采。
- `data/data_collector/collection/__init__.py` — 重导出采集与断点续采接口。
- `data/data_collector/collection/checks/collection_checks.py` — 校验采集目标数与显式任务列表。
- `data/data_collector/replay/replay.py` — 从单场景 LMDB 恢复完整物理状态并进行二次渲染。
- `data/data_collector/replay/__init__.py` — 重导出二次渲染接口。
- `data/data_collector/replay/checks/replay_checks.py` — 校验二次渲染的数据集、输出和相机。
- `data/data_collector/tests/test_data_collector.py` — 验证任务、传感器、LMDB compact 和断点续采核心契约。
- `data/data_collector/tests/__init__.py` — 包含数据采集系统的自动化回归测试。

## 文档

- `Doc/开发规范.md` — ByteDrive 强制开发规范。
- `Doc/数据采集格式.md` — 定义单场景 LMDB、逐帧状态、视觉和触觉字段。
- `Doc/Index.md` — ByteDrive 文档与源文件的单一导航入口。

## 数据可视化

- `vis/__init__.py` — 提供项目可视化工具包。
- `vis/data_vis/data_vis.py` — 优先读取 LMDB 图像，并在图像缺失时恢复物理状态重放可视化。
- `vis/data_vis/__init__.py` — 重导出 LMDB 场景可视化接口。
- `vis/data_vis/run.py` — 提供成功场景 LMDB 可视化命令行入口。
- `vis/data_vis/README.md` — 说明可视化命令、自动读图/重放策略与输出格式。
- `vis/data_vis/checks/data_vis_checks.py` — 校验可视化输入、模态和项目内输出边界。
- `vis/data_vis/checks/__init__.py` — 重导出数据可视化输入校验。
- `vis/data_vis/tests/test_data_vis.py` — 验证优先读图和缺图重放两条路径。
- `vis/data_vis/tests/__init__.py` — 包含 LMDB 场景可视化自动化测试。

## 第三方资产

- `data/data_collector/assets/franka_emika_panda/` — 固定提交的 MuJoCo Menagerie Franka Panda 模型、网格与许可证。
