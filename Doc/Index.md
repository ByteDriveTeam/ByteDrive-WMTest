# ByteDrive 文档与源文件索引

## 配置

- `config/default.yaml` — 数据采集、模型、训练与可视化全部可调参数的唯一默认值来源，并逐项说明单位和语义。
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
- `data/data_collector/scene/scene.py` — 生成确定性任务场景、构造包含 Panda 的 MJCF 并为旧场景补充虚拟触觉面。
- `data/data_collector/scene/__init__.py` — 重导出随机场景与 MJCF 构建接口。
- `data/data_collector/scene/checks/scene_checks.py` — 校验场景对象命名与初始布局间距。
- `data/data_collector/simulation/simulation.py` — 封装 MuJoCo 步进、状态读取、多相机渲染及可复用的 32×32 三轴触觉计算。
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

## 模型数据管线

- `data/model_dataset/model_dataset.py` — 从单场景 LMDB 采样固定多频率窗口并在线重放视觉与触觉。
- `data/model_dataset/__init__.py` — 重导出模型 LMDB 窗口、在线重放和归一化接口。
- `data/model_dataset/checks/model_dataset_checks.py` — 校验数据集输入及统计文件输出边界。
- `data/model_dataset/checks/__init__.py` — 重导出模型数据管线校验接口。
- `data/model_dataset/tests/test_model_dataset.py` — 验证触觉摘要、失败门控、语言定长与75%掩码。
- `data/model_dataset/tests/__init__.py` — 包含模型数据窗口与监督转换回归测试。

## 模型

- `model/__init__.py` — 提供 ByteDrive 多模态策略模型公开接口。
- `model/position/position.py` — 构造多模态时间、PETR 几何与逐层 Q/K 位置条件。
- `model/position/__init__.py` — 重导出共享 Q/K 位置编码公开接口。
- `model/position/checks/position_checks.py` — 校验位置编码输入与相机几何张量。
- `model/position/checks/__init__.py` — 重导出位置编码校验接口。
- `model/transformer/transformer.py` — 定义 BF16 Pre-Norm Transformer、模态 LoRA 与独立稠密残差流。
- `model/transformer/__init__.py` — 重导出 Transformer 内核与稠密残差公开接口。
- `model/transformer/checks/transformer_checks.py` — 校验 Transformer 与稠密残差输入。
- `model/transformer/checks/__init__.py` — 重导出 Transformer 校验接口。
- `model/policy/policy.py` — 组装多模态骨干、掩码 Predictor 与23维结构化流匹配策略。
- `model/policy/__init__.py` — 重导出 ByteDrive 策略模型与批次类型。
- `model/policy/checks/policy_checks.py` — 校验策略批次形状与教师强制输入。
- `model/policy/checks/__init__.py` — 重导出策略校验接口。
- `model/policy/tests/test_policy.py` — 验证两个无位置RegisterToken、23维流与Predictor边界。
- `model/policy/tests/__init__.py` — 包含策略结构、位置与流积分回归测试。

## 训练

- `train/__init__.py` — 提供 ByteDrive 训练、评估与损失公开接口。
- `train/run.py` — 提供归一化统计、训练与检查点评估 CLI。
- `train/objectives/objectives.py` — 计算逐层速度、最终积分、感知重建和阶段分类损失。
- `train/objectives/__init__.py` — 重导出 ByteDrive 多目标损失与epoch调度。
- `train/objectives/checks/objectives_checks.py` — 校验多目标损失输入形状。
- `train/objectives/checks/__init__.py` — 重导出多目标损失校验接口。
- `train/engine/engine.py` — 编排单GPU epoch训练、EMA更新、评估与可恢复检查点。
- `train/engine/__init__.py` — 重导出单GPU训练、EMA、评估和检查点接口。
- `train/engine/checks/engine_checks.py` — 校验训练环境与项目内输出边界。
- `train/engine/checks/__init__.py` — 重导出训练引擎校验接口。
- `train/engine/tests/test_training.py` — 验证失败行为屏蔽、感知重建保留和epoch调度。
- `train/engine/tests/__init__.py` — 包含训练损失调度与EMA回归测试。

## 文档

- `Doc/开发规范.md` — ByteDrive 强制开发规范。
- `Doc/数据采集格式.md` — 定义单场景 LMDB、逐帧状态、视觉和触觉字段。
- `Doc/Index.md` — ByteDrive 文档与源文件的单一导航入口。
- `Modeldesign.md` — 记录 ByteDrive 模型、数据监督与训练系统设计依据。

## 数据可视化

- `vis/__init__.py` — 提供项目可视化工具包。
- `vis/data_vis/data_vis.py` — 优先读取 LMDB 图像与触觉，并在缺失或强制时恢复物理状态重放和重算。
- `vis/data_vis/__init__.py` — 重导出 LMDB 场景可视化接口。
- `vis/data_vis/run.py` — 提供成功场景 LMDB 可视化命令行入口。
- `vis/data_vis/README.md` — 说明可视化命令、自动读图/重放策略与输出格式。
- `vis/data_vis/checks/data_vis_checks.py` — 校验可视化输入、模态和项目内输出边界。
- `vis/data_vis/checks/__init__.py` — 重导出数据可视化输入校验。
- `vis/data_vis/tests/test_data_vis.py` — 验证优先读图和缺图重放两条路径。
- `vis/data_vis/tests/__init__.py` — 包含 LMDB 场景可视化自动化测试。

## 第三方资产

- `data/data_collector/assets/franka_emika_panda/` — 固定提交的 MuJoCo Menagerie Franka Panda 模型、网格与许可证。
