# MuJoCo 单臂具身数据采集器

本程序使用固定版本的 Franka Panda、MuJoCo 动力学和确定性脚本专家生成成功操作轨迹。一个成功场景对应一个只读 LMDB；失败候选只存在于运行内存，不会生成最终数据库。

## 安装

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

默认配置位于 `config/default.yaml`。实验参数只允许在配置文件中定义；CLI 参数是显式覆盖，不在代码内声明第二份默认值。

## 采集与续采

```powershell
# 采集到总计 100 个成功场景；已有 30 个时仅新增 70 个
.\.venv\Scripts\python.exe -m data.data_collector.run collect --scenes 100 --resume

# 固定任务并开启多相机图像与双指触觉
.\.venv\Scripts\python.exe -m data.data_collector.run collect --scenes 20 --task SLIDE_REGRASP --render --contact-sensors --resume
```

`--scenes` 表示期望达到的成功场景总数。续采时程序扫描 `scene_000000_*.lmdb` 开始的连续有效前缀；最终 LMDB 是计数事实来源，`checkpoint.json` 可根据磁盘内容自动修复。

## 数据维护

```powershell
.\.venv\Scripts\python.exe -m data.data_collector.run inspect --dataset data/data_collector/output/dataset
.\.venv\Scripts\python.exe -m data.data_collector.run validate --dataset data/data_collector/output/dataset --deep
.\.venv\Scripts\python.exe -m data.data_collector.run compact --dataset data/data_collector/output/dataset
.\.venv\Scripts\python.exe -m data.data_collector.run rerender --dataset data/data_collector/output/dataset --scene 0 --frames 0-10 --camera overview --output data/data_collector/output/rerender
```

场景先写入 `staging/*.partial`，校验成功后通过 LMDB compact 副本发布，避免把预分配映射中的空页面带入最终数据。最终场景发布后保持不可变。

## 视觉与触觉

- 相机数量、父坐标系、分辨率、内外参及 RGB/depth/segmentation 模态均可逐台配置。
- 不采图时仍保存 MJCF、相机定义和每帧 `FULLPHYSICS` 状态，可在以后补渲染。
- 开启触觉后，每帧保存左右指尖各一张 `(32, 32, 3)` float32 力图。
- 触觉通道顺序固定为 `[normal, tangent_x, tangent_y]`；法向力非负，两个切向力保留符号。
- 原始接触列表同时保存接触双方、世界位置、法线和接触坐标系中的法向/双切向力。

完整字段定义见 `Doc/数据采集格式.md`。

## 物理抓取

Panda 的 `ee_site` 与两块指尖接触面的中心对齐。抓取没有 weld、mocap 跟随、位姿覆盖或其他绑定：物体运动始终由 MuJoCo 接触、指尖执行器、摩擦和重力决定。系统仅在物体进入夹爪捕获区、左右指法向力连续达到阈值且物体随后真实抬升时登记 `HELD_BY`；任一条件不成立的候选场景直接失败且不落盘。运输中的逻辑状态只用于任务编排和掉落检测，不向物体施加力或修改位姿；物体离开夹爪邻域或双指法向接触连续丢失后会判定掉落。放置时先物理张开夹爪，再清除逻辑状态。

斜面任务会等待物体实际越过斜面边缘、落台并连续稳定，然后执行同样的双指接触与实际抬升验证。触觉图直接来自当前 MuJoCo 接触，因此某一指显示 `0 N` 表示该帧该指确实没有接触；这种单侧接触不会通过抓取成功验证。
