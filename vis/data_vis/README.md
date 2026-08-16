# LMDB 场景可视化

该工具读取一个已发布的成功场景 LMDB，并生成带关节状态和双指触觉法向力面板的 PNG 序列。触觉热图使用固定牛顿量程和可配置伽马增强低力接触，面板同时显示每根手指的 N/Tx/Ty 峰值。若所选帧已经保存了指定相机与模态，直接读取原始数组；否则使用场景内的 MJCF 和 `FULLPHYSICS` 状态自动重放。触觉也遵循相同策略：优先读取原始 `32×32×3` 力图，缺失时使用场景内的完整传感器配置快照恢复接触并重新计算。

```powershell
.venv\Scripts\python.exe -m vis.data_vis.run `
  --dataset data/data_collector/output/dataset `
  --scene 0 `
  --camera overview `
  --modality rgb
```

可选模态为 `rgb`、`depth`、`segmentation`。使用 `--force-replay` 可忽略已存图像并验证二次渲染；使用 `--force-tactile-replay` 可忽略已存触觉并强制重新求解 N/Tx/Ty，`--no-force-tactile-replay` 则仅在触觉缺失时重算；使用 `--no-gif` 只输出 PNG。默认帧范围、采样步长、GIF 帧率、面板尺寸和色彩均在根目录 `config/default.yaml` 的 `data_vis` 节中配置。

每次输出位于 `<output>/<scene>_<camera>_<modality>/`，包含：

- `frames/frame_XXXXXXXX.png`：逐帧仪表板；
- `animation.gif`：可选动画；
- `summary.json`：帧索引、相机 `stored`/`replayed` 来源计数及触觉 `stored`/`recomputed` 来源计数。

重算过程不会修改 LMDB。面板中的 `Tactile: stored` 表示采集时记录的权威数据，`Tactile: recomputed` 表示按场景快照派生的数据；若运行环境的 MuJoCo 版本或资产与数据集不一致，派生接触力可能与采集时存在数值差异。
