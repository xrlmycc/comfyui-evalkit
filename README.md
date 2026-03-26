# ComfyUI EvalKit

ComfyUI 文生图评测节点包，适合做单图打分、批量路径排序、排名预览与导出。

## 安装

将本目录放入 `ComfyUI/custom_nodes` 后，安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果使用 `pyiqa` 相关指标，建议在当前 ComfyUI 虚拟环境内安装依赖。

## 主要节点

### 基础评分

- `EvalKit Metric Score`
  - 对输入 `IMAGE` 批量打单一指标分数
- `EvalKit Metric Rank`
  - 对同分辨率 `IMAGE batch` 做单一指标排序
- `EvalKit Preset Rank`
  - 对同分辨率 `IMAGE batch` 做质量 / 美学 / 对齐综合排序

### 路径排序

- `EvalKit Metric Rank From Path`
  - 直接从文件夹逐张加载图片并按单指标排序
- `EvalKit Preset Rank From Path`
  - 直接从文件夹逐张加载图片并按综合指标排序

这两个路径节点支持：

- 自动读取图片目录
- 自动尝试加载同名 `.txt`
- 如果没有提供 `prompt_folder_path`，会继续尝试读取图片元数据中的 prompt
- 输出 `report` 供后续预览和导出节点使用

### 排名结果处理

- `EvalKit Ranking Preview`
  - 预览 top / bottom N
- `EvalKit Ranking Export`
  - 导出 top / bottom N 图片，可附带导出同名 txt
- `EvalKit Score Summary`
  - 对多个外部分数做加权汇总

## compare_mode

`EvalKit Metric Rank From Path` 和 `EvalKit Preset Rank From Path` 支持 `compare_mode`：

- `original`
  - 保持原图尺寸和比例
  - 不拉伸，不补黑边
  - 适合真实成图筛选
- `pad`
  - 保持比例缩放后补黑边到目标尺寸
  - 适合希望减少尺寸差异但不想拉伸的情况
- `resize`
  - 直接缩放到目标尺寸
  - 适合需要标准化对比的情况

建议：

- 日常挑图：优先用 `original`
- 做模型实验横向对比：优先用 `pad` 或 `resize`

## prompt 读取规则

路径节点读取 prompt 的优先级：

1. `prompt_folder_path` 下同名 txt
2. 图片目录下同名 txt
3. 图片元数据中的 prompt / parameters / comment

如果缺少 prompt：

- 对齐类单指标节点会直接提示无法执行
- 综合排序节点会提醒你关闭对齐类指标，并在必要时自动跳过对齐类指标

## 推荐工作流

### 保持原图比例排序

`EvalKit Preset Rank From Path`

- `compare_mode = original`
- 获取 `report`

再接：

- `EvalKit Ranking Preview`
- `EvalKit Ranking Export`

### 标准化对比

方式一：

- `EvalKit Preset Rank From Path`
- `compare_mode = pad` 或 `resize`

方式二：

- `EvalKit Batch Load From Path`
- `EvalKit Preset Rank`

## report 的用途

路径排序节点输出的 `report` 是 JSON 字符串，包含：

- 文件名
- 图片路径
- prompt 来源
- 各项分数
- 排名
- warning

可以直接接到：

- `EvalKit Ranking Preview`
- `EvalKit Ranking Export`

## 节点职责说明

当前没有删除旧节点，原因是它们仍然有不同职责：

- `Batch Load From Path`：适合需要输出标准 `IMAGE batch` 的工作流
- `Rank From Path`：适合不同分辨率图片直接逐张评分排序
- `Ranking Preview / Export`：适合处理路径排名结果

## 注意事项

- 不同分辨率图片直接比较时，分数可能会受到尺寸差异影响
- `original` 模式更适合筛图，不一定适合严格 benchmark
- `clipscore` 等对齐类指标依赖 prompt，缺少 prompt 时结果没有意义
