# ComfyUI EvalKit

ComfyUI 文生图评测节点包，用于单指标打分、批量排序、综合排名、结果预览与导出。

## 这个项目能做什么

- 对 `IMAGE` 或 `IMAGE batch` 计算单一评测指标分数
- 对同尺寸 batch 做单指标或多指标综合排序
- 直接从文件夹读取图片并逐张评分，不要求先整理成同尺寸 batch
- 自动读取同名 `.txt` 或图片元数据中的 prompt
- 生成 `report`，供预览、导出、二次处理节点复用

## 支持的评测指标 / 模型

本项目通过 `pyiqa` 调用评测模型，当前内置这些指标：

- 质量类：`hyperiqa`、`dbcnn`、`qualiclip+`、`qualiclip+-spaq`、`maniqa`、`arniqa-spaq`、`topiq_nr`、`topiq_nr-spaq`
- 美学类：`clipiqa+_vitL14_512`、`musiq-ava`、`laion_aes`、`paq2piq`
- 对齐类：`clipscore`

说明：

- 综合排序节点默认使用 `qualiclip+`、`laion_aes`、`clipscore`
- 对齐类指标依赖 prompt，没有 prompt 时不适合使用
- 当前代码里所有内置指标默认都是“分数越高越好”

## 安装

将本目录放入 `ComfyUI/custom_nodes` 后，在 ComfyUI 对应 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖：

- `pyiqa`
- `setuptools<81`

如果遇到 `pkg_resources` 相关报错，通常是当前环境缺少兼容版本的 `setuptools`。

## 节点一览

### 评分与排序

- `EvalKit Metric Score`
  - 对输入 `IMAGE` 计算单一指标，输出均值、最小值、最大值和 `report`
- `EvalKit Metric Rank`
  - 对同分辨率 `IMAGE batch` 按单一指标排序，输出排序后的 batch 与最佳图
- `EvalKit Preset Rank`
  - 对同分辨率 `IMAGE batch` 按质量 / 美学 / 对齐三类指标加权综合排序

### 路径工作流

- `EvalKit Batch Load From Path`
  - 从文件夹读取图片，统一成标准 batch，适合接普通 batch 节点
- `EvalKit Metric Rank From Path`
  - 逐张读取文件夹图片并按单一指标排序，适合不同分辨率图片直接筛选
- `EvalKit Preset Rank From Path`
  - 逐张读取文件夹图片并按综合分数排序，适合真实出图目录的批量挑图

### 结果处理

- `EvalKit Ranking Preview`
  - 预览 `report` 中 top / bottom N 的拼图与摘要
- `EvalKit Ranking Export`
  - 导出 top / bottom N 图片，可选导出同名 txt prompt
- `EvalKit Score Summary`
  - 对多个外部分数做加权汇总

## compare_mode 说明

`EvalKit Metric Rank From Path` 和 `EvalKit Preset Rank From Path` 支持 3 种比较方式：

- `original`：保持原图尺寸与比例，不拉伸、不补边，适合日常筛图
- `pad`：等比缩放后补黑边到目标尺寸，适合减少尺寸差异
- `resize`：直接缩放到目标尺寸，适合严格标准化对比

建议：

- 日常挑图优先用 `original`
- 做横向 benchmark 优先用 `pad` 或 `resize`

## prompt 读取规则

路径类节点会按以下优先级读取 prompt：

1. `prompt_folder_path` 下同名 `.txt`
2. 图片目录下同名 `.txt`
3. 图片元数据中的 `prompt` / `parameters` / `Description` / `Comment`

当 prompt 缺失时：

- `clipscore` 这类对齐类单指标节点会直接报错
- 综合排序节点会给出提醒，并在必要时自动关闭对齐类指标

## 推荐用法

### 真实出图目录挑图

- 使用 `EvalKit Preset Rank From Path`
- `compare_mode = original`
- 将输出 `report` 接到 `EvalKit Ranking Preview` 或 `EvalKit Ranking Export`

### 同一批图做标准化对比

- 方式一：`EvalKit Preset Rank From Path` + `compare_mode = pad/resize`
- 方式二：`EvalKit Batch Load From Path` → `EvalKit Preset Rank`

## report 包含什么

路径排序节点输出的 `report` 是 JSON 字符串，通常包含：

- 文件名与图片路径
- prompt 内容、来源与 txt 路径
- 原始尺寸与处理后尺寸
- 单项分数或综合分数
- 排名与 warning

## 注意事项

- 不同分辨率图片直接比较时，分数会受到尺寸差异影响
- `original` 更适合筛图，不一定适合严格 benchmark
- 对齐类指标必须结合 prompt 才有意义
- 路径节点支持的图片格式包括 `png`、`jpg`、`jpeg`、`webp`、`bmp`、`tif`、`tiff`
