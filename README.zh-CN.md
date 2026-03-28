# ComfyUI EvalKit

[English](./README.md) | [简体中文](./README.zh-CN.md)

> ComfyUI 文生图评测节点包。
> 支持单指标打分、综合排序、排名预览与结果导出。

## ✨ 功能

- 📊 `EvalKit Metric Score` 支持直接连接 `IMAGE` / `IMAGE batch` 批量打分
- 🏆 `EvalKit Metric Rank` 与 `EvalKit Preset Rank` 支持直接连接 `IMAGE` / `IMAGE batch` 批量排序
- 📁 直接从文件夹逐张读取图片并排序，不要求先整理成同尺寸 batch
- 🖼️ 排序节点支持 `separate / pad / resize / crop` 4 种分辨率处理模式
- 📝 自动读取同名 `.txt` 或图片元数据中的 prompt
- 📦 输出 `report` 供预览、导出和后续处理节点复用

## 🧠 支持的评测指标 / 模型

本项目通过 `pyiqa` 调用评测模型，当前支持：

- 质量类：`hyperiqa`、`dbcnn`、`qualiclip+`、`qualiclip+-spaq`、`maniqa`、`arniqa-spaq`、`topiq_nr`、`topiq_nr-spaq`
- 美学类：`clipiqa+_vitL14_512`、`musiq-ava`、`laion_aes`、`paq2piq`
- 对齐类：`clipscore`

说明：

- 综合排序默认使用 `qualiclip+`、`laion_aes`、`clipscore`
- 对齐类指标依赖 prompt
- 当前内置指标默认按“分数越高越好”处理

## 🚀 安装

将本目录放入 `ComfyUI/custom_nodes` 后，在 ComfyUI 使用的 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖：

- `pyiqa`
- `setuptools<81`

如果遇到 `pkg_resources` 报错，通常是当前环境缺少兼容版本的 `setuptools`。

## 🧩 节点

### 评分与排序

- `EvalKit Metric Score`
  - 对输入 `IMAGE` / `IMAGE batch` 计算单一指标，支持 `separate / pad / resize / crop`
- `EvalKit Metric Rank`
  - 对输入 `IMAGE` / `IMAGE batch` 按单一指标排序，输出 `ranked_images` + `best_image`
- `EvalKit Preset Rank`
  - 对输入 `IMAGE` / `IMAGE batch` 按质量 / 美学 / 对齐三类指标加权综合排序，输出 `ranked_images`、`best_image`、`quality_image`、`aesthetic_image`、`alignment_image`

### 路径工作流

- `EvalKit Batch Load From Path`
  - 从文件夹读取图片并整理成标准 batch，支持补边 / 拉伸 / 裁剪
- `EvalKit Metric Rank From Path`
  - 逐张读取文件夹图片并按单一指标排序，同时输出完整排名图片 batch
- `EvalKit Preset Rank From Path`
  - 逐张读取文件夹图片并按综合分数排序，同时输出完整排名图片 batch，以及各单项指标最高分图片；启用 `alignment_metric` 需要填写prompt，否则会略过对齐类指标排序

### 结果处理

- `EvalKit Ranking Preview`
  - 预览 ranking report 中的 top / bottom 结果
- `EvalKit Ranking Export`
  - 导出 top / bottom 图片，并可选导出同名 prompt `.txt`
- `EvalKit Score Summary`
  - 对外部分数做加权汇总

## ⚖️ compare_mode

`EvalKit Metric Score`、`EvalKit Metric Rank`、`EvalKit Preset Rank`、`EvalKit Metric Rank From Path` 和 `EvalKit Preset Rank From Path` 都支持 4 种比较模式：

- `separate` — 每张图按原始分辨率单独计算分数，默认模式
- `pad` — 等比缩放后补黑边到目标尺寸
- `resize` — 直接拉伸到目标尺寸
- `crop` — 等比放大后居中裁剪到目标尺寸

说明：

- `separate` 不会修改评分时使用的图片尺寸
- `separate` 下如果图片分辨率不同，仍然会逐张按原尺寸计算分数
- `ranked_images` 只是为了组成 ComfyUI 的 IMAGE batch 才会对较小图片补黑边到当前批次最大尺寸，不会裁剪、不会拉伸

建议：

- 日常筛图优先使用 `separate`
- 标准化对比优先使用 `pad`、`resize` 或 `crop`

## 📝 prompt 读取规则

路径输入节点按以下优先级读取 prompt：

1. `prompt_folder_path` 下同名 `.txt`
2. 图片目录下同名 `.txt`
3. 图片元数据中的 `prompt`、`parameters`、`Description`、`Comment`

当 prompt 缺失时：

- `clipscore` 这类对齐类单指标排序无法执行
- `EvalKit Preset Rank From Path` 如果启用了 `alignment_metric`，会略过对齐类指标排序

## 🔧 推荐工作流

### 从真实出图目录中挑选最佳结果

- `EvalKit Preset Rank From Path`
- `compare_mode = separate`
- 将 `report` 输出接到 `EvalKit Ranking Preview` 或 `EvalKit Ranking Export`

### 做更标准化的横向对比

- 方式一：`EvalKit Preset Rank From Path` + `compare_mode = pad` / `resize` / `crop`
- 方式二：`EvalKit Batch Load From Path` → `EvalKit Preset Rank`

## 📄 report 包含内容

评分 / 排序节点输出的 JSON `report` 通常包含：

- 文件名与图片路径
- prompt 文本、来源与 prompt 文件路径
- 原始尺寸与处理后尺寸
- 单项分数或综合分数
- 排名结果与 warnings

## ⚠️ 注意事项

- 不同分辨率图片直接比较时，分数可能受到尺寸差异影响
- `separate` 更适合真实筛图，不一定适合严格 benchmark
- `separate` 下若图片分辨率不一致，`ranked_images` 可能会因为 batch 输出而出现补黑边后的尺寸
- 对齐类指标必须结合 prompt 才有意义
- 路径节点支持 `png`、`jpg`、`jpeg`、`webp`、`bmp`、`tif`、`tiff`
