# ComfyUI EvalKit

[English](./README.md) | [简体中文](./README.zh-CN.md)

> ComfyUI 文生图评测节点包。
> 支持单指标打分、综合排序、排名预览与结果导出。

## ✨ 功能

- 📊 对 `IMAGE` 或 `IMAGE batch` 进行单指标打分
- 🏆 对同尺寸 batch 做单指标或综合排序
- 📁 直接从文件夹逐张读取图片并排序，不要求先整理成同尺寸 batch
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
  - 对输入 `IMAGE` 计算单一指标，输出 mean / min / max / count / report
- `EvalKit Metric Rank`
  - 对同分辨率 `IMAGE batch` 按单一指标排序
- `EvalKit Preset Rank`
  - 对同分辨率 `IMAGE batch` 按质量 / 美学 / 对齐三类指标加权综合排序

### 路径工作流

- `EvalKit Batch Load From Path`
  - 从文件夹读取图片并整理成标准 batch
- `EvalKit Metric Rank From Path`
  - 逐张读取文件夹图片并按单一指标排序
- `EvalKit Preset Rank From Path`
  - 逐张读取文件夹图片并按综合分数排序

### 结果处理

- `EvalKit Ranking Preview`
  - 预览 ranking report 中的 top / bottom 结果
- `EvalKit Ranking Export`
  - 导出 top / bottom 图片，并可选导出同名 prompt `.txt`
- `EvalKit Score Summary`
  - 对外部分数做加权汇总

## ⚖️ compare_mode

`EvalKit Metric Rank From Path` 和 `EvalKit Preset Rank From Path` 支持 3 种比较模式：

- `original` — 保持原图尺寸与比例，不拉伸、不补边
- `pad` — 等比缩放后补黑边到目标尺寸
- `resize` — 直接缩放到目标尺寸

建议：

- 日常筛图优先使用 `original`
- 标准化对比优先使用 `pad` 或 `resize`

## 📝 prompt 读取规则

路径类节点按以下优先级读取 prompt：

1. `prompt_folder_path` 下同名 `.txt`
2. 图片目录下同名 `.txt`
3. 图片元数据中的 `prompt`、`parameters`、`Description`、`Comment`

当 prompt 缺失时：

- `clipscore` 这类对齐类单指标排序无法执行
- 综合排序会给出 warning，并在必要时自动关闭对齐类指标

## 🔧 推荐工作流

### 从真实出图目录中挑选最佳结果

- `EvalKit Preset Rank From Path`
- `compare_mode = original`
- 将 `report` 输出接到 `EvalKit Ranking Preview` 或 `EvalKit Ranking Export`

### 做更标准化的横向对比

- 方式一：`EvalKit Preset Rank From Path` + `compare_mode = pad` 或 `resize`
- 方式二：`EvalKit Batch Load From Path` → `EvalKit Preset Rank`

## 📄 report 包含内容

路径排序节点输出的 JSON `report` 通常包含：

- 文件名与图片路径
- prompt 文本、来源与 prompt 文件路径
- 原始尺寸与处理后尺寸
- 单项分数或综合分数
- 排名结果与 warnings

## ⚠️ 注意事项

- 不同分辨率图片直接比较时，分数可能受到尺寸差异影响
- `original` 更适合真实筛图，不一定适合严格 benchmark
- 对齐类指标必须结合 prompt 才有意义
- 路径节点支持 `png`、`jpg`、`jpeg`、`webp`、`bmp`、`tif`、`tiff`
