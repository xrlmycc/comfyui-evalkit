# ComfyUI EvalKit

[English](./README.md) | [简体中文](./README.zh-CN.md)

> Image evaluation nodes for ComfyUI.
> Score, rank, preview, and export image results with single metrics or weighted presets.

## ✨ Features

- 📊 Score `IMAGE` or `IMAGE batch` with a single evaluation metric
- 🏆 Rank same-size batches by one metric or by weighted preset scores
- 📁 Rank images directly from a folder without building a fixed-size batch first
- 📝 Auto-load prompts from matching `.txt` files or image metadata
- 📦 Reuse JSON `report` output for preview and export workflows

## 🧠 Supported Metrics

This project uses `pyiqa` as the evaluation backend.

- Quality: `hyperiqa`, `dbcnn`, `qualiclip+`, `qualiclip+-spaq`, `maniqa`, `arniqa-spaq`, `topiq_nr`, `topiq_nr-spaq`
- Aesthetic: `clipiqa+_vitL14_512`, `musiq-ava`, `laion_aes`, `paq2piq`
- Alignment: `clipscore`

Notes:

- Default preset ranking uses `qualiclip+`, `laion_aes`, and `clipscore`
- Alignment metrics require prompt text
- Current built-in metrics are configured as higher-is-better

## 🚀 Installation

Place this folder under `ComfyUI/custom_nodes`, then install dependencies in the Python environment used by ComfyUI:

```bash
python -m pip install -r requirements.txt
```

Dependencies:

- `pyiqa`
- `setuptools<81`

If you see a `pkg_resources` error, the environment usually needs a compatible `setuptools` version.

## 🧩 Nodes

### Scoring & Ranking

- `EvalKit Metric Score`
  - Score input `IMAGE` with one metric and return mean / min / max / count / report
- `EvalKit Metric Rank`
  - Rank a same-resolution `IMAGE batch` by one metric
- `EvalKit Preset Rank`
  - Rank a same-resolution `IMAGE batch` with weighted quality / aesthetic / alignment scores

### Folder-Based Workflow

- `EvalKit Batch Load From Path`
  - Load images from a folder and normalize them into a standard batch
- `EvalKit Metric Rank From Path`
  - Read images one by one from a folder and rank them by a single metric
- `EvalKit Preset Rank From Path`
  - Read images one by one from a folder and rank them by a weighted preset

### Result Processing

- `EvalKit Ranking Preview`
  - Preview top / bottom results from a ranking report
- `EvalKit Ranking Export`
  - Export top / bottom images and optionally matching prompt `.txt` files
- `EvalKit Score Summary`
  - Merge external scores with custom weights

## ⚖️ compare_mode

`EvalKit Metric Rank From Path` and `EvalKit Preset Rank From Path` support 3 compare modes:

- `original` — keep original size and aspect ratio, no stretch, no padding
- `pad` — scale with aspect ratio preserved, then pad to target size
- `resize` — directly resize to target size

Recommended usage:

- Use `original` for real-world image picking
- Use `pad` or `resize` for more standardized comparisons

## 📝 Prompt Loading Rules

Folder-based nodes read prompt text in this order:

1. Matching `.txt` under `prompt_folder_path`
2. Matching `.txt` in the image folder
3. Image metadata such as `prompt`, `parameters`, `Description`, or `Comment`

If prompt text is missing:

- Alignment-only ranking with `clipscore` cannot run
- Preset ranking warns and can automatically disable alignment scoring when needed

## 🔧 Recommended Workflows

### Pick the best image from a real output folder

- `EvalKit Preset Rank From Path`
- `compare_mode = original`
- Send the `report` output into `EvalKit Ranking Preview` or `EvalKit Ranking Export`

### Run more standardized comparisons

- Option A: `EvalKit Preset Rank From Path` with `compare_mode = pad` or `resize`
- Option B: `EvalKit Batch Load From Path` → `EvalKit Preset Rank`

## 📄 What's Inside report

The path ranking nodes output a JSON `report` that usually includes:

- file names and image paths
- prompt text, source, and prompt file path
- original size and processed size
- single metric scores or combined scores
- ranking results and warnings

## ⚠️ Notes

- Comparing different resolutions can affect scores
- `original` is often best for practical selection, not strict benchmarking
- Alignment metrics only make sense when prompt text is available
- Supported image formats include `png`, `jpg`, `jpeg`, `webp`, `bmp`, `tif`, and `tiff`
