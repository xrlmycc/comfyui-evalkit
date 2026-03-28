# ComfyUI EvalKit

[English](./README.md) | [简体中文](./README.zh-CN.md)

> Image evaluation nodes for ComfyUI.
> Score, rank, preview, and export image results with single metrics or weighted presets.

## ✨ Features

- 📊 `EvalKit Metric Score` can score connected `IMAGE` / `IMAGE batch` inputs
- 🏆 `EvalKit Metric Rank` and `EvalKit Preset Rank` can rank connected `IMAGE` / `IMAGE batch` inputs
- 📁 Rank images directly from a folder without building a fixed-size batch first
- 🖼️ Ranking nodes support `separate / pad / resize / crop` resolution handling modes
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
  - Score input `IMAGE` / `IMAGE batch` with one metric, with `separate / pad / resize / crop`
- `EvalKit Metric Rank`
  - Rank input `IMAGE` / `IMAGE batch` by one metric and output `ranked_images` + `best_image`
- `EvalKit Preset Rank`
  - Rank input `IMAGE` / `IMAGE batch` with weighted quality / aesthetic / alignment scores and output `ranked_images`, `best_image`, `quality_image`, `aesthetic_image`, and `alignment_image`

### Folder-Based Workflow

- `EvalKit Batch Load From Path`
  - Load images from a folder and normalize them into a standard batch with pad / stretch / crop
- `EvalKit Metric Rank From Path`
  - Read images one by one from a folder, rank them by a single metric, and output the full ranked batch
- `EvalKit Preset Rank From Path`
  - Read images one by one from a folder, rank them by a weighted preset, and output the full ranked batch plus the top image for each metric; Enabling `alignment_metric` requires a prompt; otherwise, alignment metrics sorting will be skipped.

### Result Processing

- `EvalKit Ranking Preview`
  - Preview top / bottom results from a ranking report
- `EvalKit Ranking Export`
  - Export top / bottom images and optionally matching prompt `.txt` files
- `EvalKit Score Summary`
  - Merge external scores with custom weights

## ⚖️ compare_mode

`EvalKit Metric Score`, `EvalKit Metric Rank`, `EvalKit Preset Rank`, `EvalKit Metric Rank From Path`, and `EvalKit Preset Rank From Path` support 4 compare modes:

- `separate` — score each image at its original resolution, this is the default
- `pad` — scale with aspect ratio preserved, then pad to target size
- `resize` — directly stretch to target size
- `crop` — scale to fill, then center crop to target size

Notes:

- `separate` does not change the image size used for scoring
- In `separate`, images are still scored one by one at their original sizes even if resolutions differ
- `ranked_images` only pads smaller images with black borders to build a valid ComfyUI IMAGE batch, without cropping or stretching

Recommended usage:

- Use `separate` for real-world image picking
- Use `pad`, `resize`, or `crop` for more standardized comparisons

## 📝 Prompt Loading Rules

Path-based nodes read prompt text in this order:

1. Matching `.txt` under `prompt_folder_path`
2. Matching `.txt` in the image folder
3. Image metadata such as `prompt`, `parameters`, `Description`, or `Comment`

If prompt text is missing:

- Alignment-only ranking with `clipscore` cannot run
- `EvalKit Preset Rank From Path` raises an error when `alignment_metric` is enabled, Alignment-related metrics will be skipped in the sorting process.

## 🔧 Recommended Workflows

### Pick the best image from a real output folder

- `EvalKit Preset Rank From Path`
- `compare_mode = separate`
- Send the `report` output into `EvalKit Ranking Preview` or `EvalKit Ranking Export`

### Run more standardized comparisons

- Option A: `EvalKit Preset Rank From Path` with `compare_mode = pad` / `resize` / `crop`
- Option B: `EvalKit Batch Load From Path` → `EvalKit Preset Rank`

## 📄 What's Inside report

Scoring and ranking nodes output a JSON `report` that usually includes:

- file names and image paths
- prompt text, source, and prompt file path
- original size and processed size
- single metric scores or combined scores
- ranking results and warnings

## ⚠️ Notes

- Comparing different resolutions can affect scores
- `separate` is often best for practical selection, not strict benchmarking
- In `separate`, `ranked_images` may appear larger because batch output uses black padding for mixed resolutions
- Alignment metrics only make sense when prompt text is available
- Path nodes support `png`, `jpg`, `jpeg`, `webp`, `bmp`, `tif`, and `tiff`
