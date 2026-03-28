import importlib
import json
import os
import re
import shutil
import sys
import types
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None


METRIC_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "hyperiqa": {"category": "quality", "lower_better": False},
    "dbcnn": {"category": "quality", "lower_better": False},
    "qualiclip+": {"category": "quality", "lower_better": False},
    "qualiclip+-spaq": {"category": "quality", "lower_better": False},
    "maniqa": {"category": "quality", "lower_better": False},
    "arniqa-spaq": {"category": "quality", "lower_better": False},
    "topiq_nr": {"category": "quality", "lower_better": False},
    "topiq_nr-spaq": {"category": "quality", "lower_better": False},
    "clipiqa+_vitL14_512": {"category": "aesthetic", "lower_better": False},
    "musiq-ava": {"category": "aesthetic", "lower_better": False},
    "laion_aes": {"category": "aesthetic", "lower_better": False},
    "paq2piq": {"category": "aesthetic", "lower_better": False},
    "clipscore": {"category": "alignment", "lower_better": False},
}

QUALITY_METRICS = tuple(
    metric_name
    for metric_name, config in METRIC_DEFINITIONS.items()
    if config["category"] == "quality"
)
AESTHETIC_METRICS = tuple(
    metric_name
    for metric_name, config in METRIC_DEFINITIONS.items()
    if config["category"] == "aesthetic"
)
ALIGNMENT_METRICS = tuple(
    metric_name
    for metric_name, config in METRIC_DEFINITIONS.items()
    if config["category"] == "alignment"
)
ALL_METRICS = tuple(METRIC_DEFINITIONS.keys())
OPTIONAL_QUALITY_METRICS = ("disabled",) + QUALITY_METRICS
OPTIONAL_AESTHETIC_METRICS = ("disabled",) + AESTHETIC_METRICS
OPTIONAL_ALIGNMENT_METRICS = ("disabled",) + ALIGNMENT_METRICS

_METRIC_CACHE: Dict[Tuple[str, str], object] = {}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
BATCH_RESIZE_MODES = ("pad", "resize", "crop")
COMPARE_MODES = ("separate", "pad", "resize", "crop")


def _ensure_pkg_resources_compat():
    if "pkg_resources" in sys.modules:
        return

    try:
        import pkg_resources  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    try:
        packaging = importlib.import_module("packaging")
        if not hasattr(packaging, "version"):
            packaging.version = importlib.import_module("packaging.version")

        compat_module = types.ModuleType("pkg_resources")
        compat_module.packaging = packaging
        sys.modules["pkg_resources"] = compat_module
    except Exception:
        pass


def _get_torch_device(device_mode: str):
    if device_mode == "cpu":
        return torch.device("cpu")

    if comfy_model_management is not None:
        try:
            return comfy_model_management.get_torch_device()
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _get_pyiqa():
    try:
        import pyiqa

        return pyiqa
    except Exception as exc:
        raise RuntimeError(
            "未找到 pyiqa。请先在 ComfyUI 对应环境中安装本节点包 requirements.txt 里的依赖。"
        ) from exc


def _get_metric(metric_name: str, device_mode: str):
    pyiqa = _get_pyiqa()
    _ensure_pkg_resources_compat()
    device = _get_torch_device(device_mode)
    cache_key = (metric_name, str(device))

    if cache_key not in _METRIC_CACHE:
        try:
            metric = pyiqa.create_metric(metric_name, device=device)
        except ModuleNotFoundError as exc:
            if exc.name == "pkg_resources":
                raise RuntimeError(
                    "当前指标依赖的 clip 包需要 pkg_resources。请在对应 ComfyUI 虚拟环境中安装 setuptools<81 后重启 ComfyUI。"
                ) from exc
            raise
        metric.eval()
        _METRIC_CACHE[cache_key] = metric

    return _METRIC_CACHE[cache_key], device


def _ensure_batch_image(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("IMAGE 输入必须是 torch.Tensor")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("IMAGE 输入必须是 HWC 或 BHWC")
    if image.shape[-1] < 3:
        raise ValueError("IMAGE 输入至少需要 3 个通道")
    return image[..., :3].float().clamp(0.0, 1.0).contiguous()


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2).contiguous()


def _resolve_captions(prompt: str, batch_size: int) -> List[str]:
    prompt = prompt or ""
    if batch_size == 1:
        return [prompt]

    try:
        prompt_payload = json.loads(prompt)
        if isinstance(prompt_payload, list):
            prompt_values = [str(item) if item is not None else "" for item in prompt_payload]
            if len(prompt_values) == batch_size:
                return prompt_values
    except Exception:
        pass

    prompt_lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if len(prompt_lines) == batch_size:
        return prompt_lines

    return [prompt] * batch_size


def _tensor_to_scores(output, batch_size: int) -> List[float]:
    if isinstance(output, tuple):
        output = output[0]

    if not torch.is_tensor(output):
        output = torch.tensor(output, dtype=torch.float32)

    scores = output.detach().float().cpu().reshape(-1)

    if scores.numel() == batch_size:
        return [float(value) for value in scores.tolist()]

    if scores.numel() == 1:
        return [float(scores.item()) for _ in range(batch_size)]

    if scores.numel() % batch_size == 0:
        scores = scores.reshape(batch_size, -1).mean(dim=1)
        return [float(value) for value in scores.tolist()]

    raise RuntimeError("评分结果形状异常，无法按批次还原。")


def _metric_lower_better(metric_name: str, sort_mode: str) -> bool:
    if sort_mode == "higher_better":
        return False
    if sort_mode == "lower_better":
        return True
    return bool(METRIC_DEFINITIONS[metric_name]["lower_better"])


def _normalize_scores(scores: List[float], lower_better: bool) -> List[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]

    min_value = min(scores)
    max_value = max(scores)

    if max_value == min_value:
        return [0.5] * len(scores)

    if lower_better:
        return [1.0 - (score - min_value) / (max_value - min_value) for score in scores]

    return [(score - min_value) / (max_value - min_value) for score in scores]


def _run_metric(metric_name: str, image: torch.Tensor, prompt: str, device_mode: str) -> Tuple[List[float], Dict[str, object]]:
    batch_image = _ensure_batch_image(image)
    metric, device = _get_metric(metric_name, device_mode)
    input_image = _to_nchw(batch_image).to(device)
    kwargs = {}

    if metric_name == "clipscore":
        if not prompt.strip():
            raise ValueError("clipscore 需要填写 prompt。")
        kwargs["caption_list"] = _resolve_captions(prompt, input_image.shape[0])

    with torch.inference_mode():
        output = metric(input_image, **kwargs)

    scores = _tensor_to_scores(output, input_image.shape[0])
    summary = {
        "metric": metric_name,
        "device": str(device),
        "count": len(scores),
        "scores": scores,
        "mean": float(sum(scores) / len(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
    }
    return scores, summary


def _build_score_report(metric_name: str, scores: List[float], extra: Dict[str, object] = None) -> str:
    payload = {
        "metric": metric_name,
        "scores": [float(value) for value in scores],
        "count": len(scores),
        "mean": float(sum(scores) / len(scores)) if scores else 0.0,
        "min": float(min(scores)) if scores else 0.0,
        "max": float(max(scores)) if scores else 0.0,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _list_image_files(folder_path: str) -> List[str]:
    image_paths = []
    for file_name in sorted(os.listdir(folder_path), key=_natural_sort_key):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(IMAGE_EXTENSIONS):
            image_paths.append(file_path)
    return image_paths


def _parse_comfy_prompt_text(prompt_payload):
    if not isinstance(prompt_payload, dict):
        return ""

    texts = []
    for node_data in prompt_payload.values():
        if not isinstance(node_data, dict):
            continue
        inputs = node_data.get("inputs")
        if not isinstance(inputs, dict):
            continue
        text_value = inputs.get("text")
        if isinstance(text_value, str) and text_value.strip():
            texts.append(text_value.strip())

    unique_texts = []
    seen = set()
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    return "\n".join(unique_texts)


def _parse_image_metadata_prompt(img: Image.Image) -> str:
    info = dict(getattr(img, "info", {}) or {})

    direct_keys = ("prompt", "parameters", "Description", "Comment", "comment")
    for key in direct_keys:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            if key == "prompt":
                try:
                    prompt_payload = json.loads(value)
                    prompt_text = _parse_comfy_prompt_text(prompt_payload)
                    if prompt_text:
                        return prompt_text
                except Exception:
                    return value.strip()
            return value.strip()

    exif = None
    try:
        exif = img.getexif()
    except Exception:
        exif = None

    if exif:
        for key in (0x010E, 0x9286):
            value = exif.get(key)
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="ignore")
                except Exception:
                    value = ""
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _load_prompt_text(image_path: str, prompt_folder_path: str):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    candidate_directories = []

    if prompt_folder_path and prompt_folder_path.strip():
        candidate_directories.append(prompt_folder_path.strip())

    candidate_directories.append(os.path.dirname(image_path))

    for directory in candidate_directories:
        txt_path = os.path.join(directory, f"{base_name}.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
            if text:
                return text, txt_path, "txt"

    with Image.open(image_path) as img:
        prompt_text = _parse_image_metadata_prompt(img)
    if prompt_text:
        return prompt_text, "", "metadata"

    return "", "", "empty"


def _normalize_compare_mode(compare_mode: str) -> str:
    if compare_mode == "original":
        return "separate"
    return compare_mode


def _resize_image_for_batch(image: Image.Image, target_width: int, target_height: int, resize_mode: str):
    image = image.convert("RGB")
    if resize_mode == "resize":
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    if resize_mode == "crop":
        return ImageOps.fit(image, (target_width, target_height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    contained = ImageOps.contain(image, (target_width, target_height), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    offset_x = (target_width - contained.width) // 2
    offset_y = (target_height - contained.height) // 2
    background.paste(contained, (offset_x, offset_y))
    return background


def _pil_to_comfy_tensor(image: Image.Image):
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def _load_image_tensor_from_path(image_path: str):
    with Image.open(image_path) as image_file:
        image_file = ImageOps.exif_transpose(image_file)
        original_width, original_height = image_file.size
        tensor = _pil_to_comfy_tensor(image_file)
    return tensor, (int(original_width), int(original_height))


def _stack_image_tensors(image_tensors):
    normalized_items = []
    for tensor in image_tensors:
        batch_tensor = _ensure_batch_image(tensor)
        for index in range(batch_tensor.shape[0]):
            normalized_items.append(batch_tensor[index : index + 1])

    if not normalized_items:
        raise ValueError("没有可拼接的图片。")

    max_height = max(item.shape[1] for item in normalized_items)
    max_width = max(item.shape[2] for item in normalized_items)
    stacked = []

    for item in normalized_items:
        height = item.shape[1]
        width = item.shape[2]
        if height == max_height and width == max_width:
            stacked.append(item)
            continue

        canvas = torch.zeros(
            (1, max_height, max_width, item.shape[3]),
            dtype=item.dtype,
            device=item.device,
        )
        offset_y = (max_height - height) // 2
        offset_x = (max_width - width) // 2
        canvas[:, offset_y : offset_y + height, offset_x : offset_x + width, :] = item
        stacked.append(canvas)

    return torch.cat(stacked, dim=0)


def _build_ranked_images_output(image_tensors, compare_mode: str):
    normalized_items = []
    for tensor in image_tensors:
        batch_tensor = _ensure_batch_image(tensor)
        for index in range(batch_tensor.shape[0]):
            normalized_items.append(batch_tensor[index : index + 1])

    if not normalized_items:
        raise ValueError("没有可输出的排名图片。")

    return _stack_image_tensors(normalized_items)


def _load_image_for_compare(image_path: str, compare_mode: str, target_width: int, target_height: int):
    compare_mode = _normalize_compare_mode(compare_mode)
    with Image.open(image_path) as image_file:
        image_file = ImageOps.exif_transpose(image_file)
        original_width, original_height = image_file.size
        processed_image = image_file.convert("RGB")

        if compare_mode in ("pad", "resize", "crop"):
            processed_image = _resize_image_for_batch(
                processed_image,
                target_width=target_width,
                target_height=target_height,
                resize_mode=compare_mode,
            )

        tensor = _pil_to_comfy_tensor(processed_image)
        processed_width, processed_height = processed_image.size

    return tensor, (int(original_width), int(original_height)), (int(processed_width), int(processed_height))


def _comfy_tensor_to_pil(image: torch.Tensor):
    batch_image = _ensure_batch_image(image)
    if batch_image.shape[0] != 1:
        raise ValueError("转换 PIL 时只支持单张图片。")
    image_array = (batch_image[0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_array, mode="RGB")


def _collect_image_entries(
    image,
    prompt: str = "",
    compare_mode: str = "separate",
    target_width: int = 1024,
    target_height: int = 1024,
):
    compare_mode = _normalize_compare_mode(compare_mode)
    batch_image = _ensure_batch_image(image)
    prompt_values = _resolve_captions(prompt, batch_image.shape[0])
    missing_prompt_count = sum(1 for value in prompt_values if not str(value).strip())
    entries = []

    for index in range(batch_image.shape[0]):
        image_tensor = batch_image[index : index + 1]
        original_height = int(image_tensor.shape[1])
        original_width = int(image_tensor.shape[2])
        processed_tensor = image_tensor
        processed_width = original_width
        processed_height = original_height

        if compare_mode != "separate":
            processed_image = _resize_image_for_batch(
                _comfy_tensor_to_pil(image_tensor),
                target_width=target_width,
                target_height=target_height,
                resize_mode=compare_mode,
            )
            processed_tensor = _pil_to_comfy_tensor(processed_image)
            processed_width, processed_height = processed_image.size

        entries.append(
            {
                "index": index,
                "file_name": f"input_{index:04d}",
                "image_path": "",
                "prompt_text": prompt_values[index],
                "prompt_source": "input",
                "prompt_path": "",
                "original_size": [original_width, original_height],
                "processed_size": [int(processed_width), int(processed_height)],
                "image_tensor": processed_tensor,
            }
        )

    warnings = []
    notice = ""
    if missing_prompt_count > 0:
        notice = f"检测到 {missing_prompt_count}/{len(entries)} 张输入图片缺少 prompt。"
        warnings.append(notice)

    return entries, {
        "source_mode": "image",
        "warnings": warnings,
        "notice": notice,
        "missing_prompt_count": missing_prompt_count,
        "compare_mode": compare_mode,
        "target_size": [int(target_width), int(target_height)],
    }


def _run_metric_on_entries(metric_name: str, entries, device_mode: str):
    raw_scores = []
    device = ""

    for entry in entries:
        scores, summary = _run_metric(metric_name, entry["image_tensor"], entry["prompt_text"], device_mode)
        raw_scores.append(float(scores[0]))
        device = str(summary["device"])

    return raw_scores, {
        "metric": metric_name,
        "device": device,
        "count": len(raw_scores),
        "scores": raw_scores,
        "mean": float(sum(raw_scores) / len(raw_scores)) if raw_scores else 0.0,
        "min": float(min(raw_scores)) if raw_scores else 0.0,
        "max": float(max(raw_scores)) if raw_scores else 0.0,
    }


def _resolve_metric_best_index(metric_result: Dict[str, object], fallback_index: int):
    scores = metric_result.get("scores", [])
    if not scores:
        return int(fallback_index)
    return int(max(range(len(scores)), key=lambda index: float(scores[index])))


def _collect_folder_items(image_folder_path: str, prompt_folder_path: str = "", limit: int = 0):
    folder_path = (image_folder_path or "").strip()
    if not folder_path:
        raise ValueError("image_folder_path 不能为空。")
    if not os.path.isdir(folder_path):
        raise ValueError(f"图片目录不存在: {folder_path}")

    prompt_path = (prompt_folder_path or "").strip()
    if prompt_path and not os.path.isdir(prompt_path):
        raise ValueError(f"prompt 目录不存在: {prompt_path}")

    image_files = _list_image_files(folder_path)
    if limit > 0:
        image_files = image_files[:limit]
    if not image_files:
        raise ValueError("指定目录下没有找到可加载的图片。")

    items = []
    for image_path in image_files:
        prompt_text, source_path, source_type = _load_prompt_text(image_path, prompt_path)
        items.append(
            {
                "image_path": image_path,
                "file_name": os.path.basename(image_path),
                "prompt_text": prompt_text,
                "prompt_path": source_path,
                "prompt_source": source_type,
            }
        )
    return items, folder_path, prompt_path


def _collect_prompt_warnings(items, prompt_folder_path: str):
    missing_prompt_count = sum(1 for item in items if not item["prompt_text"].strip())
    warnings = []
    notice = ""

    if missing_prompt_count <= 0:
        return warnings, notice, missing_prompt_count

    prompt_hint = "请提供 prompt_folder_path，或在图片目录下放置同名 txt，或在图片元数据中写入 prompt。"
    if prompt_folder_path.strip():
        notice = (
            f"检测到 {missing_prompt_count}/{len(items)} 张图片缺少 prompt。"
            " 建议检查 prompt_folder_path 下是否存在与图片同名的 txt，或关闭对齐类指标。"
        )
    else:
        notice = (
            f"检测到 {missing_prompt_count}/{len(items)} 张图片缺少 prompt。"
            " 当前未填写 prompt_folder_path，建议关闭对齐类指标，或补充同名 txt / 图片元数据 prompt。"
        )

    warnings.append(notice)
    warnings.append(prompt_hint)
    return warnings, notice, missing_prompt_count


def _extract_ranking_rows(report: str):
    try:
        payload = json.loads(report)
    except Exception as exc:
        raise ValueError("report 不是合法的 JSON 字符串。") from exc

    ranking = payload.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("report 中未找到 ranking 数据。")

    return payload, ranking


def _select_ranking_rows(report: str, select_mode: str, count: int):
    payload, ranking = _extract_ranking_rows(report)
    actual_count = min(max(int(count), 1), len(ranking))
    if select_mode == "bottom":
        selected = list(reversed(ranking[-actual_count:]))
    else:
        selected = ranking[:actual_count]
    return payload, ranking, selected


def _get_ranking_score_value(row: Dict[str, object]):
    if "combined_score" in row and row["combined_score"] is not None:
        return float(row["combined_score"]), "combined_score"
    if "score" in row and row["score"] is not None:
        return float(row["score"]), "score"
    return 0.0, "score"


def _build_preview_grid(rows, thumbnail_width: int, thumbnail_height: int, columns: int):
    preview_tiles = []
    for row in rows:
        with Image.open(row["image_path"]) as image_file:
            image_file = ImageOps.exif_transpose(image_file)
            tile = _resize_image_for_batch(image_file, thumbnail_width, thumbnail_height, "pad")
        preview_tiles.append(tile)

    column_count = max(1, min(int(columns), len(preview_tiles)))
    row_count = (len(preview_tiles) + column_count - 1) // column_count
    canvas = Image.new("RGB", (column_count * thumbnail_width, row_count * thumbnail_height), (0, 0, 0))

    for index, tile in enumerate(preview_tiles):
        x = (index % column_count) * thumbnail_width
        y = (index // column_count) * thumbnail_height
        canvas.paste(tile, (x, y))

    return _pil_to_comfy_tensor(canvas)


class EvalKitMetricScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "metric_name": (ALL_METRICS,),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "compare_mode": (COMPARE_MODES, {"default": "separate"}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "device_mode": (("auto", "cpu"), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("mean_score", "min_score", "max_score", "count", "report")
    FUNCTION = "score"
    CATEGORY = "evaluation/evalkit"

    def score(
        self,
        image,
        metric_name,
        prompt="",
        compare_mode="separate",
        target_width=1024,
        target_height=1024,
        device_mode="auto",
    ):
        entries, context = _collect_image_entries(
            image=image,
            prompt=prompt,
            compare_mode=compare_mode,
            target_width=target_width,
            target_height=target_height,
        )
        if metric_name in ALIGNMENT_METRICS and context["missing_prompt_count"] > 0:
            raise ValueError("当前输入中存在缺少 prompt 的图片，无法执行对齐类单指标打分。")
        scores, summary = _run_metric_on_entries(metric_name, entries, device_mode)
        report_rows = [
            {
                "index": int(entry["index"]),
                "file_name": entry["file_name"],
                "image_path": entry["image_path"],
                "prompt_text": entry["prompt_text"],
                "prompt_source": entry["prompt_source"],
                "prompt_path": entry["prompt_path"],
                "score": float(scores[index]),
                "original_size": entry["original_size"],
                "processed_size": entry["processed_size"],
            }
            for index, entry in enumerate(entries)
        ]
        report = _build_score_report(
            metric_name,
            scores,
            {
                "device": summary["device"],
                "source_mode": context["source_mode"],
                "compare_mode": context["compare_mode"],
                "target_size": context["target_size"],
                "warnings": context["warnings"],
                "items": report_rows,
            },
        )
        return (
            float(summary["mean"]),
            float(summary["min"]),
            float(summary["max"]),
            int(summary["count"]),
            report,
        )


class EvalKitMetricRank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "metric_name": (ALL_METRICS,),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "compare_mode": (COMPARE_MODES, {"default": "separate"}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "sort_mode": (("auto", "higher_better", "lower_better"), {"default": "auto"}),
                "device_mode": (("auto", "cpu"), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("ranked_images", "best_image", "best_score", "best_index", "report")
    FUNCTION = "rank"
    CATEGORY = "evaluation/evalkit"

    def rank(
        self,
        image,
        metric_name,
        prompt="",
        compare_mode="separate",
        target_width=1024,
        target_height=1024,
        sort_mode="auto",
        device_mode="auto",
    ):
        entries, context = _collect_image_entries(
            image=image,
            prompt=prompt,
            compare_mode=compare_mode,
            target_width=target_width,
            target_height=target_height,
        )
        if metric_name in ALIGNMENT_METRICS and context["missing_prompt_count"] > 0:
            raise ValueError("当前输入中存在缺少 prompt 的图片，无法执行对齐类单指标排名。")
        scores, _ = _run_metric_on_entries(metric_name, entries, device_mode)
        lower_better = _metric_lower_better(metric_name, sort_mode)
        order = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=not lower_better,
        )
        ranked_entries = [entries[index] for index in order]
        ranked_images = _build_ranked_images_output([entry["image_tensor"] for entry in ranked_entries], compare_mode)
        best_index = int(order[0])
        best_score = float(scores[best_index])
        ranking = [
            {
                "rank": rank + 1,
                "index": int(index),
                "file_name": entries[index]["file_name"],
                "image_path": entries[index]["image_path"],
                "prompt_text": entries[index]["prompt_text"],
                "prompt_source": entries[index]["prompt_source"],
                "prompt_path": entries[index]["prompt_path"],
                "score": float(scores[index]),
                "original_size": entries[index]["original_size"],
                "processed_size": entries[index]["processed_size"],
            }
            for rank, index in enumerate(order)
        ]
        report = json.dumps(
            {
                "metric": metric_name,
                "source_mode": context["source_mode"],
                "compare_mode": context["compare_mode"],
                "target_size": context["target_size"],
                "sort_mode": sort_mode,
                "lower_better": lower_better,
                "warnings": context["warnings"],
                "ranking": ranking,
            },
            ensure_ascii=False,
            indent=2,
        )
        return ranked_images, entries[best_index]["image_tensor"], best_score, best_index, report


class EvalKitPresetRank:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "compare_mode": (COMPARE_MODES, {"default": "separate"}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "quality_metric": (OPTIONAL_QUALITY_METRICS, {"default": "qualiclip+"}),
                "aesthetic_metric": (OPTIONAL_AESTHETIC_METRICS, {"default": "laion_aes"}),
                "alignment_metric": (OPTIONAL_ALIGNMENT_METRICS, {"default": "clipscore"}),
                "quality_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "aesthetic_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "alignment_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "device_mode": (("auto", "cpu"), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "ranked_images",
        "best_image",
        "quality_image",
        "aesthetic_image",
        "alignment_image",
        "report",
    )
    FUNCTION = "rank"
    CATEGORY = "evaluation/evalkit"

    def rank(
        self,
        image,
        prompt="",
        compare_mode="separate",
        target_width=1024,
        target_height=1024,
        quality_metric="qualiclip+",
        aesthetic_metric="laion_aes",
        alignment_metric="clipscore",
        quality_weight=1.0,
        aesthetic_weight=1.0,
        alignment_weight=1.0,
        device_mode="auto",
    ):
        entries, context = _collect_image_entries(
            image=image,
            prompt=prompt,
            compare_mode=compare_mode,
            target_width=target_width,
            target_height=target_height,
        )
        metric_specs = [
            ("quality", quality_metric, float(quality_weight)),
            ("aesthetic", aesthetic_metric, float(aesthetic_weight)),
            ("alignment", alignment_metric, float(alignment_weight)),
        ]

        metric_results = {}
        combined = [0.0] * len(entries)
        total_weight = 0.0

        for metric_role, metric_name, weight in metric_specs:
            if metric_name == "disabled" or weight <= 0:
                metric_results[metric_role] = {
                    "metric": metric_name,
                    "weight": weight,
                    "scores": [],
                    "normalized_scores": [],
                    "mean": 0.0,
                }
                continue

            if metric_name in ALIGNMENT_METRICS and context["missing_prompt_count"] > 0:
                metric_results[metric_role] = {
                    "metric": "disabled",
                    "weight": 0.0,
                    "scores": [],
                    "normalized_scores": [],
                    "mean": 0.0,
                }
                if "已自动关闭对齐类指标，本次综合排名不会计算 alignment_metric。" not in context["warnings"]:
                    context["warnings"].append("已自动关闭对齐类指标，本次综合排名不会计算 alignment_metric。")
                continue

            scores, summary = _run_metric_on_entries(metric_name, entries, device_mode)
            normalized_scores = _normalize_scores(
                scores,
                lower_better=bool(METRIC_DEFINITIONS[metric_name]["lower_better"]),
            )
            metric_results[metric_role] = {
                "metric": metric_name,
                "weight": weight,
                "scores": scores,
                "normalized_scores": normalized_scores,
                "mean": float(summary["mean"]),
            }
            total_weight += weight

            for index, value in enumerate(normalized_scores):
                combined[index] += value * weight

        if total_weight <= 0:
            raise ValueError("至少启用一个权重大于 0 的指标。")

        combined = [value / total_weight for value in combined]
        order = sorted(range(len(combined)), key=lambda index: combined[index], reverse=True)
        ranked_entries = [entries[index] for index in order]
        ranked_images = _build_ranked_images_output([entry["image_tensor"] for entry in ranked_entries], compare_mode)
        best_index = int(order[0])
        quality_best_index = _resolve_metric_best_index(metric_results["quality"], best_index)
        aesthetic_best_index = _resolve_metric_best_index(metric_results["aesthetic"], best_index)
        alignment_best_index = _resolve_metric_best_index(metric_results["alignment"], best_index)
        report_rows = []

        for rank, index in enumerate(order, start=1):
            report_rows.append(
                {
                    "rank": rank,
                    "index": int(index),
                    "file_name": entries[index]["file_name"],
                    "image_path": entries[index]["image_path"],
                    "prompt_text": entries[index]["prompt_text"],
                    "prompt_source": entries[index]["prompt_source"],
                    "prompt_path": entries[index]["prompt_path"],
                    "combined_score": float(combined[index]),
                    "quality_score": float(metric_results["quality"]["scores"][index]) if metric_results["quality"]["scores"] else None,
                    "aesthetic_score": float(metric_results["aesthetic"]["scores"][index]) if metric_results["aesthetic"]["scores"] else None,
                    "alignment_score": float(metric_results["alignment"]["scores"][index]) if metric_results["alignment"]["scores"] else None,
                    "original_size": entries[index]["original_size"],
                    "processed_size": entries[index]["processed_size"],
                }
            )

        report = json.dumps(
            {
                "source_mode": context["source_mode"],
                "compare_mode": context["compare_mode"],
                "target_size": context["target_size"],
                "best_index": best_index,
                "best_combined_score": float(combined[best_index]),
                "metric_best_indexes": {
                    "quality": quality_best_index,
                    "aesthetic": aesthetic_best_index,
                    "alignment": alignment_best_index,
                },
                "metrics": metric_results,
                "warnings": context["warnings"],
                "ranking": report_rows,
                "total_weight": total_weight,
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            ranked_images,
            entries[best_index]["image_tensor"],
            entries[quality_best_index]["image_tensor"],
            entries[aesthetic_best_index]["image_tensor"],
            entries[alignment_best_index]["image_tensor"],
            report,
        )


class EvalKitScoreSummary:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "score_a": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "score_b": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "score_c": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "score_d": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "weight_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "weight_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "weight_c": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "weight_d": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "label_a": ("STRING", {"default": "score_a"}),
                "label_b": ("STRING", {"default": "score_b"}),
                "label_c": ("STRING", {"default": "score_c"}),
                "label_d": ("STRING", {"default": "score_d"}),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("weighted_score", "report")
    FUNCTION = "summarize"
    CATEGORY = "evaluation/evalkit"

    def summarize(
        self,
        score_a,
        score_b,
        score_c,
        score_d,
        weight_a,
        weight_b,
        weight_c,
        weight_d,
        label_a="score_a",
        label_b="score_b",
        label_c="score_c",
        label_d="score_d",
    ):
        items = [
            {"label": label_a, "score": float(score_a), "weight": float(weight_a)},
            {"label": label_b, "score": float(score_b), "weight": float(weight_b)},
            {"label": label_c, "score": float(score_c), "weight": float(weight_c)},
            {"label": label_d, "score": float(score_d), "weight": float(weight_d)},
        ]
        enabled_items = [item for item in items if item["weight"] > 0]

        if not enabled_items:
            raise ValueError("至少需要一个权重大于 0。")

        total_weight = sum(item["weight"] for item in enabled_items)
        weighted_score = sum(item["score"] * item["weight"] for item in enabled_items) / total_weight
        report = json.dumps(
            {
                "weighted_score": float(weighted_score),
                "total_weight": float(total_weight),
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )
        return float(weighted_score), report


class EvalKitBatchLoadFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder_path": ("STRING", {"default": "", "multiline": False}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "resize_mode": (BATCH_RESIZE_MODES, {"default": "pad"}),
            },
            "optional": {
                "prompt_folder_path": ("STRING", {"default": "", "multiline": False}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("images", "prompt", "filenames", "count", "report")
    FUNCTION = "load"
    CATEGORY = "evaluation/evalkit"

    def load(
        self,
        image_folder_path,
        target_width,
        target_height,
        resize_mode,
        prompt_folder_path="",
        limit=0,
    ):
        folder_path = (image_folder_path or "").strip()
        if not folder_path:
            raise ValueError("image_folder_path 不能为空。")
        if not os.path.isdir(folder_path):
            raise ValueError(f"图片目录不存在: {folder_path}")

        prompt_path = (prompt_folder_path or "").strip()
        if prompt_path and not os.path.isdir(prompt_path):
            raise ValueError(f"prompt 目录不存在: {prompt_path}")

        image_files = _list_image_files(folder_path)
        if limit > 0:
            image_files = image_files[:limit]
        if not image_files:
            raise ValueError("指定目录下没有找到可加载的图片。")

        image_tensors = []
        prompt_lines = []
        file_names = []
        report_rows = []

        for image_path in image_files:
            with Image.open(image_path) as image_file:
                image_file = ImageOps.exif_transpose(image_file)
                original_width, original_height = image_file.size
                processed_image = _resize_image_for_batch(
                    image_file,
                    target_width=target_width,
                    target_height=target_height,
                    resize_mode=resize_mode,
                )

            prompt_text, source_path, source_type = _load_prompt_text(image_path, prompt_path)
            image_tensors.append(_pil_to_comfy_tensor(processed_image))
            prompt_lines.append(prompt_text)
            file_names.append(os.path.basename(image_path))
            report_rows.append(
                {
                    "file_name": os.path.basename(image_path),
                    "image_path": image_path,
                    "original_size": [int(original_width), int(original_height)],
                    "output_size": [int(target_width), int(target_height)],
                    "prompt_source": source_type,
                    "prompt_path": source_path,
                    "has_prompt": bool(prompt_text.strip()),
                }
            )

        batch_tensor = torch.cat(image_tensors, dim=0)
        report = json.dumps(
            {
                "image_folder_path": folder_path,
                "prompt_folder_path": prompt_path,
                "resize_mode": resize_mode,
                "target_size": [int(target_width), int(target_height)],
                "count": len(file_names),
                "items": report_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        return batch_tensor, json.dumps(prompt_lines, ensure_ascii=False), "\n".join(file_names), len(file_names), report


class EvalKitMetricRankFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder_path": ("STRING", {"default": "", "multiline": False}),
                "metric_name": (ALL_METRICS,),
            },
            "optional": {
                "prompt_folder_path": ("STRING", {"default": "", "multiline": False}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "compare_mode": (COMPARE_MODES, {"default": "separate"}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "sort_mode": (("auto", "higher_better", "lower_better"), {"default": "auto"}),
                "device_mode": (("auto", "cpu"), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT", "STRING", "STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("ranked_images", "best_image", "best_score", "best_filename", "best_prompt", "best_index", "report", "notice")
    FUNCTION = "rank"
    CATEGORY = "evaluation/evalkit"

    def rank(
        self,
        image_folder_path,
        metric_name,
        prompt_folder_path="",
        limit=0,
        compare_mode="separate",
        target_width=1024,
        target_height=1024,
        sort_mode="auto",
        device_mode="auto",
    ):
        compare_mode = _normalize_compare_mode(compare_mode)
        items, folder_path, prompt_path = _collect_folder_items(image_folder_path, prompt_folder_path, limit)
        warnings, notice, missing_prompt_count = _collect_prompt_warnings(items, prompt_path)
        if metric_name in ALIGNMENT_METRICS and missing_prompt_count > 0:
            raise ValueError(
                "当前目录中存在缺少 prompt 的图片，无法执行对齐类单指标排名。"
                " 请提供完整的 prompt_folder_path / 同名 txt / 图片元数据 prompt，或改用非对齐类指标。"
            )
        lower_better = _metric_lower_better(metric_name, sort_mode)
        ranking = []

        for index, item in enumerate(items):
            image_tensor, original_size, processed_size = _load_image_for_compare(
                item["image_path"],
                compare_mode,
                target_width,
                target_height,
            )
            scores, _ = _run_metric(metric_name, image_tensor, item["prompt_text"], device_mode)
            ranking.append(
                {
                    "index": index,
                    "file_name": item["file_name"],
                    "image_path": item["image_path"],
                    "prompt_text": item["prompt_text"],
                    "prompt_source": item["prompt_source"],
                    "prompt_path": item["prompt_path"],
                    "score": float(scores[0]),
                    "original_size": list(original_size),
                    "processed_size": list(processed_size),
                    "_image_tensor": image_tensor,
                }
            )

        ranking.sort(key=lambda row: row["score"], reverse=not lower_better)
        for rank_index, row in enumerate(ranking, start=1):
            row["rank"] = rank_index

        best_row = ranking[0]
        ranked_images = _build_ranked_images_output([row["_image_tensor"] for row in ranking], compare_mode)
        best_image = best_row["_image_tensor"]
        for row in ranking:
            row.pop("_image_tensor", None)
        report = json.dumps(
            {
                "image_folder_path": folder_path,
                "prompt_folder_path": prompt_path,
                "metric": metric_name,
                "compare_mode": compare_mode,
                "target_size": [int(target_width), int(target_height)],
                "sort_mode": sort_mode,
                "lower_better": lower_better,
                "count": len(ranking),
                "warnings": warnings,
                "ranking": ranking,
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            ranked_images,
            best_image,
            float(best_row["score"]),
            best_row["file_name"],
            best_row["prompt_text"],
            int(best_row["index"]),
            report,
            notice,
        )


class EvalKitPresetRankFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder_path": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "prompt_folder_path": ("STRING", {"default": "", "multiline": False}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "compare_mode": (COMPARE_MODES, {"default": "separate"}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "quality_metric": (OPTIONAL_QUALITY_METRICS, {"default": "qualiclip+"}),
                "aesthetic_metric": (OPTIONAL_AESTHETIC_METRICS, {"default": "laion_aes"}),
                "alignment_metric": (OPTIONAL_ALIGNMENT_METRICS, {"default": "clipscore"}),
                "quality_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "aesthetic_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "alignment_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
                "device_mode": (("auto", "cpu"), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = (
        "ranked_images",
        "best_image",
        "best_filename",
        "best_prompt",
        "quality_image",
        "aesthetic_image",
        "alignment_image",
        "report",
        "notice",
    )
    FUNCTION = "rank"
    CATEGORY = "evaluation/evalkit"

    def rank(
        self,
        image_folder_path,
        prompt_folder_path="",
        limit=0,
        compare_mode="separate",
        target_width=1024,
        target_height=1024,
        quality_metric="qualiclip+",
        aesthetic_metric="laion_aes",
        alignment_metric="clipscore",
        quality_weight=1.0,
        aesthetic_weight=1.0,
        alignment_weight=1.0,
        device_mode="auto",
    ):
        compare_mode = _normalize_compare_mode(compare_mode)
        items, folder_path, prompt_path = _collect_folder_items(image_folder_path, prompt_folder_path, limit)
        warnings, notice, missing_prompt_count = _collect_prompt_warnings(items, prompt_path)
        effective_alignment_metric = alignment_metric
        effective_alignment_weight = float(alignment_weight)
        if effective_alignment_metric != "disabled" and effective_alignment_weight > 0 and missing_prompt_count > 0:
            raise ValueError(
                "当前目录中存在缺少 prompt 的图片，无法稳定输出 best_image 与 alignment_image。"
                " 请关闭 alignment_metric，或提供完整的 prompt_folder_path / 同名 txt / 图片元数据 prompt。"
            )
        metric_specs = [
            ("quality", quality_metric, float(quality_weight)),
            ("aesthetic", aesthetic_metric, float(aesthetic_weight)),
            ("alignment", effective_alignment_metric, effective_alignment_weight),
        ]

        metric_results = {}
        combined = [0.0] * len(items)
        total_weight = 0.0
        processed_entries = []

        for item in items:
            image_tensor, original_size, processed_size = _load_image_for_compare(
                item["image_path"],
                compare_mode,
                target_width,
                target_height,
            )
            processed_entries.append(
                {
                    **item,
                    "image_tensor": image_tensor,
                    "original_size": list(original_size),
                    "processed_size": list(processed_size),
                }
            )

        for metric_role, metric_name, weight in metric_specs:
            if metric_name == "disabled" or weight <= 0:
                metric_results[metric_role] = {"metric": metric_name, "weight": weight, "scores": [], "mean": 0.0}
                continue

            raw_scores = []
            for entry in processed_entries:
                scores, _ = _run_metric(metric_name, entry["image_tensor"], entry["prompt_text"], device_mode)
                raw_scores.append(float(scores[0]))

            normalized_scores = _normalize_scores(
                raw_scores,
                lower_better=bool(METRIC_DEFINITIONS[metric_name]["lower_better"]),
            )
            metric_results[metric_role] = {
                "metric": metric_name,
                "weight": weight,
                "scores": raw_scores,
                "normalized_scores": normalized_scores,
                "mean": float(sum(raw_scores) / len(raw_scores)) if raw_scores else 0.0,
            }
            total_weight += weight
            for index, value in enumerate(normalized_scores):
                combined[index] += value * weight

        if total_weight <= 0:
            raise ValueError("至少启用一个权重大于 0 的指标。")

        combined = [value / total_weight for value in combined]
        ranking = []
        for index, entry in enumerate(processed_entries):
            ranking.append(
                {
                    "index": index,
                    "file_name": entry["file_name"],
                    "image_path": entry["image_path"],
                    "prompt_text": entry["prompt_text"],
                    "prompt_source": entry["prompt_source"],
                    "prompt_path": entry["prompt_path"],
                    "combined_score": float(combined[index]),
                    "quality_score": float(metric_results["quality"]["scores"][index]) if metric_results["quality"]["scores"] else None,
                    "aesthetic_score": float(metric_results["aesthetic"]["scores"][index]) if metric_results["aesthetic"]["scores"] else None,
                    "alignment_score": float(metric_results["alignment"]["scores"][index]) if metric_results["alignment"]["scores"] else None,
                    "original_size": entry["original_size"],
                    "processed_size": entry["processed_size"],
                    "_image_tensor": entry["image_tensor"],
                }
            )

        ranking.sort(key=lambda row: row["combined_score"], reverse=True)
        for rank_index, row in enumerate(ranking, start=1):
            row["rank"] = rank_index

        best_row = ranking[0]
        best_index = int(best_row["index"])
        quality_best_index = _resolve_metric_best_index(metric_results["quality"], best_index)
        aesthetic_best_index = _resolve_metric_best_index(metric_results["aesthetic"], best_index)
        alignment_best_index = _resolve_metric_best_index(metric_results["alignment"], best_index)
        ranked_images = _build_ranked_images_output([row["_image_tensor"] for row in ranking], compare_mode)
        best_image = best_row["_image_tensor"]
        quality_image = processed_entries[quality_best_index]["image_tensor"]
        aesthetic_image = processed_entries[aesthetic_best_index]["image_tensor"]
        alignment_image = processed_entries[alignment_best_index]["image_tensor"]
        for row in ranking:
            row.pop("_image_tensor", None)

        report = json.dumps(
            {
                "image_folder_path": folder_path,
                "prompt_folder_path": prompt_path,
                "compare_mode": compare_mode,
                "target_size": [int(target_width), int(target_height)],
                "best_index": best_index,
                "best_combined_score": float(best_row["combined_score"]),
                "metric_best_indexes": {
                    "quality": quality_best_index,
                    "aesthetic": aesthetic_best_index,
                    "alignment": alignment_best_index,
                },
                "metrics": metric_results,
                "count": len(ranking),
                "warnings": warnings,
                "ranking": ranking,
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            ranked_images,
            best_image,
            best_row["file_name"],
            best_row["prompt_text"],
            quality_image,
            aesthetic_image,
            alignment_image,
            report,
            notice,
        )


class EvalKitRankingPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report": ("STRING", {"default": "", "multiline": True}),
                "select_mode": (("top", "bottom"), {"default": "top"}),
                "count": ("INT", {"default": 9, "min": 1, "max": 512, "step": 1}),
                "thumbnail_width": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 8}),
                "thumbnail_height": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 8}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("preview_image", "summary", "filenames", "count")
    FUNCTION = "preview"
    CATEGORY = "evaluation/evalkit"

    def preview(self, report, select_mode, count, thumbnail_width, thumbnail_height, columns):
        payload, _, selected = _select_ranking_rows(report, select_mode, count)
        preview_image = _build_preview_grid(selected, thumbnail_width, thumbnail_height, columns)
        warnings = payload.get("warnings", [])
        summary_lines = []
        if isinstance(warnings, list):
            for warning in warnings:
                if isinstance(warning, str) and warning.strip():
                    summary_lines.append(f"warning: {warning.strip()}")

        for row in selected:
            score_value, score_label = _get_ranking_score_value(row)
            summary_lines.append(
                f"rank={row.get('rank', '')} file={row.get('file_name', '')} {score_label}={score_value:.6f}"
            )

        return (
            preview_image,
            "\n".join(summary_lines),
            "\n".join(str(row.get("file_name", "")) for row in selected),
            len(selected),
        )


class EvalKitRankingExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "report": ("STRING", {"default": "", "multiline": True}),
                "output_folder_path": ("STRING", {"default": "", "multiline": False}),
                "select_mode": (("top", "bottom"), {"default": "top"}),
                "count": ("INT", {"default": 10, "min": 1, "max": 100000, "step": 1}),
                "filename_mode": (("rank_prefix", "original_name"), {"default": "rank_prefix"}),
            },
            "optional": {
                "export_prompt_txt": (("enabled", "disabled"), {"default": "enabled"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("output_folder_path", "exported_count", "summary")
    FUNCTION = "export"
    CATEGORY = "evaluation/evalkit"

    def export(self, report, output_folder_path, select_mode, count, filename_mode, export_prompt_txt="enabled"):
        destination = (output_folder_path or "").strip()
        if not destination:
            raise ValueError("output_folder_path 不能为空。")

        payload, _, selected = _select_ranking_rows(report, select_mode, count)
        os.makedirs(destination, exist_ok=True)
        exported_files = []

        for row in selected:
            source_path = row["image_path"]
            original_name = os.path.basename(source_path)
            if filename_mode == "rank_prefix":
                target_name = f"{int(row.get('rank', 0)):03d}_{original_name}"
            else:
                target_name = original_name

            target_path = os.path.join(destination, target_name)
            shutil.copy2(source_path, target_path)
            exported_files.append(target_name)

            prompt_text = str(row.get("prompt_text", "") or "").strip()
            if export_prompt_txt == "enabled" and prompt_text:
                txt_path = os.path.splitext(target_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as file:
                    file.write(prompt_text)

        summary_payload = {
            "output_folder_path": destination,
            "select_mode": select_mode,
            "count": len(exported_files),
            "warnings": payload.get("warnings", []),
            "files": exported_files,
        }
        return destination, len(exported_files), json.dumps(summary_payload, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "EvalKitBatchLoadFromPath": EvalKitBatchLoadFromPath,
    "EvalKitMetricScore": EvalKitMetricScore,
    "EvalKitMetricRank": EvalKitMetricRank,
    "EvalKitMetricRankFromPath": EvalKitMetricRankFromPath,
    "EvalKitPresetRank": EvalKitPresetRank,
    "EvalKitPresetRankFromPath": EvalKitPresetRankFromPath,
    "EvalKitRankingPreview": EvalKitRankingPreview,
    "EvalKitRankingExport": EvalKitRankingExport,
    "EvalKitScoreSummary": EvalKitScoreSummary,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EvalKitBatchLoadFromPath": "EvalKit Batch Load From Path",
    "EvalKitMetricScore": "EvalKit Metric Score",
    "EvalKitMetricRank": "EvalKit Metric Rank",
    "EvalKitMetricRankFromPath": "EvalKit Metric Rank From Path",
    "EvalKitPresetRank": "EvalKit Preset Rank",
    "EvalKitPresetRankFromPath": "EvalKit Preset Rank From Path",
    "EvalKitRankingPreview": "EvalKit Ranking Preview",
    "EvalKitRankingExport": "EvalKit Ranking Export",
    "EvalKitScoreSummary": "EvalKit Score Summary",
}
