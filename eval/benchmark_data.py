"""ScreenSpot / ScreenSpot-v2 / ScreenSpot-Pro loading.

Expected local layout (see README)::

    data/eval/ScreenSpot/data/*.parquet
    data/eval/ScreenSpot_v2/data/*.parquet
    data/eval/ScreenSpot-Pro/annotations/*.json + images/

All returned bboxes are ``[x0, y0, x1, y1]`` normalized to the original
screenshot.  ScreenSpot and ScreenSpot-v2 already store normalized boxes;
ScreenSpot-Pro stores pixel boxes and is normalized by the image size.
"""
from __future__ import annotations

import json
import math
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

EXPECTED_COUNTS = {"screenspot": 1272, "screenspot_v2": 1272, "screenspot_pro": 1581}


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_examples_lazy(benchmark: str, data_root: Path) -> Sequence[Mapping[str, Any]]:
    """Return the benchmark rows; ScreenSpot images are decoded on access."""

    if benchmark in ("screenspot", "screenspot_v2"):
        from datasets import load_dataset

        folder = data_root / ("ScreenSpot" if benchmark == "screenspot" else "ScreenSpot_v2")
        parquet_files = sorted((folder / "data").glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {folder / 'data'}")
        rows = load_dataset("parquet", data_files={"test": [str(p) for p in parquet_files]}, split="test")
    elif benchmark == "screenspot_pro":
        annotation_dir = data_root / "ScreenSpot-Pro" / "annotations"
        files = sorted(path for path in annotation_dir.glob("*.json") if path.name != "all.json")
        if not files:
            raise FileNotFoundError(f"No ScreenSpot-Pro annotations found under {annotation_dir}")
        rows = [dict(item) for path in files for item in json.loads(path.read_text(encoding="utf-8"))]
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if len(rows) != EXPECTED_COUNTS[benchmark]:
        raise ValueError(f"{benchmark} contains {len(rows)} samples; expected {EXPECTED_COUNTS[benchmark]}")
    return rows


def normalized_bbox_schema(benchmark: str, bbox, width: int, height: int) -> list[float]:
    """Normalize a released bbox without clipping its boundaries."""

    values = [float(value) for value in bbox]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid {benchmark} bbox: {bbox!r}")
    if benchmark == "screenspot_pro":
        values = [values[0] / width, values[1] / height, values[2] / width, values[3] / height]
    # ScreenSpot-v2 index 783 has x_max slightly above 1; keep it as released.
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f"Invalid ordered {benchmark} bbox: {values!r}")
    return values


def _open_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, Mapping) and value.get("bytes") is not None:
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, Mapping) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    return Image.open(str(value)).convert("RGB")


def image_and_metadata_schema(benchmark: str, row: Mapping[str, Any], data_root: Path):
    """Return ``(RGB image, {sample_key, instruction, bbox, image_size})``."""

    if benchmark in ("screenspot", "screenspot_v2"):
        image = _open_image(row["image"])
        key = f"{row['file_name']}\t{row['instruction']}"
    else:
        image = Image.open(data_root / "ScreenSpot-Pro" / "images" / str(row["img_filename"])).convert("RGB")
        key = str(row["id"])
        declared = tuple(float(v) for v in row.get("img_size", image.size))
        if declared != tuple(float(v) for v in image.size):
            image.close()
            raise ValueError(f"Declared/original image size mismatch for {row['img_filename']}")
    width, height = image.size
    return image, {
        "sample_key": key,
        "instruction": str(row["instruction"]),
        "bbox": normalized_bbox_schema(benchmark, row["bbox"], width, height),
        "image_size": [width, height],
    }
