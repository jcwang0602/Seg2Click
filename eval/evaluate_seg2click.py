"""Evaluate a Seg2Click checkpoint on ScreenSpot, ScreenSpot-v2 or ScreenSpot-Pro.

Each invocation evaluates exactly one benchmark; run the three benchmarks as
three independent processes (see ``scripts/eval/evaluate_seg2click.sh``).

A prediction is a hit when the decoded click lies inside the ground-truth
bbox (``Hit@1``).  Outputs written to ``OUTPUT_DIR/<benchmark>/``:

* ``predictions.json`` -- one record per example (point, hit, decode reason)
* ``metrics.json``     -- overall and per-category Hit@1
"""

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from eval.benchmark_data import (
    EXPECTED_COUNTS,
    _atomic_json,
    image_and_metadata_schema,
    load_examples_lazy,
)
from seg2click.predictor import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, Seg2ClickPredictor


def _categories(benchmark: str, row) -> dict[str, str]:
    if benchmark == "screenspot_pro":
        return {"group": str(row["group"]), "ui_type": str(row["ui_type"])}
    return {"data_source": str(row["data_source"]), "data_type": str(row["data_type"])}


def _breakdown(predictions: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    tables: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for record in predictions:
        for key, value in record["categories"].items():
            tables[key][value].append(record["hit"])
    return {
        key: {value: {"hit_at_1": sum(hits) / len(hits), "num_examples": len(hits)}
              for value, hits in sorted(groups.items())}
        for key, groups in tables.items()
    }


def evaluate(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir).resolve() / args.benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    rows = load_examples_lazy(args.benchmark, data_root)
    limit = len(rows) if args.max_samples <= 0 else min(args.max_samples, len(rows))

    predictor = Seg2ClickPredictor.from_pretrained(
        args.checkpoint, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
    )
    predictions: list[dict] = []
    started = time.time()
    for start in tqdm(range(0, limit, args.batch_size), desc=args.benchmark):
        indices = range(start, min(start + args.batch_size, limit))
        batch = [(index, rows[index]) for index in indices]
        loaded = [image_and_metadata_schema(args.benchmark, row, data_root) for _, row in batch]
        results = predictor.predict_batch([image for image, _ in loaded], [meta["instruction"] for _, meta in loaded])
        for (index, row), (image, meta), result in zip(batch, loaded, results):
            x0, y0, x1, y1 = meta["bbox"]
            x, y = result.point
            predictions.append({
                "index": index,
                "sample_key": meta["sample_key"],
                "instruction": meta["instruction"],
                "predicted_point": [x, y],
                "bbox": meta["bbox"],
                "hit": bool(x0 <= x <= x1 and y0 <= y <= y1),
                "decode_reason": result.decode_reason,
                "categories": _categories(args.benchmark, row),
            })
            image.close()

    hits = sum(record["hit"] for record in predictions)
    metrics = {
        "benchmark": args.benchmark,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_examples": len(predictions),
        "expected_examples": EXPECTED_COUNTS[args.benchmark],
        "complete": len(predictions) == EXPECTED_COUNTS[args.benchmark],
        "num_hits": hits,
        "hit_at_1": hits / len(predictions) if predictions else 0.0,
        "breakdown": _breakdown(predictions),
        "elapsed_seconds": time.time() - started,
        "batch_size": args.batch_size,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
    }
    _atomic_json(predictions, output_dir / "predictions.json")
    _atomic_json(metrics, output_dir / "metrics.json")
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Seg2Click checkpoint directory")
    parser.add_argument("--benchmark", choices=tuple(EXPECTED_COUNTS), required=True)
    parser.add_argument("--data_root", default="data/eval")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the full benchmark")
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--attn_implementation", default=None,
                        help="Default: flash_attention_2 for Qwen2/2.5-VL when installed, otherwise sdpa")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    return args


if __name__ == "__main__":
    result = evaluate(parse_args())
    summary = {key: result[key] for key in ("benchmark", "num_examples", "num_hits", "hit_at_1")}
    print(json.dumps(summary, indent=2), flush=True)
