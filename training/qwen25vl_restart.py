"""Qwen2.5-VL output-directory protection for safe retries (no model computation)."""

from __future__ import annotations

import json
from pathlib import Path

from training.common import _is_main_process

def prepare_output_dir(output_dir: Path, resume: str | None, *,
                       experiment_name: str = "seg2click_qwen25vl_3b") -> str | None:
    """Keep failed startup retries in place; never silently replace trained artifacts."""
    resume = (resume or "").strip()
    resume_path = Path(resume).resolve() if resume else None
    if resume_path is not None:
        if not resume_path.is_dir():
            raise FileNotFoundError(f"--resume_from_checkpoint is not a directory: {resume_path}")
        if resume_path.parent != output_dir.resolve() or not resume_path.name.startswith("checkpoint-"):
            raise ValueError("Resume must name an explicit checkpoint inside this experiment's output directory")
    # Only rank 0 owns collision checks. Other ranks may arrive after rank 0
    # has written metadata and must not confuse that with a previous run.
    if not _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(resume_path) if resume_path is not None else None
    if (output_dir / "completion_receipt.json").exists():
        raise RuntimeError(f"Seg2Click experiment already has a completion receipt: {output_dir}")
    checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    if checkpoints and resume_path is None:
        raise RuntimeError(
            "Seg2Click output already contains checkpoints. Retry in this same output directory "
            "with --resume_from_checkpoint set explicitly: " + ", ".join(str(path) for path in checkpoints)
        )
    if output_dir.exists() and resume_path is None:
        startup_names = {
            "train.log", "run_metadata.json", "triton_cache", "tilelang_cache",
            "tilelang_tmp", "torchinductor_cache", "pycache",
        }
        material = [path for path in output_dir.iterdir() if path.name not in startup_names]
        if material:
            raise RuntimeError(
                f"Seg2Click output contains artifacts that cannot be overwritten: {material}. "
                "Keep this output path and inspect the interrupted run; resume an explicit checkpoint when available."
            )
        prior_metadata = output_dir / "run_metadata.json"
        if prior_metadata.is_file():
            prior = json.loads(prior_metadata.read_text(encoding="utf-8"))
            if prior.get("experiment") != experiment_name:
                raise ValueError(f"Startup metadata belongs to another experiment: {prior_metadata}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(resume_path) if resume_path is not None else None


def record_previous_attempt(previous: dict, metadata: dict) -> None:
    """Refuse retries that change the experiment; keep the earlier attempt history."""
    for key in ("experiment", "seed", "initialization", "data", "token_ids", "objective", "model"):
        if previous.get(key) != metadata[key]:
            raise ValueError(f"Retry changes the existing experiment's {key}; refusing to overwrite metadata")
    runtime_only_fields = {"logging_dir", "local_rank"}
    previous_args = {key: value for key, value in previous.get("training", {}).get("arguments", {}).items()
                     if key not in runtime_only_fields}
    current_args = {key: value for key, value in metadata["training"]["arguments"].items()
                    if key not in runtime_only_fields}
    changes = {key: {"previous": previous_args.get(key), "current": current_args.get(key)}
               for key in previous_args.keys() | current_args.keys()
               if previous_args.get(key) != current_args.get(key)}
    if changes:
        raise ValueError(f"Retry changes training arguments: {changes}")
    attempt = {"created_at_utc": previous.get("created_at_utc"),
               "resume_from_checkpoint": previous.get("resume_from_checkpoint"),
               "status": previous.get("status")}
    metadata["previous_attempts"] = [*previous.get("previous_attempts", []), attempt]
