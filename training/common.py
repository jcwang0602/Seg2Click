"""Shared bookkeeping used by the independent mask-training entry points.

Model construction, CLI defaults, losses, and experiment-specific restart rules
remain in their entry points. The helpers here preserve the original behavior.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import transformers



def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _is_main_process() -> bool:
    return _rank() == 0


def _distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP/Accelerate wrappers while retaining the Phase 1 wrapper."""
    current = model
    seen = set()
    while hasattr(current, "module") and id(current) not in seen:
        seen.add(id(current))
        current = current.module
    return current


def _trainer_model(trainer: transformers.Trainer) -> torch.nn.Module:
    model = trainer.model
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None:
        try:
            model = accelerator.unwrap_model(model, keep_torch_compile=False)
        except TypeError:
            model = accelerator.unwrap_model(model)
    return _unwrap(model)


def _set_use_cache(model: torch.nn.Module, value: bool) -> None:
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = bool(value)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = bool(value)
    base = getattr(model, "base_model", None)
    if base is not None and base is not model:
        _set_use_cache(base, value)


def _serialise_training_args(args: transformers.TrainingArguments) -> dict[str, Any]:
    try:
        value = args.to_dict()
    except Exception:
        value = vars(args).copy()
    return json.loads(json.dumps(value, default=str))


def _assert_finite_phase1_state(trainer: transformers.Trainer) -> None:
    model = _trainer_model(trainer)
    bad_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and not torch.isfinite(parameter).all()
    ]
    bad_optimizer = []
    optimizer = trainer.optimizer
    if optimizer is not None:
        optimizer = getattr(optimizer, "optimizer", optimizer)
        for index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                    bad_optimizer.append(f"{index}:{name}")
    if bad_parameters or bad_optimizer:
        raise FloatingPointError(f"Non-finite Phase 1 state: parameters={bad_parameters}, optimizer={bad_optimizer}")


def _assert_finite_seg2click_state(trainer: transformers.Trainer) -> None:
    model = _trainer_model(trainer)
    bad_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and not torch.isfinite(parameter).all()
    ]
    bad_optimizer = []
    optimizer = trainer.optimizer
    if optimizer is not None:
        optimizer = getattr(optimizer, "optimizer", optimizer)
        for index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                    bad_optimizer.append(f"{index}:{name}")
    if bad_parameters or bad_optimizer:
        raise FloatingPointError(f"Non-finite Seg2Click state: parameters={bad_parameters}, optimizer={bad_optimizer}")


def _require_phase1_decoder_artifact(output_dir: Path) -> tuple[Path, Path]:
    """Validate the decoder manifest/weights written for evaluation."""

    manifest_path = output_dir / "phase1_mask_decoder.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Phase 1 decoder manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Phase 1 decoder manifest: {manifest_path}") from exc
    weights_name = manifest.get("weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise ValueError(f"Phase 1 decoder manifest has no weights entry: {manifest_path}")
    weights_path = (output_dir / weights_name).resolve()
    if weights_path.parent != output_dir.resolve():
        raise ValueError(f"Phase 1 decoder weights must be directly inside {output_dir}: {weights_name!r}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing Phase 1 decoder weights: {weights_path}")
    return manifest_path, weights_path


def _require_seg2click_decoder_artifact(output_dir: Path) -> tuple[Path, Path]:
    """Validate the decoder manifest/weights written for evaluation."""

    manifest_path = output_dir / "phase1_mask_decoder.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Seg2Click decoder manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Seg2Click decoder manifest: {manifest_path}") from exc
    weights_name = manifest.get("weights")
    if not isinstance(weights_name, str) or not weights_name:
        raise ValueError(f"Seg2Click decoder manifest has no weights entry: {manifest_path}")
    weights_path = (output_dir / weights_name).resolve()
    if weights_path.parent != output_dir.resolve():
        raise ValueError(f"Seg2Click decoder weights must be directly inside {output_dir}: {weights_name!r}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing Seg2Click decoder weights: {weights_path}")
    return manifest_path, weights_path


def _prepare_phase1_output_dir(output_dir: Path, resume: Optional[str]) -> Optional[str]:
    """Fail closed on accidental reuse; resume must be explicitly requested."""
    resume = (resume or "").strip()
    # ``torchrun`` starts all ranks together.  Only rank 0 owns the collision
    # check; otherwise a rank that arrives a few milliseconds after rank 0 has
    # written metadata would mistake a valid fresh launch for an accidental
    # reuse.  All ranks still receive the explicitly requested resume path.
    if not _is_main_process():
        if resume:
            resume_path = Path(resume).resolve()
            if not resume_path.is_dir():
                raise FileNotFoundError(f"--resume_from_checkpoint is not a directory: {resume_path}")
            return str(resume_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        return None
    checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    if checkpoints and not resume:
        raise RuntimeError(
            "Phase 1 output already contains checkpoints. Set --resume_from_checkpoint to an "
            "explicit checkpoint path instead of silently restarting: " + ", ".join(str(path) for path in checkpoints)
        )
    if output_dir.exists() and not resume:
        transient_names = {
            "train.log",
            "triton_cache",
            "tilelang_cache",
            "tilelang_tmp",
        }
        material = [path for path in output_dir.iterdir() if path.name not in transient_names]
        if material:
            raise RuntimeError(
                f"Phase 1 output directory is non-empty: {output_dir}. "
                "Choose a new --output_dir or set --resume_from_checkpoint explicitly."
            )
    if resume:
        resume_path = Path(resume).resolve()
        if not resume_path.is_dir():
            raise FileNotFoundError(f"--resume_from_checkpoint is not a directory: {resume_path}")
        return str(resume_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return None
