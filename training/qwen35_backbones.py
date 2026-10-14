"""Qwen3.5 Seg2Click decoder bindings and 4B/9B restart safeguards.

The wrapper, decoder serializer and manifest schema must stay paired so that
all existing checkpoints remain loadable. Model computations live in seg2click.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from training.common import _is_main_process

MODEL_SIZES = ("0.8b", "2b", "4b", "9b")
DEFAULT_EXPERIMENT_NAME = "seg2click_qwen35"


@dataclass(frozen=True)
class DecoderBinding:
    wrapper_type: type
    save_decoder: Callable
    assert_fp32: Callable
    configure_trainable: Callable
    schema: int = 2
    scale_key: str = "pixel_shuffle_factor"


def decoder_binding(model_size: str = "0.8b") -> DecoderBinding:
    """Pair the Seg2Click wrapper with the decoder serializer of one backbone size."""
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown Qwen3.5 model size: {model_size}")
    from seg2click import fusion
    from seg2click.mask_model import assert_phase1_decoder_fp32, configure_phase1_trainable_parameters

    wrapper, save = {
        "0.8b": (fusion.VisionConcatMlpPhase1Qwen35MaskWrapper, fusion.save_vision_concat_mlp_phase1_decoder),
        "2b": (fusion.VisionConcatMlpPhase1Qwen35MaskWrapper, fusion.save_vision_concat_mlp_phase1_decoder),
        "4b": (fusion.Qwen35FourBVisionConcatMlpMaskWrapper, fusion.save_qwen35_4b_vision_concat_mlp_decoder),
        "9b": (fusion.Qwen35NineBVisionConcatMlpMaskWrapper, fusion.save_qwen35_9b_vision_concat_mlp_decoder),
    }[model_size]
    return DecoderBinding(wrapper, save, assert_phase1_decoder_fp32, configure_phase1_trainable_parameters)


def model_metadata(model_size: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if model_size in {"4b", "9b"}:
        hidden = {"4b": 2560, "9b": 4096}[model_size]
        metadata.update(hidden_size=hidden, attention_implementation="sdpa",
                        vision_patch_size=16, vision_spatial_merge_size=2,
                        vision_output_hidden_size=hidden)
    metadata["vision_encoder_features"] = "LLM final image-token state plus Qwen3.5 merged vision tower pooler_output"
    metadata["vision_fusion"] = "residual concat MLP with zero-initialized output layer"
    return metadata


def validate_backbone(model: Any, model_size: str) -> None:
    if model_size == "0.8b":
        return
    if model_size == "2b":
        assert model.config.text_config.hidden_size == 2048, model.config
        assert model.config.vision_config.patch_size == 16, model.config
        print("Q35-2B loaded: hidden_size=2048; attention=", model.config._attn_implementation, flush=True)
        return
    hidden = {"4b": 2560, "9b": 4096}[model_size]
    label = f"Qwen3.5-{model_size.upper()}"
    if model.config.model_type != "qwen3_5":
        raise ValueError(f"Expected Qwen3.5, got {model.config.model_type!r}")
    if int(model.config.text_config.hidden_size) != hidden:
        raise ValueError(f"{label} text hidden size must be {hidden}, got {model.config.text_config.hidden_size}")
    vision_config = model.config.vision_config
    expected_vision = {"out_hidden_size": hidden, "patch_size": 16, "spatial_merge_size": 2}
    observed_vision = {key: int(getattr(vision_config, key)) for key in expected_vision}
    if observed_vision != expected_vision:
        raise ValueError(f"Unexpected {label} vision configuration: {observed_vision}")
    if model.config._attn_implementation != "sdpa":
        raise ValueError(f"The {label} scaling experiment requires SDPA, got {model.config._attn_implementation!r}")
    print(f"Q35-{model_size.upper()} loaded: hidden_size={hidden}; vision_out={hidden}; attention=sdpa", flush=True)


def _validate_restart_arguments(previous: Mapping[str, Any], current: Mapping[str, Any], model_size: str) -> None:
    """Require identical settings when retrying in the experiment output."""
    changes = {
        key: {"previous": previous.get(key), "current": current.get(key)}
        for key in previous.keys() | current.keys()
        if previous.get(key) != current.get(key)
    }
    if changes:
        raise ValueError(f"Retry changes Qwen3.5-{model_size.upper()} training arguments: {changes}")


def _prepare_output_dir(output_dir: Path, resume: Optional[str], *, model_size: str, experiment_name: str) -> Optional[str]:
    """Preserve one output root across startup failures and checkpoint resumes."""
    resume = (resume or "").strip()
    resume_path = Path(resume).resolve() if resume else None
    if resume_path is not None:
        if not resume_path.is_dir():
            raise FileNotFoundError(f"--resume_from_checkpoint is not a directory: {resume_path}")
        if resume_path.parent != output_dir.resolve() or not resume_path.name.startswith("checkpoint-"):
            raise ValueError(
                f"Resume must name an explicit checkpoint inside the Qwen3.5-{model_size.upper()} output directory"
            )
    if not _is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(resume_path) if resume_path is not None else None
    if (output_dir / "completion_receipt.json").exists():
        raise RuntimeError(f"Qwen3.5-{model_size.upper()} experiment already completed: {output_dir}")
    checkpoints = sorted(output_dir.glob("checkpoint-*")) if output_dir.exists() else []
    if checkpoints and resume_path is None:
        raise RuntimeError(
            f"Qwen3.5-{model_size.upper()} output already contains checkpoints. Retry in this same output directory "
            "with --resume_from_checkpoint set explicitly: "
            + ", ".join(str(path) for path in checkpoints)
        )
    if output_dir.exists() and resume_path is None:
        startup_names = {
            "train.log", "run_metadata.json", "triton_cache", "tilelang_cache",
            "tilelang_tmp", "torchinductor_cache", "pycache",
        }
        material = [path for path in output_dir.iterdir() if path.name not in startup_names]
        if material:
            raise RuntimeError(
                f"Qwen3.5-{model_size.upper()} output contains artifacts that cannot be overwritten: {material}. "
                "Keep this output path and resume an explicit checkpoint when available."
            )
        prior_metadata = output_dir / "run_metadata.json"
        if prior_metadata.is_file():
            prior = json.loads(prior_metadata.read_text(encoding="utf-8"))
            if prior.get("experiment") != experiment_name:
                raise ValueError(f"Startup metadata belongs to another experiment: {prior_metadata}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(resume_path) if resume_path is not None else None


def write_restart_metadata(metadata: dict[str, Any], output_dir: Path, resume_checkpoint: Optional[str], model_size: str) -> None:
    from seg2click.runtime import write_json_atomic

    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in (
            "experiment", "backbone", "seed", "initialization", "data",
            "token_ids", "objective", "model",
        ):
            if previous.get(key) != metadata[key]:
                raise ValueError(
                    f"Retry changes the existing Qwen3.5-{model_size.upper()} experiment's {key}"
                )
        runtime_only_fields = {"logging_dir", "local_rank"}
        previous_args = {
            key: value
            for key, value in previous.get("training", {}).get("arguments", {}).items()
            if key not in runtime_only_fields
        }
        current_args = {
            key: value
            for key, value in metadata["training"]["arguments"].items()
            if key not in runtime_only_fields
        }
        _validate_restart_arguments(previous_args, current_args, model_size)
        metadata["previous_attempts"] = [
            *previous.get("previous_attempts", []),
            {
                "created_at_utc": previous.get("created_at_utc"),
                "resume_from_checkpoint": previous.get("resume_from_checkpoint"),
                "status": previous.get("status"),
            },
        ]
    metadata["resume_from_checkpoint"] = resume_checkpoint
    write_json_atomic(metadata, metadata_path)
