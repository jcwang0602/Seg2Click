"""Seg2Click training on Qwen3.5 (0.8B / 2B / 4B / 9B) with LoRA.

Choose the backbone with --model_size; batch size, accumulation, learning
rate and decoder capacity are normal CLI options.
scripts/train/qwen35_seg2click.sh sets the defaults used in the paper.
"""

from __future__ import annotations

# Preserve direct-file execution after moving this entry point under training/.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _EntryPath
    sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from seg2click.runtime import (
    PHASE1_DATA_CONFIG,
    PHASE1_EXPECTED_RECORDS,
    PHASE1_TOTAL_RECORDS,
    Phase1IndexedDataset,
    Phase1TrainingCollator,
    install_qwen35_compatibility,
    make_data_args,
    prepare_phase1_tokenizer_and_model,
    sha256_file,
    write_json_atomic,
)

DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 24576
DEFAULT_BATCH_PER_DEVICE = 4
DEFAULT_GRAD_ACCUMULATION = 1
DEFAULT_EPOCHS = 1
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SAVE_STEPS = 2000
DEFAULT_DECODER_CHANNELS = 1024
DEFAULT_PIXEL_SHUFFLE_FACTOR = 32
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 5720064
DEFAULT_MAX_CONV_TURNS = 10
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05


from training.common import (
    _str_to_bool,
    _is_main_process,
    _distributed_barrier,
    _trainer_model,
    _set_use_cache,
    _assert_finite_phase1_state as _assert_finite_state,
    _serialise_training_args,
    _require_phase1_decoder_artifact as _require_decoder_artifact,
    _prepare_phase1_output_dir as _prepare_output_dir,
)
from training.mask_trainers import Phase1MaskTrainer as _Phase1MaskTrainer
from training.qwen35_backbones import (
    MODEL_SIZES, DEFAULT_EXPERIMENT_NAME,
    decoder_binding, model_metadata, validate_backbone, write_restart_metadata,
    _prepare_output_dir as _prepare_scaling_output_dir,
)


def _parse_args(argv=None) -> argparse.Namespace:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--model_size", choices=MODEL_SIZES, type=str.lower, default="0.8b")
    selection, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[selector])
    parser.add_argument("--experiment_name", default=f"{DEFAULT_EXPERIMENT_NAME}_{selection.model_size}",
        help="Experiment identifier stored in run_metadata.json; does not affect computation.")
    parser.add_argument("--model_name_or_path", default=f"checkpoints/Qwen3.5-{selection.model_size.upper()}")
    parser.add_argument("--data_path", default=PHASE1_DATA_CONFIG)
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_train_epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_BATCH_PER_DEVICE if selection.model_size == "0.8b" else 1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUMULATION if selection.model_size == "0.8b" else 4)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_conv_turns", type=int, default=DEFAULT_MAX_CONV_TURNS)
    parser.add_argument("--decoder_channels", type=int, default=DEFAULT_DECODER_CHANNELS)
    parser.add_argument(
        "--pixel_shuffle_factor",
        type=int,
        default=DEFAULT_PIXEL_SHUFFLE_FACTOR,
        help="Qwen3.5 post-merge to processed-image expansion factor (16*2=32).",
    )
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--bf16", type=_str_to_bool, default=True)
    parser.add_argument("--tf32", type=_str_to_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=_str_to_bool, default=selection.model_size == "9b")
    config = parser.parse_args(argv)
    positive_integer_fields = (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "save_steps",
        "save_total_limit",
        "logging_steps",
        "dataloader_num_workers",
        "model_max_length",
        "min_pixels",
        "max_pixels",
        "max_conv_turns",
        "decoder_channels",
        "pixel_shuffle_factor",
        "lora_r",
        "lora_alpha",
    )
    invalid = [name for name in positive_integer_fields if getattr(config, name) <= 0]
    if invalid:
        parser.error("these arguments must be positive: " + ", ".join(invalid))
    if config.num_train_epochs <= 0 or config.learning_rate <= 0:
        parser.error("num_train_epochs and learning_rate must be positive")
    if config.lora_dropout < 0 or config.lora_dropout >= 1:
        parser.error("lora_dropout must be in [0, 1)")
    return config


def trainer_class(model_size: str = "0.8b"):
    """Bind the existing Trainer to the matching checkpoint serializer/schema."""
    binding = decoder_binding(model_size)

    class Phase1MaskTrainer(_Phase1MaskTrainer):
        wrapper_type = binding.wrapper_type
        _assert_decoder_fp32 = staticmethod(binding.assert_fp32)
        _save_decoder = staticmethod(binding.save_decoder)
        decoder_schema = binding.schema
        decoder_scale_key = binding.scale_key

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.phase1_init = str(kwargs.pop("phase1_init", f"checkpoints/Qwen3.5-{model_size.upper()}"))
            self.phase1_seed = int(kwargs.pop("phase1_seed", DEFAULT_SEED))
            super().__init__(*args, **kwargs)
            # Loss is a query/sample mean, so retain ordinary accumulation scaling.
            self.model_accepts_loss_kwargs = False

    return Phase1MaskTrainer


def _merge_and_save_final(
    trainer: _Phase1MaskTrainer,
    output_dir: Path,
    processor: Any,
    tokenizer: Any,
    base_path: Path,
    trainable_counts: Mapping[str, int],
    seed: int,
) -> None:
    model = _trainer_model(trainer)
    if not isinstance(model, trainer.wrapper_type):
        raise TypeError(f"Unexpected model at final save: {type(model).__name__}")
    if not isinstance(model.base_model, PeftModel):
        raise TypeError("Phase 1 final save requires a PEFT model")

    # Preserve the adapter next to the merged standalone model. This allows
    # independent audits to reconstruct the exact base+adapter path.
    model.base_model.save_pretrained(str(output_dir), safe_serialization=True, save_embedding_layers=False)
    trainer._save_decoder(
        model,
        output_dir,
        base_model_name_or_path=str(base_path),
        extra_manifest={
            "checkpoint": False,
            "seed": seed,
            "proxy_mask": "bbox_hard",
            "trainable_parameters": dict(trainable_counts),
        },
    )

    merged = model.base_model.merge_and_unload(safe_merge=True)
    _set_use_cache(merged, True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    processor.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Keep the decoder manifest/weights after any model save operation and
    # record that the final model is the merged artifact.
    trainer._save_decoder(
        model,
        output_dir,
        base_model_name_or_path=str(base_path),
        extra_manifest={
            "checkpoint": False,
            "seed": seed,
            "proxy_mask": "bbox_hard",
            "trainable_parameters": dict(trainable_counts),
            "merged_model": True,
        },
    )


def _build_model_and_tokenizer(
    base_path: Path,
    *,
    model_max_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    dtype: torch.dtype = torch.bfloat16,
    model_size: str = "0.8b",
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(base_path),
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(base_path),
        dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    validate_backbone(model, model_size)
    token_ids = prepare_phase1_tokenizer_and_model(
        model,
        tokenizer,
        model_max_length=model_max_length,
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, token_ids


def main() -> None:
    config = _parse_args()
    binding = decoder_binding(config.model_size)
    install_qwen35_compatibility()
    transformers.set_seed(config.seed)

    base_path = Path(config.model_name_or_path).resolve()
    output_dir = Path(config.output_dir).resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Phase 1 official initialization is missing: {base_path}")
    data_config = (Path.cwd() / config.data_path).resolve()
    expected_data_config = (Path.cwd() / PHASE1_DATA_CONFIG).resolve()
    if data_config != expected_data_config:
        raise ValueError(f"Phase 1 must use the unified data config {expected_data_config}; received {data_config}")
    if not data_config.is_file():
        raise FileNotFoundError(f"Missing unified Phase 1 data config: {data_config}")
    if config.model_size in {"4b", "9b"}:
        resume_checkpoint = _prepare_scaling_output_dir(
            output_dir, config.resume_from_checkpoint,
            model_size=config.model_size, experiment_name=config.experiment_name,
        )
    else:
        resume_checkpoint = _prepare_output_dir(output_dir, config.resume_from_checkpoint)

    model, tokenizer, token_ids = _build_model_and_tokenizer(
        base_path,
        model_max_length=config.model_max_length,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        model_size=config.model_size,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        str(base_path),
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        local_files_only=True,
    )
    data_args = make_data_args(
        processor,
        data_path=str(data_config),
        image_folder=config.image_folder,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_conv_turns=config.max_conv_turns,
    )
    dataset = Phase1IndexedDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=data_config,
        data_args=data_args,
    )
    collator = Phase1TrainingCollator(tokenizer=tokenizer, model_max_length=config.model_max_length)

    wrapper = binding.wrapper_type(
        model,
        decoder_channels=config.decoder_channels,
        **{binding.scale_key: getattr(config, binding.scale_key)},
        image_token_id=token_ids["image_token_id"],
        pointer_pad_token_id=token_ids["pointer_pad_token_id"],
    )
    _set_use_cache(wrapper, False)
    trainable_counts = binding.configure_trainable(wrapper, require_lora=True)
    binding.assert_fp32(wrapper)
    if _is_main_process():
        print(f"Phase 1 trainable parameter contract: {trainable_counts}", flush=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    updates_per_epoch = math.ceil(
        len(dataset) / (world_size * config.per_device_train_batch_size * config.gradient_accumulation_steps)
    )
    gradient_kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}} if config.model_size == "9b" else {}
    args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        lr_scheduler_type=config.lr_scheduler_type,
        bf16=config.bf16,
        tf32=config.tf32,
        gradient_checkpointing=config.gradient_checkpointing,
        **gradient_kwargs,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        label_names=[],
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = trainer_class(config.model_size)(
        model=wrapper,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        phase1_init=str(base_path),
        phase1_seed=config.seed,
    )

    if _is_main_process():
        metadata = {
            "schema_version": 1,
            "status": "started",
            "experiment": config.experiment_name,
            "seed": config.seed,
            "initialization": {
                "path": str(base_path),
                "config_sha256": sha256_file(base_path / "config.json"),
            },
            "data": {
                "config_path": str(data_config),
                "config_sha256": sha256_file(data_config),
                "sources": PHASE1_EXPECTED_RECORDS,
                "total_records": PHASE1_TOTAL_RECORDS,
                "max_conv_turns": int(data_args.max_conv_turns),
            },
            "token_ids": token_ids,
            "objective": {
                "target": "bbox_hard_proxy_cell_overlap",
                "loss": "0.5_bce_plus_0.5_dice",
                "language_loss_weight": 0.0,
                "pointer_loss_weight": 0.0,
            },
            "model": {
                "lora": {
                    "r": config.lora_r,
                    "alpha": config.lora_alpha,
                    "dropout": config.lora_dropout,
                    "target_modules": "all-linear",
                },
                "decoder_channels": config.decoder_channels,
                binding.scale_key: getattr(config, binding.scale_key),
                **model_metadata(config.model_size),
                "trainable_parameters": trainable_counts,
            },
            "training": {
                "world_size": world_size,
                "updates_per_epoch": updates_per_epoch,
                "warmup_steps": 0,
                "arguments": _serialise_training_args(args),
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if config.model_size in {"4b", "9b"}:
            metadata["backbone"] = f"Qwen3.5-{config.model_size.upper()}"
            write_restart_metadata(metadata, output_dir, resume_checkpoint, config.model_size)
        else:
            write_json_atomic(metadata, output_dir / "run_metadata.json")

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    _assert_finite_state(trainer)
    binding.assert_fp32(_trainer_model(trainer))
    trainer.save_state()

    # Ensure every rank has finished its optimizer/RNG writes before rank 0
    # performs the large merged-model export.  A second barrier keeps the
    # remaining workers alive until that export and receipt are complete.
    _distributed_barrier()
    if _is_main_process():
        trainer.save_model(str(output_dir))
        _merge_and_save_final(
            trainer,
            output_dir,
            processor,
            tokenizer,
            base_path,
            trainable_counts,
            config.seed,
        )
        decoder_manifest, decoder_weights = _require_decoder_artifact(output_dir)
        final_metadata = {
            "status": "complete",
            "global_step": int(trainer.state.global_step),
            "epoch": float(trainer.state.epoch or 0.0),
            "output_dir": str(output_dir),
            "artifacts": {
                "mask_decoder_manifest": str(decoder_manifest),
                "mask_decoder_weights": str(decoder_weights),
                "mask_decoder_sha256": sha256_file(decoder_weights),
            },
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(final_metadata, output_dir / "completion_receipt.json")
    _distributed_barrier()


if __name__ == "__main__":
    main()
