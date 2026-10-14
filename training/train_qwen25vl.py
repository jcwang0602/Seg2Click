"""Qwen2.5-VL Seg2Click training entry point (3B/7B).

Model size, batch size, learning rate and gradient-checkpointing strategy are
command-line options. Only LoRA adapters and the FP32 mask decoder are trained.
"""

from __future__ import annotations

# Preserve direct-file execution after moving this entry point under training/.
if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _EntryPath
    sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))


import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from seg2click.qwen_vl_model import (
    Seg2ClickQwen25VLWrapper,
    assert_decoder_fp32,
    configure_trainable_parameters,
    save_seg2click_decoder,
)
from seg2click.checkpointing import Seg2ClickQwen25VLSplitCheckpointingWrapper
from seg2click.runtime import (
    DATA_CONFIG,
    EXPECTED_RECORDS,
    TOTAL_RECORDS,
    Qwen25VLIndexedDataset as Seg2ClickIndexedDataset,
    Seg2ClickTrainingCollator,
    install_qwen25vl_compatibility,
    make_data_args,
    prepare_qwen25vl_tokenizer_and_model as prepare_tokenizer_and_model,
    sha256_file,
    write_json_atomic,
)

DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 24576
DEFAULT_BATCH_PER_DEVICE = 1
DEFAULT_GRAD_ACCUMULATION = 4
DEFAULT_EPOCHS = 1
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SAVE_STEPS = 2000
DEFAULT_DECODER_CHANNELS = 1024
DEFAULT_PIXEL_SHUFFLE_FACTOR = 28
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 5720064
DEFAULT_MAX_CONV_TURNS = 10
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05


from training.common import (
    _str_to_bool,
    _rank,
    _is_main_process,
    _distributed_barrier,
    _unwrap,
    _trainer_model,
    _set_use_cache,
    _assert_finite_seg2click_state as _assert_finite_state,
    _serialise_training_args,
    _require_seg2click_decoder_artifact as _require_decoder_artifact,
)
from training.mask_trainers import Seg2ClickTrainer as _Seg2ClickTrainer
from training.qwen25vl_restart import prepare_output_dir as _prepare_output_dir, record_previous_attempt


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_size", choices=("3b", "7b"), default="3b")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--attn_implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--checkpointing_mode", choices=("global", "split"), default="split",
                        help="global: one Trainer-level GC switch; split: independent vision / LLM checkpointing.")
    parser.add_argument("--vision_gradient_checkpointing", type=_str_to_bool, default=False)
    parser.add_argument("--llm_gradient_checkpointing", type=_str_to_bool, default=True)
    parser.add_argument("--experiment_name", default=None,
                        help="Experiment identifier stored in run_metadata.json; does not affect computation.")
    parser.add_argument("--data_path", default=DATA_CONFIG)
    parser.add_argument("--image_folder", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_train_epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--per_device_train_batch_size", type=int, default=DEFAULT_BATCH_PER_DEVICE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DEFAULT_GRAD_ACCUMULATION)
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
        help="Qwen2.5-VL post-merge to processed-image expansion factor (14*2=28).",
    )
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--bf16", type=_str_to_bool, default=True)
    parser.add_argument("--tf32", type=_str_to_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=_str_to_bool, default=True)
    config = parser.parse_args(argv)
    if config.model_name_or_path is None:
        config.model_name_or_path = f"checkpoints/Qwen2.5-VL-{config.model_size.upper()}-Instruct"
    if config.experiment_name is None:
        config.experiment_name = f"seg2click_qwen25vl_{config.model_size}"
    if config.checkpointing_mode == "split":
        config.gradient_checkpointing = config.vision_gradient_checkpointing or config.llm_gradient_checkpointing
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
    if config.pixel_shuffle_factor != 28:
        parser.error("Qwen2.5-VL requires --pixel_shuffle_factor 28 (patch size 14 times merge size 2)")
    if config.min_pixels > config.max_pixels:
        parser.error("min_pixels must not exceed max_pixels")
    return config


class Seg2ClickTrainer(_Seg2ClickTrainer):
    """Trainer with mask-only loss and adapter/decoder checkpoint saving."""

    wrapper_type = Seg2ClickQwen25VLWrapper
    _assert_decoder_fp32 = staticmethod(assert_decoder_fp32)
    _save_decoder = staticmethod(save_seg2click_decoder)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.base_initialization = str(kwargs.pop("base_initialization", "checkpoints/Qwen2.5-VL-3B-Instruct"))
        self.experiment_seed = int(kwargs.pop("experiment_seed", DEFAULT_SEED))
        super().__init__(*args, **kwargs)
        # ``Seg2ClickQwen25VLWrapper`` returns a query/sample-mean mask loss;
        # it does not use Trainer's token-count argument.  Transformers 5.x
        # otherwise detects ``**kwargs`` on the wrapper and skips normal
        # gradient-accumulation scaling, changing the requested objective by
        # a factor of ``gradient_accumulation_steps``.
        self.model_accepts_loss_kwargs = False


def wrapper_type_for(config):
    """Select the original wrapper without changing its computations or state keys."""
    if config.checkpointing_mode == "split":
        return Seg2ClickQwen25VLSplitCheckpointingWrapper
    return Seg2ClickQwen25VLWrapper


def build_wrapper(model, token_ids, config):
    options = {}
    if config.checkpointing_mode == "split":
        options.update(vision_gradient_checkpointing=config.vision_gradient_checkpointing,
                       llm_gradient_checkpointing=config.llm_gradient_checkpointing)
    return wrapper_type_for(config)(
        model, decoder_channels=config.decoder_channels,
        pixel_shuffle_factor=config.pixel_shuffle_factor,
        image_token_id=token_ids["image_token_id"],
        pointer_pad_token_id=token_ids["pointer_pad_token_id"], **options,
    )


def _merge_and_save_final(
    trainer: Seg2ClickTrainer,
    output_dir: Path,
    processor: Any,
    tokenizer: Any,
    base_path: Path,
    trainable_counts: Mapping[str, int],
    seed: int,
) -> None:
    model = _trainer_model(trainer)
    if not isinstance(model, Seg2ClickQwen25VLWrapper):
        raise TypeError(f"Unexpected model at final save: {type(model).__name__}")
    if not isinstance(model.base_model, PeftModel):
        raise TypeError("Seg2Click final save requires a PEFT model")

    # Preserve the adapter next to the merged standalone model. This allows
    # independent audits to reconstruct the exact base+adapter path.
    model.base_model.save_pretrained(str(output_dir), safe_serialization=True, save_embedding_layers=False)
    save_seg2click_decoder(
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
    save_seg2click_decoder(
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
    attn_implementation: str = "sdpa",
    model_size: str = "3b",
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(base_path),
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True,
    )
    model, loading_info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_path),
        dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=True,
        attn_implementation=attn_implementation,
        output_loading_info=True,
    )
    loading_errors = {
        key: loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if loading_errors:
        raise RuntimeError(f"Official Qwen2.5-VL backbone did not load completely: {loading_errors}")
    if model.config.model_type != "qwen2_5_vl":
        raise ValueError(f"Expected an official Qwen2.5-VL initialization, got {model.config.model_type!r}")
    if model_size == "7b" and int(model.config.text_config.hidden_size) != 3584:
        raise ValueError("Expected Qwen2.5-VL-7B hidden size 3584")
    token_ids = prepare_tokenizer_and_model(
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
    install_qwen25vl_compatibility()
    transformers.set_seed(config.seed)

    base_path = Path(config.model_name_or_path).resolve()
    output_dir = Path(config.output_dir).resolve()
    if not base_path.is_dir():
        raise FileNotFoundError(f"Seg2Click official initialization is missing: {base_path}")
    data_config = (Path.cwd() / config.data_path).resolve()
    expected_data_config = (Path.cwd() / DATA_CONFIG).resolve()
    if data_config != expected_data_config:
        raise ValueError(f"Seg2Click must use the unified data config {expected_data_config}; received {data_config}")
    if not data_config.is_file():
        raise FileNotFoundError(f"Missing unified Seg2Click data config: {data_config}")
    resume_checkpoint = _prepare_output_dir(
        output_dir, config.resume_from_checkpoint,
        experiment_name=config.experiment_name,
    )

    model, tokenizer, token_ids = _build_model_and_tokenizer(
        base_path,
        model_max_length=config.model_max_length,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
        attn_implementation=config.attn_implementation,
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
    dataset = Seg2ClickIndexedDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=data_config,
        data_args=data_args,
    )
    collator = Seg2ClickTrainingCollator(tokenizer=tokenizer, model_max_length=config.model_max_length)

    wrapper = build_wrapper(model, token_ids, config)
    _set_use_cache(wrapper, False)
    trainable_counts = configure_trainable_parameters(wrapper, require_lora=True)
    assert_decoder_fp32(wrapper)
    if _is_main_process():
        print(f"Seg2Click trainable parameter contract: {trainable_counts}", flush=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    updates_per_epoch = math.ceil(
        len(dataset) / (world_size * config.per_device_train_batch_size * config.gradient_accumulation_steps)
    )
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
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
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
    trainer = Seg2ClickTrainer(
        model=wrapper,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        base_initialization=str(base_path),
        experiment_seed=config.seed,
    )

    if _is_main_process():
        metadata = {
            "schema_version": 1,
            "status": "started",
            "backbone": f"Qwen2.5-VL-{config.model_size.upper()}-Instruct",
            "experiment": config.experiment_name,
            "seed": config.seed,
            "initialization": {
                "path": str(base_path),
                "config_sha256": sha256_file(base_path / "config.json"),
                "weights_loading": "no_missing_unexpected_or_mismatched_keys",
            },
            "data": {
                "config_path": str(data_config),
                "config_sha256": sha256_file(data_config),
                "sources": EXPECTED_RECORDS,
                "total_records": TOTAL_RECORDS,
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
                "pixel_shuffle_factor": config.pixel_shuffle_factor,
                "vision_patch_size": 14,
                "vision_spatial_merge_size": 2,
                "vision_encoder_features": "LLM final image-token state plus Qwen2.5-VL post-merge vision features",
                "vision_fusion": "residual concat MLP with zero-initialized output layer",
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
        if config.checkpointing_mode == "split":
            metadata["model"].update(
                gradient_checkpointing=wrapper.checkpointing_state(),
                gradient_checkpointing_use_reentrant=False,
            )
        metadata_path = output_dir / "run_metadata.json"
        if metadata_path.is_file():
            record_previous_attempt(json.loads(metadata_path.read_text(encoding="utf-8")), metadata)
        metadata["resume_from_checkpoint"] = resume_checkpoint
        write_json_atomic(metadata, metadata_path)

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    _assert_finite_state(trainer)
    assert_decoder_fp32(_trainer_model(trainer))
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
