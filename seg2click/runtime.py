"""Shared data loading, token initialization and processor setup for Seg2Click.

All mask experiments use the indexed sources in ``data/data_config.yaml``.
Backbone-specific wrappers retain their vocabulary checks, patch
geometry and sequence-length defaults; dataset and collation logic is shared.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import transformers
import yaml


# Transformers 5.16 moved this symbol, while the repository dataset imports it
# from the old location.  Install the compatibility alias before importing the
# dataset module.  This is process-local and does not alter repository files.
import transformers.trainer as _hf_trainer
import torch.nn as _nn

if not hasattr(_hf_trainer, "ALL_LAYERNORM_LAYERS"):
    _hf_trainer.ALL_LAYERNORM_LAYERS = [_nn.LayerNorm]

from qwen_vl_utils import process_vision_info as _process_vision_info_impl

import seg2click.dataset as _legacy_dataset
from seg2click.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    IGNORE_INDEX,
)


PHASE1_DATA_CONFIG = "data/data_config.yaml"
PHASE1_EXPECTED_RECORDS = {
    "uground_bbox": 756451,
    "guienv_bbox": 262445,
    "guiact_bbox": 42545,
    "amex_bbox": 101537,
    "androidcontrol_bbox": 39701,
    "wave_ui_bbox": 52529,
}
PHASE1_TOTAL_RECORDS = sum(PHASE1_EXPECTED_RECORDS.values())


DATA_CONFIG = PHASE1_DATA_CONFIG
EXPECTED_RECORDS = PHASE1_EXPECTED_RECORDS
TOTAL_RECORDS = PHASE1_TOTAL_RECORDS


def _install_compatibility(marker: str, patch_size: int) -> None:
    """Install the two Transformers/qwen-vl-utils process-local shims."""

    # ``apply_chat_template`` in Transformers 5 returns a BatchEncoding for a
    # legacy call which asks for tokenization but not a dict.  The original
    # dataset formatter expects a plain list of ids in those calls.
    tokenizer_base = transformers.tokenization_utils_base.PreTrainedTokenizerBase
    original = tokenizer_base.apply_chat_template
    if not getattr(original, marker, False):

        def _apply_chat_template(self, *args: Any, **kwargs: Any):
            if (
                kwargs.get("tokenize", True)
                and "return_dict" not in kwargs
                and "return_tensors" not in kwargs
            ):
                kwargs["return_dict"] = False
            return original(self, *args, **kwargs)

        setattr(_apply_chat_template, marker, True)
        tokenizer_base.apply_chat_template = _apply_chat_template

    # The inherited formatter omits patch size; explicitly reset it when
    # switching backbones within the same process.
    def _process_vision_info(
        conversations: Any,
        return_video_kwargs: bool = False,
        return_video_metadata: bool = False,
        image_patch_size: int = patch_size,
        **kwargs: Any,
    ):
        del image_patch_size
        kwargs.pop("return_video_kwargs", None)
        kwargs.pop("return_video_metadata", None)
        return _process_vision_info_impl(
            conversations,
            return_video_kwargs=return_video_kwargs,
            return_video_metadata=return_video_metadata,
            image_patch_size=patch_size,
            **kwargs,
        )

    _legacy_dataset.process_vision_info = _process_vision_info


def install_qwen35_compatibility() -> None:
    _install_compatibility("_phase1_qwen35_legacy_list_compat", 16)


def install_qwen3vl_compatibility() -> None:
    _install_compatibility("_seg2click_qwen3vl_legacy_list_compat", 16)


def install_qwen25vl_compatibility() -> None:
    _install_compatibility("_seg2click_qwen25vl_legacy_list_compat", 14)


def install_qwen2vl_2b_compatibility() -> None:
    _install_compatibility("_seg2click_qwen2vl_2b_legacy_list_compat", 14)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_tokenizer_and_model(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_max_length: int,
    backbone: str,
) -> Dict[str, int]:
    """Resize embeddings using only tokenizer-visible rows for their mean."""

    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    existing.update(getattr(tokenizer, "extra_special_tokens", []) or [])
    missing = [token for token in ADDITIONAL_SPECIAL_TOKENS if token not in existing]

    input_embeddings = model.get_input_embeddings()
    # Physical embeddings may include unused padding. Exclude those rows
    # from the deterministic initialization of the five pointer tokens.
    old_token_count = int(len(tokenizer))
    old_embedding_rows = int(input_embeddings.weight.shape[0])
    if old_token_count > old_embedding_rows:
        raise ValueError(
            "Tokenizer has more tokens than the model embedding matrix: "
            f"{old_token_count} > {old_embedding_rows}"
        )
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    target_vocab = int(len(tokenizer))
    if target_vocab < old_token_count:
        raise ValueError(
            f"Adding {'Phase 1' if backbone == 'qwen35' else 'Seg2Click'} tokens unexpectedly reduced tokenizer size: "
            f"{target_vocab} < {old_token_count}"
        )
    if target_vocab != old_embedding_rows:
        with torch.no_grad():
            input_mean = input_embeddings.weight[:old_token_count].detach().float().mean(dim=0)
            output_embeddings = model.get_output_embeddings()
            output_mean = (
                output_embeddings.weight[:old_token_count].detach().float().mean(dim=0)
                if output_embeddings is not None
                else None
            )
        model.resize_token_embeddings(target_vocab, pad_to_multiple_of=None, mean_resizing=False)
        with torch.no_grad():
            input_embeddings = model.get_input_embeddings()
            input_embeddings.weight[old_token_count:target_vocab].copy_(
                input_mean.to(
                    device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype
                )
            )
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and output_mean is not None:
                output_embeddings.weight[old_token_count:target_vocab].copy_(
                    output_mean.to(
                        device=output_embeddings.weight.device, dtype=output_embeddings.weight.dtype
                    )
                )
    elif target_vocab > old_token_count:
        # The physical matrix already has enough rows.  This branch is useful
        # for a model implementation that declines a shrink operation; fill
        # exactly the newly exposed tokenizer rows and leave unused padding
        # rows untouched.
        with torch.no_grad():
            input_mean = input_embeddings.weight[:old_token_count].detach().float().mean(dim=0)
            input_embeddings.weight[old_token_count:target_vocab].copy_(
                input_mean.to(
                    device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype
                )
            )
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                output_mean = (
                    output_embeddings.weight[:old_token_count].detach().float().mean(dim=0)
                )
                output_embeddings.weight[old_token_count:target_vocab].copy_(
                    output_mean.to(
                        device=output_embeddings.weight.device, dtype=output_embeddings.weight.dtype
                    )
                )

    # Transformers 5 no longer exposes the old attribute used by the legacy
    # formatter.  Keep an instance-local compatibility alias.
    tokenizer.__dict__["additional_special_tokens"] = list(ADDITIONAL_SPECIAL_TOKENS)
    if int(model_max_length) <= 0:
        raise ValueError(f"model_max_length must be positive, got {model_max_length}")
    tokenizer.model_max_length = int(model_max_length)
    tokenizer.padding_side = "right"

    # Keep nested and top-level vocab metadata synchronized where available.
    config = getattr(model, "config", None)
    if config is not None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "vocab_size"):
            text_config.vocab_size = target_vocab
        if hasattr(config, "vocab_size"):
            config.vocab_size = target_vocab
        config.pointer_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
        config.pointer_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
        config.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)

    ids = {
        "pointer_start_token_id": int(tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)),
        "pointer_end_token_id": int(tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)),
        "pointer_pad_token_id": int(tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)),
        "image_token_id": int(getattr(config, "image_token_id")),
        "vocab_size": target_vocab,
    }
    if backbone == "qwen35":
        expected = {
            "pointer_start_token_id": 248079,
            "pointer_end_token_id": 248080,
            "pointer_pad_token_id": 248081,
        }
        for key, value in expected.items():
            if ids[key] != value:
                raise ValueError(f"Unexpected Phase 1 token id {key}: {ids[key]} != {value}")
    else:
        vocab = tokenizer.get_vocab()
        special_ids = []
        for token in ADDITIONAL_SPECIAL_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if token not in vocab or token_id != vocab[token] or encoded != [token_id]:
                raise ValueError(
                    f"Seg2Click token is not represented by one vocabulary id: {token!r}"
                )
            if not 0 <= int(token_id) < target_vocab:
                raise ValueError(f"Seg2Click token id is outside the resized vocabulary: {token!r}")
            special_ids.append(int(token_id))
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("Seg2Click special tokens must have distinct vocabulary ids")
        image_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        if ids["image_token_id"] != image_token_id or image_token_id in special_ids:
            raise ValueError(f"{backbone} image token does not match its model configuration")
    return ids


def prepare_phase1_tokenizer_and_model(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_max_length: int = 8192,
) -> Dict[str, int]:
    """Qwen3.5: preserve fixed pointer token IDs and the 8192-token default."""
    return _prepare_tokenizer_and_model(
        model, tokenizer, model_max_length=model_max_length, backbone="qwen35"
    )


def prepare_qwen3vl_tokenizer_and_model(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_max_length: int = 24576,
) -> Dict[str, int]:
    return _prepare_tokenizer_and_model(
        model, tokenizer, model_max_length=model_max_length, backbone="Qwen3-VL"
    )


def prepare_qwen25vl_tokenizer_and_model(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_max_length: int = 24576,
) -> Dict[str, int]:
    return _prepare_tokenizer_and_model(
        model, tokenizer, model_max_length=model_max_length, backbone="Qwen2.5-VL"
    )


def prepare_qwen2vl_tokenizer_and_model(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_max_length: int = 24576,
) -> Dict[str, int]:
    return _prepare_tokenizer_and_model(
        model, tokenizer, model_max_length=model_max_length, backbone="Qwen2-VL-2B"
    )


def _configure_processor(
    processor: Any,
    tokenizer: Any,
    *,
    min_pixels: int,
    max_pixels: int,
    patch_size: int,
    backbone: str | None = None,
) -> Any:
    """Install the legacy image-processor aliases required by preprocessing."""

    processor.tokenizer = tokenizer
    image_processor = processor.image_processor
    image_processor.patch_size = int(
        getattr(image_processor, "patch_size", patch_size) or patch_size
    )
    image_processor.merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
    if backbone is not None and (
        image_processor.patch_size != 14 or image_processor.merge_size != 2
    ):
        raise ValueError(f"{backbone} requires patch_size=14 and merge_size=2")
    size = getattr(image_processor, "size", {})
    if not hasattr(image_processor, "min_pixels"):
        shortest = size.get("shortest_edge") if isinstance(size, Mapping) else None
        image_processor.min_pixels = int(shortest or min_pixels)
    if not hasattr(image_processor, "max_pixels"):
        longest = size.get("longest_edge") if isinstance(size, Mapping) else None
        image_processor.max_pixels = int(longest or max_pixels)
    # Force the requested bounds even when the processor exposes stale defaults.
    image_processor.min_pixels = int(min_pixels)
    image_processor.max_pixels = int(max_pixels)
    if backbone is not None and isinstance(size, Mapping):
        image_processor.size = dict(
            size, shortest_edge=int(min_pixels), longest_edge=int(max_pixels)
        )
    return processor


def configure_processor(
    processor: Any, tokenizer: Any, *, min_pixels: int = 3136, max_pixels: int = 5720064
) -> Any:
    """Qwen3.5/Qwen3-VL geometry (16px fallback; retain processor size metadata)."""
    return _configure_processor(
        processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels, patch_size=16
    )


def configure_qwen25vl_processor(
    processor: Any, tokenizer: Any, *, min_pixels: int = 3136, max_pixels: int = 5720064
) -> Any:
    return _configure_processor(
        processor,
        tokenizer,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        patch_size=14,
        backbone="Qwen2.5-VL",
    )


def configure_qwen2vl_processor(
    processor: Any, tokenizer: Any, *, min_pixels: int = 3136, max_pixels: int = 5720064
) -> Any:
    return _configure_processor(
        processor,
        tokenizer,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        patch_size=14,
        backbone="Qwen2-VL-2B",
    )


class Phase1IndexedDataset(_legacy_dataset.LazySupervisedDataset):
    """Lazy Phase 1 dataset backed by prepared JSONL offset indexes."""

    _protocol = "Phase 1"
    _configure_processor = staticmethod(configure_processor)

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        data_path: str | Path,
        data_args: Any,
    ) -> None:
        config_path = Path(data_path).resolve()
        # The data contract is rooted at the repository containing ``seg2click``.
        expected_path = (Path(__file__).resolve().parents[1] / PHASE1_DATA_CONFIG).resolve()
        if config_path != expected_path:
            raise ValueError(f"{self._protocol} must use {expected_path}; received {config_path}")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)

        # Do not call LazySupervisedDataset.__init__: it reads every JSON array.
        torch.utils.data.Dataset.__init__(self)
        self.tokenizer = tokenizer
        self.processor = self._configure_processor(
            processor,
            tokenizer,
            min_pixels=int(getattr(data_args, "min_pixels", 3136)),
            max_pixels=int(getattr(data_args, "max_pixels", 5720064)),
        )
        self._sources = []
        self._cumulative_sizes = []
        self.include_pointer_targets = False
        self.include_bbox_targets = True
        self.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
        self.pointer_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
        self.pointer_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
        self.data_args = data_args
        self.data_args.dataset_paths = []
        self.data_args.data_path = str(config_path)
        self.data_args.data_split = None
        self.data_args.early_mix_text = bool(getattr(self.data_args, "early_mix_text", False))
        self.data_args.max_conv_turns = int(getattr(self.data_args, "max_conv_turns", 10))

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        datasets = config.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"No datasets found in {config_path}")

        total = 0
        seen_names = set()
        config_dir = config_path.parent
        for entry in datasets:
            raw_json = entry.get("json_path")
            raw_images = entry.get("images_folder")
            name = str(entry.get("name") or Path(str(raw_json)).stem)
            if name in seen_names:
                raise ValueError(f"Duplicate {self._protocol} dataset name: {name}")
            seen_names.add(name)
            sampling_strategy = str(entry.get("sampling_strategy", "all"))
            if sampling_strategy != "all":
                raise ValueError(
                    f"{self._protocol} requires sampling_strategy='all' for {name}, "
                    f"got {sampling_strategy!r}"
                )
            json_path = self._resolve_path(config_dir, raw_json)
            images_folder = self._resolve_path(config_dir, raw_images)
            if json_path.endswith(".json"):
                jsonl_path = json_path[:-5] + ".jsonl"
            elif json_path.endswith(".jsonl"):
                jsonl_path = json_path
            else:
                raise ValueError(f"Unsupported configured source: {json_path}")
            index_path = jsonl_path + ".idx"
            if not Path(jsonl_path).is_file() or not Path(index_path).is_file():
                raise FileNotFoundError(
                    f"Prepared {self._protocol} source missing for {name}: {jsonl_path}, {index_path}"
                )
            if not Path(images_folder).is_dir():
                raise FileNotFoundError(f"Missing images folder for {name}: {images_folder}")
            source = _legacy_dataset.IndexedJsonlSource(
                name=name,
                jsonl_path=jsonl_path,
                index_path=index_path,
                images_folder=images_folder,
            )
            expected = PHASE1_EXPECTED_RECORDS.get(name)
            if expected is not None and len(source) != expected:
                raise ValueError(f"Record count mismatch for {name}: {len(source)} != {expected}")
            self.data_args.dataset_paths.append(json_path)
            self._append_source(source)
            total += len(source)

        expected_names = set(PHASE1_EXPECTED_RECORDS)
        if seen_names != expected_names:
            raise ValueError(
                f"{self._protocol} data config sources differ from the fixed protocol: "
                f"missing={sorted(expected_names - seen_names)}, "
                f"unexpected={sorted(seen_names - expected_names)}"
            )
        if total != PHASE1_TOTAL_RECORDS:
            raise ValueError(
                f"{self._protocol} total record count mismatch: {total} != {PHASE1_TOTAL_RECORDS}"
            )

    @staticmethod
    def _resolve_path(config_dir: Path, value: Any) -> str:
        if value is None:
            return str(config_dir.resolve())
        path = Path(str(value))
        if not path.is_absolute():
            path = config_dir / path
        return str(path.resolve())


class Qwen25VLIndexedDataset(Phase1IndexedDataset):
    """The same indexed protocol with Qwen2.5-VL's 14px processor setup."""

    _protocol = "Seg2Click"
    _configure_processor = staticmethod(configure_qwen25vl_processor)


class Qwen2VLIndexedDataset(Phase1IndexedDataset):
    """The same indexed protocol with Qwen2-VL's 14px processor setup."""

    _protocol = "Seg2Click"
    _configure_processor = staticmethod(configure_qwen2vl_processor)


@dataclass
class Phase1TrainingCollator:
    """Collate variable-query records without padding visual target fields."""

    _protocol = "Phase 1"
    _name = "Phase1TrainingCollator"

    tokenizer: Any
    model_max_length: int = 8192

    @staticmethod
    def _pad(values: Sequence[torch.Tensor], padding_value: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            values, batch_first=True, padding_value=padding_value
        )

    def __call__(self, instances: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not instances:
            raise ValueError(f"{self._name} received an empty batch")
        max_length = int(self.model_max_length)
        input_ids = [
            torch.as_tensor(x["input_ids"], dtype=torch.long).reshape(-1)[:max_length]
            for x in instances
        ]
        labels = [
            torch.as_tensor(x["labels"], dtype=torch.long).reshape(-1)[:max_length]
            for x in instances
        ]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
            self.tokenizer.pad_token_id = pad_id
        batch: Dict[str, Any] = {
            "input_ids": self._pad(input_ids, int(pad_id)),
            "labels": self._pad(labels, IGNORE_INDEX),
        }
        batch["attention_mask"] = batch["input_ids"].ne(int(pad_id))
        if "pixel_values" not in instances[0] or "image_grid_thw" not in instances[0]:
            raise KeyError(f"{self._protocol} records must contain pixel_values and image_grid_thw")
        batch["pixel_values"] = torch.cat(
            [torch.as_tensor(x["pixel_values"]) for x in instances], dim=0
        )
        batch["image_grid_thw"] = torch.cat(
            [torch.as_tensor(x["image_grid_thw"]) for x in instances], dim=0
        )

        for key in ("bbox_targets", "bbox_image_indices", "bbox_image_sizes", "id", "image_counts"):
            if key in instances[0]:
                batch[key] = [x.get(key) for x in instances]
        # The wrapper derives modality ids when the legacy preprocessing did not
        # retain them.  Keeping this explicit flag in the batch makes accidental
        # use of the pointer/LM losses easier to diagnose.
        batch["compute_mask_loss"] = True
        return batch


@dataclass
class Seg2ClickTrainingCollator(Phase1TrainingCollator):
    """Qwen2/Qwen2.5-VL retain their 24576-token collation default."""

    _protocol = "Seg2Click"
    _name = "Seg2ClickTrainingCollator"
    model_max_length: int = 24576


def write_json_atomic(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def make_data_args(
    processor: Any,
    *,
    data_path: str = PHASE1_DATA_CONFIG,
    image_folder: str = "",
    min_pixels: int = 3136,
    max_pixels: int = 5720064,
    max_conv_turns: int = 10,
) -> Any:
    """Return a small namespace compatible with the inherited formatter."""

    from types import SimpleNamespace

    return SimpleNamespace(
        data_path=data_path,
        data_split=None,
        early_mix_text=False,
        image_folder=image_folder,
        min_pixels=int(min_pixels),
        max_pixels=int(max_pixels),
        max_conv_turns=int(max_conv_turns),
        processor=processor,
        dataset_paths=[],
    )


__all__ = [
    "PHASE1_DATA_CONFIG",
    "PHASE1_EXPECTED_RECORDS",
    "PHASE1_TOTAL_RECORDS",
    "DATA_CONFIG",
    "EXPECTED_RECORDS",
    "TOTAL_RECORDS",
    "Phase1IndexedDataset",
    "Qwen25VLIndexedDataset",
    "Qwen2VLIndexedDataset",
    "Phase1TrainingCollator",
    "Seg2ClickTrainingCollator",
    "configure_processor",
    "configure_qwen25vl_processor",
    "configure_qwen2vl_processor",
    "install_qwen35_compatibility",
    "install_qwen3vl_compatibility",
    "install_qwen25vl_compatibility",
    "install_qwen2vl_2b_compatibility",
    "make_data_args",
    "prepare_phase1_tokenizer_and_model",
    "prepare_qwen3vl_tokenizer_and_model",
    "prepare_qwen25vl_tokenizer_and_model",
    "prepare_qwen2vl_tokenizer_and_model",
    "sha256_file",
    "write_json_atomic",
]
