"""Mask-only Trainer behavior shared by the existing experiment variants.

The concrete entry-point classes keep their constructor defaults and bind the
same model wrapper and decoder functions they used before refactoring. Full
fine-tuning variants retain their separate save/resume implementations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import transformers
from peft import PeftModel

from training.common import _trainer_model, _unwrap


class Phase1MaskTrainer(transformers.Trainer):
    """Shared LoRA checkpoint behavior; subclasses bind their original decoder."""

    decoder_schema = 2
    decoder_scale_key = "pixel_shuffle_factor"

    @classmethod
    def _load_decoder_checkpoint(cls, model: torch.nn.Module, checkpoint: Path) -> None:
        """Restore the experiment-local FP32 decoder from one checkpoint."""
        manifest_path = checkpoint / "phase1_mask_decoder.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Phase 1 checkpoint is missing decoder manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Phase 1 decoder manifest: {manifest_path}") from exc
        weights_name = manifest.get("weights")
        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError(f"Decoder manifest has no weights entry: {manifest_path}")
        weights_path = checkpoint / weights_name
        if not weights_path.is_file():
            raise FileNotFoundError(f"Phase 1 decoder weights are missing: {weights_path}")
        if int(manifest.get("schema_version", -1)) != cls.decoder_schema:
            raise ValueError(f"Unsupported Phase 1 decoder schema: {manifest.get('schema_version')!r}")
        expected = {
            "hidden_size": int(model.mask_decoder.hidden_size),
            "decoder_channels": int(model.mask_decoder.decoder_channels),
            cls.decoder_scale_key: int(getattr(model.mask_decoder, cls.decoder_scale_key)),
        }
        for key, value in expected.items():
            if int(manifest.get(key, -1)) != value:
                raise ValueError(f"Phase 1 decoder {key} mismatch: {manifest.get(key)!r} != {value}")
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file

                state = load_file(str(weights_path), device="cpu")
            except Exception as exc:
                raise RuntimeError(f"Unable to load decoder safetensors: {weights_path}") from exc
        else:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        load_result = model.mask_decoder.load_state_dict(state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError(
                "Phase 1 decoder state mismatch: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        cls._assert_decoder_fp32(model)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: torch.nn.Module | None = None) -> None:
        """Restore adapter-only checkpoints plus the separate mask decoder.

        HuggingFace's generic loader recognizes PEFT adapters only when the
        PEFT object is the Trainer model itself.  Phase 1 wraps that object in
        ``Phase1Qwen35MaskWrapper`` and therefore needs this explicit loader.
        Optimizer/scheduler/state files remain owned by the base Trainer.
        """
        checkpoint = Path(resume_from_checkpoint).resolve()
        target = _unwrap(model or self.model)
        if not isinstance(target, self.wrapper_type):
            raise TypeError(f"Unexpected Phase 1 resume model: {type(target).__name__}")
        if not isinstance(target.base_model, PeftModel):
            raise TypeError("Phase 1 resume requires a PEFT LoRA base model")
        adapter_config = checkpoint / "adapter_config.json"
        adapter_weights = checkpoint / "adapter_model.safetensors"
        adapter_weights_pt = checkpoint / "adapter_model.bin"
        has_adapter = adapter_config.is_file() and (adapter_weights.is_file() or adapter_weights_pt.is_file())
        decoder_manifest = checkpoint / "phase1_mask_decoder.json"
        if not has_adapter:
            # Permit a future full-state checkpoint to use the standard loader.
            # An adapter-shaped directory without its decoder must fail loudly
            # instead of silently restarting the mask head.
            if adapter_config.is_file() or adapter_weights.is_file() or adapter_weights_pt.is_file():
                raise FileNotFoundError(f"Phase 1 checkpoint has incomplete adapter files: {checkpoint}")
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            if decoder_manifest.is_file():
                self._load_decoder_checkpoint(target, checkpoint)
            return result
        if not decoder_manifest.is_file():
            raise FileNotFoundError(f"Phase 1 checkpoint is missing decoder manifest: {decoder_manifest}")
        active = getattr(target.base_model, "active_adapters", ["default"])
        if isinstance(active, str):
            active_name = active
        else:
            active_name = active[0] if active else "default"
        try:
            adapter_device = str(next(target.base_model.parameters()).device)
        except StopIteration:
            adapter_device = "cpu"
        target.base_model.load_adapter(str(checkpoint), active_name, is_trainable=True, torch_device=adapter_device)
        self._load_decoder_checkpoint(target, checkpoint)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        del num_items_in_batch
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("Phase 1 model returned no mask loss")
        if not torch.isfinite(loss):
            raise FloatingPointError("Phase 1 mask loss is NaN or Inf")
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        """Save only the LoRA adapter and decoder for lightweight checkpoints."""
        del state_dict
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model = _trainer_model(self)
        if not isinstance(model, self.wrapper_type):
            raise TypeError(f"Unexpected Phase 1 model type: {type(model).__name__}")
        if not isinstance(model.base_model, PeftModel):
            raise TypeError("Phase 1 checkpoint requires a PEFT LoRA base model")
        # The resized embeddings are frozen and deterministically recreated
        # from the official tokenizer on resume.  Explicitly exclude them so
        # every checkpoint remains a compact LoRA+decoder artifact rather than
        # silently copying the ~1 GB language embedding matrix.
        model.base_model.save_pretrained(str(destination), safe_serialization=True, save_embedding_layers=False)
        self._save_decoder(
            model,
            destination,
            base_model_name_or_path=self.phase1_init,
            extra_manifest={"checkpoint": True, "seed": self.phase1_seed},
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(destination))
        elif getattr(self.data_collator, "tokenizer", None) is not None:
            self.data_collator.tokenizer.save_pretrained(str(destination))
        torch.save(self.args, destination / "training_args.bin")


class Seg2ClickTrainer(transformers.Trainer):
    """Shared LoRA checkpoint behavior; subclasses bind their original decoder."""

    decoder_schema = 2
    decoder_scale_key = "pixel_shuffle_factor"

    @classmethod
    def _load_decoder_checkpoint(cls, model: torch.nn.Module, checkpoint: Path) -> None:
        """Restore the experiment-local FP32 decoder from one checkpoint."""
        manifest_path = checkpoint / "phase1_mask_decoder.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Seg2Click checkpoint is missing decoder manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Seg2Click decoder manifest: {manifest_path}") from exc
        weights_name = manifest.get("weights")
        if not isinstance(weights_name, str) or not weights_name:
            raise ValueError(f"Decoder manifest has no weights entry: {manifest_path}")
        weights_path = checkpoint / weights_name
        if not weights_path.is_file():
            raise FileNotFoundError(f"Seg2Click decoder weights are missing: {weights_path}")
        if int(manifest.get("schema_version", -1)) != cls.decoder_schema:
            raise ValueError(f"Unsupported Seg2Click decoder schema: {manifest.get('schema_version')!r}")
        expected = {
            "hidden_size": int(model.mask_decoder.hidden_size),
            "decoder_channels": int(model.mask_decoder.decoder_channels),
            cls.decoder_scale_key: int(getattr(model.mask_decoder, cls.decoder_scale_key)),
        }
        for key, value in expected.items():
            if int(manifest.get(key, -1)) != value:
                raise ValueError(f"Seg2Click decoder {key} mismatch: {manifest.get(key)!r} != {value}")
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file

                state = load_file(str(weights_path), device="cpu")
            except Exception as exc:
                raise RuntimeError(f"Unable to load decoder safetensors: {weights_path}") from exc
        else:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        load_result = model.mask_decoder.load_state_dict(state, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError(
                "Seg2Click decoder state mismatch: "
                f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
            )
        cls._assert_decoder_fp32(model)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: torch.nn.Module | None = None) -> None:
        """Restore adapter-only checkpoints plus the separate mask decoder.

        HuggingFace's generic loader recognizes PEFT adapters only when the
        PEFT object is the Trainer model itself.  Seg2Click wraps that object in
        ``Seg2ClickQwen25VLWrapper`` and therefore needs this explicit loader.
        Optimizer/scheduler/state files remain owned by the base Trainer.
        """
        checkpoint = Path(resume_from_checkpoint).resolve()
        target = _unwrap(model or self.model)
        if not isinstance(target, self.wrapper_type):
            raise TypeError(f"Unexpected Seg2Click resume model: {type(target).__name__}")
        if not isinstance(target.base_model, PeftModel):
            raise TypeError("Seg2Click resume requires a PEFT LoRA base model")
        adapter_config = checkpoint / "adapter_config.json"
        adapter_weights = checkpoint / "adapter_model.safetensors"
        adapter_weights_pt = checkpoint / "adapter_model.bin"
        has_adapter = adapter_config.is_file() and (adapter_weights.is_file() or adapter_weights_pt.is_file())
        decoder_manifest = checkpoint / "phase1_mask_decoder.json"
        if not has_adapter:
            # Permit a future full-state checkpoint to use the standard loader.
            # An adapter-shaped directory without its decoder must fail loudly
            # instead of silently restarting the mask head.
            if adapter_config.is_file() or adapter_weights.is_file() or adapter_weights_pt.is_file():
                raise FileNotFoundError(f"Seg2Click checkpoint has incomplete adapter files: {checkpoint}")
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
            if decoder_manifest.is_file():
                self._load_decoder_checkpoint(target, checkpoint)
            return result
        if not decoder_manifest.is_file():
            raise FileNotFoundError(f"Seg2Click checkpoint is missing decoder manifest: {decoder_manifest}")
        active = getattr(target.base_model, "active_adapters", ["default"])
        if isinstance(active, str):
            active_name = active
        else:
            active_name = active[0] if active else "default"
        try:
            adapter_device = str(next(target.base_model.parameters()).device)
        except StopIteration:
            adapter_device = "cpu"
        target.base_model.load_adapter(str(checkpoint), active_name, is_trainable=True, torch_device=adapter_device)
        self._load_decoder_checkpoint(target, checkpoint)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        del num_items_in_batch
        outputs = model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("Seg2Click model returned no mask loss")
        if not torch.isfinite(loss):
            raise FloatingPointError("Seg2Click mask loss is NaN or Inf")
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        """Save only the LoRA adapter and decoder for lightweight checkpoints."""
        del state_dict
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model = _trainer_model(self)
        if not isinstance(model, self.wrapper_type):
            raise TypeError(f"Unexpected Seg2Click model type: {type(model).__name__}")
        if not isinstance(model.base_model, PeftModel):
            raise TypeError("Seg2Click checkpoint requires a PEFT LoRA base model")
        # The resized embeddings are frozen and deterministically recreated
        # from the official tokenizer on resume.  Explicitly exclude them so
        # every checkpoint remains a compact LoRA+decoder artifact rather than
        # silently copying the ~1 GB language embedding matrix.
        model.base_model.save_pretrained(str(destination), safe_serialization=True, save_embedding_layers=False)
        self._save_decoder(
            model,
            destination,
            base_model_name_or_path=self.base_initialization,
            extra_manifest={"checkpoint": True, "seed": self.experiment_seed},
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(destination))
        elif getattr(self.data_collator, "tokenizer", None) is not None:
            self.data_collator.tokenizer.save_pretrained(str(destination))
        torch.save(self.args, destination / "training_args.bin")
