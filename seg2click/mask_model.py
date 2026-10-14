"""Seg2Click mask model for Qwen3.5 GUI grounding.

A caller supplies a Qwen3.5 model (optionally wrapped by PEFT), and this module composes a small,
query-conditioned decoder on top of the final hidden states at image tokens.

Training contract:

* one still image per record and one dynamic post-merge visual grid;
* bbox *hard* proxy masks are the only Phase 1 target;
* loss is ``0.5 * BCEWithLogits + 0.5 * Dice``;
* the decoder is full-precision (FP32), while the backbone may run in BF16;
* no language-model or pointer loss is added by this wrapper.

The core decoder and target/loss helpers do not import Transformers, so they
can be smoke-tested in a minimal PyTorch environment.  Transformers is only
needed when ``Phase1Qwen35MaskWrapper`` is instantiated with a real model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from seg2click.mask_utils import (
    _as_positive_int,
    _normalise_boxes,
    build_hard_bbox_targets,
    dice_loss_from_logits,
    _first_attr,
    _infer_hidden_size,
    _infer_model_id,
    _infer_spatial_merge_size,
    _extract_final_hidden_states,
    _unwrap_sample_boxes,
)

from seg2click.pixel_shuffle import DEFAULT_PIXEL_SHUFFLE_FACTOR, Qwen35PixelShuffleMaskDecoder


PHASE1_SCHEMA_VERSION = 2
DEFAULT_DECODER_CHANNELS = 128


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Phase 1's fixed ``0.5 * BCE + 0.5 * Dice`` objective.

    BCE is evaluated in FP32 for numerical stability even when the caller is
    under BF16 autocast.  The default weights are intentionally exposed as
    keyword arguments for testing, but the Phase 1 entry point should retain
    their fixed default values.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits/targets shape mismatch: {tuple(logits.shape)} != {tuple(targets.shape)}"
        )
    if logits.ndim < 3:
        raise ValueError(
            "Mask tensors must be [num_targets, height, width] (or batched higher-rank), "
            f"got {tuple(logits.shape)}"
        )
    if not math.isfinite(float(bce_weight)) or bce_weight < 0:
        raise ValueError("bce_weight must be finite and non-negative")
    if not math.isfinite(float(dice_weight)) or dice_weight < 0:
        raise ValueError("dice_weight must be finite and non-negative")
    if bce_weight == 0 and dice_weight == 0:
        raise ValueError("At least one loss weight must be positive")
    target_float = targets.to(device=logits.device, dtype=torch.float32)
    logits_float = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits_float, target_float, reduction="mean")
    dice = dice_loss_from_logits(logits_float, target_float)
    loss = float(bce_weight) * bce + float(dice_weight) * dice
    if not torch.isfinite(loss):
        raise FloatingPointError("Phase 1 mask loss is NaN or Inf")
    return loss


Phase1MaskDecoder = Qwen35PixelShuffleMaskDecoder


@dataclass
class Phase1MaskOutput:
    """Small output object compatible with HuggingFace Trainer's ``.loss``."""

    loss: Optional[torch.Tensor]
    mask_loss: Optional[torch.Tensor]
    mask_logits: List[torch.Tensor]
    mask_targets: Optional[List[torch.Tensor]]
    base_outputs: Any = None
    hidden_states: Optional[torch.Tensor] = None
    grid_shapes: Optional[List[Tuple[int, int]]] = None

    @property
    def logits(self) -> Any:
        """Expose language logits for callers that still inspect them."""

        return getattr(self.base_outputs, "logits", None)

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        values = (self.loss, self.logits, self.mask_logits, self.base_outputs)
        return values[key]


class Phase1Qwen35MaskWrapper(nn.Module):
    """Compose ``Phase1MaskDecoder`` with an existing Qwen3.5 model.

    The wrapped backbone is intentionally untouched.  During ``forward`` all
    language/pointer labels are removed before invoking it, so the returned
    loss consists solely of the proxy-mask objective.  LoRA parameters inside
    the wrapped model remain in the autograd graph and can be optimized by a
    caller's Trainer.
    """

    # Transformers 5.x inspects this marker before deciding whether to pass
    # ``num_items_in_batch`` into ``compute_loss``.  The Phase 1 objective is
    # already a mean over query/sample masks and deliberately does not consume
    # that argument; advertising ``False`` keeps gradient-accumulation scaling
    # exact instead of applying an additional token-count normalization.
    accepts_loss_kwargs = False

    def __init__(
        self,
        base_model: nn.Module,
        *,
        decoder_channels: int = DEFAULT_DECODER_CHANNELS,
        pixel_shuffle_factor: int = DEFAULT_PIXEL_SHUFFLE_FACTOR,
        image_token_id: Optional[int] = None,
        pointer_pad_token_id: Optional[int] = None,
        spatial_merge_size: Optional[int] = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        hidden_size = _infer_hidden_size(base_model)
        self.mask_decoder = Phase1MaskDecoder(
            hidden_size,
            decoder_channels=decoder_channels,
            pixel_shuffle_factor=pixel_shuffle_factor,
        )
        self.image_token_id = (
            _infer_model_id(base_model, "image_token_id", None)
            if image_token_id is None
            else int(image_token_id)
        )
        self.pointer_pad_token_id = (
            _infer_model_id(base_model, "pointer_pad_token_id", None)
            if pointer_pad_token_id is None
            else int(pointer_pad_token_id)
        )
        self.spatial_merge_size = (
            _infer_spatial_merge_size(base_model)
            if spatial_merge_size is None
            else _as_positive_int(spatial_merge_size, "spatial_merge_size")
        )
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        if self.image_token_id is None:
            raise ValueError("image_token_id must be supplied or present in base_model.config")
        if self.pointer_pad_token_id is None:
            raise ValueError(
                "pointer_pad_token_id must be supplied or present in base_model.config"
            )
        self.config = getattr(base_model, "config", None)
        self.mask_decoder.to(dtype=torch.float32)

    @property
    def visual(self) -> Any:
        """Compatibility access to the wrapped Qwen visual tower."""

        return getattr(getattr(self.base_model, "model", None), "visual", None)

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    # HuggingFace Trainer calls these hooks directly when
    # ``gradient_checkpointing=True``.  Delegate to the untouched backbone so
    # wrapping it does not silently disable the Phase 1 memory-saving path.
    def gradient_checkpointing_enable(self, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.base_model, "gradient_checkpointing_enable", None)
        if method is None:
            raise AttributeError("The wrapped Qwen3.5 model has no gradient_checkpointing_enable")
        return method(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args: Any, **kwargs: Any) -> Any:
        method = getattr(self.base_model, "gradient_checkpointing_disable", None)
        if method is None:
            return None
        return method(*args, **kwargs)

    def enable_input_require_grads(self) -> Any:
        method = getattr(self.base_model, "enable_input_require_grads", None)
        if method is None:
            return None
        return method()

    def disable_input_require_grads(self) -> Any:
        method = getattr(self.base_model, "disable_input_require_grads", None)
        if method is None:
            return None
        return method()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate generation to the untouched base model."""

        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Save the LoRA/base component and the decoder in the same directory.

        HuggingFace ``Trainer`` invokes ``model.save_pretrained`` at every
        checkpoint interval.  Delegating only to the base model would produce
        checkpoints that cannot restore the newly trained decoder, so each
        invocation also writes the small FP32 decoder plus its manifest.  The
        standalone ``save_phase1_decoder`` helper remains available for an
        explicit final-save call.
        """

        result = self.base_model.save_pretrained(*args, **kwargs)
        if args:
            save_directory = args[0]
        else:
            save_directory = kwargs.get("save_directory")
        if save_directory is not None:
            save_phase1_decoder(self, save_directory)
        return result

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        bbox_targets: Optional[Any] = None,
        mask_targets: Optional[Any] = None,
        mask_target_image_indices: Optional[Any] = None,
        mask_target_grid_hw: Optional[Any] = None,
        *,
        compute_mask_loss: bool = True,
        output_hidden_states: bool = True,
        **kwargs: Any,
    ) -> Phase1MaskOutput:
        if input_ids is None:
            input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("Phase1 wrapper requires input_ids")
        if image_grid_thw is None:
            image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("Phase1 wrapper requires image_grid_thw")
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}")
        batch_size = input_ids.shape[0]
        grid_tensor = torch.as_tensor(image_grid_thw)
        # The repository collator normally concatenates one ``[1,3]`` row per
        # sample into ``[batch,3]``.  A generic Trainer collator may instead
        # retain a singleton image dimension (``[batch,1,3]``); accept that
        # equivalent representation for a batch of still images.
        if grid_tensor.ndim == 3 and grid_tensor.shape[1] == 1:
            grid_tensor = grid_tensor[:, 0, :]
        elif grid_tensor.ndim == 1 and batch_size == 1 and grid_tensor.numel() == 3:
            grid_tensor = grid_tensor.reshape(1, 3)
        if grid_tensor.ndim != 2 or grid_tensor.shape[-1] != 3:
            raise ValueError(
                f"image_grid_thw must have shape [num_images,3], got {tuple(grid_tensor.shape)}"
            )
        if grid_tensor.shape[0] != batch_size:
            raise ValueError(
                "Phase 1 requires exactly one image per sample: "
                f"num_images={grid_tensor.shape[0]}, batch_size={batch_size}"
            )
        # Keep the caller-provided global targets separate from the local list
        # returned in ``Phase1MaskOutput``.
        global_mask_targets = mask_targets

        # Do not accidentally optimize the pointer/LM objectives from the
        # baseline.  They are removed only from this new call's kwargs.
        base_kwargs = dict(kwargs)
        base_kwargs.pop("input_ids", None)
        base_kwargs.pop("bbox_targets", None)
        # The independent data module may provide already-rasterized global
        # targets.  They are consumed below and must never reach the Qwen
        # backbone, whose forward signature does not know these fields.
        base_kwargs.pop("mask_targets", None)
        base_kwargs.pop("mask_target_image_indices", None)
        base_kwargs.pop("mask_target_grid_hw", None)
        base_kwargs.pop("bbox_image_sizes", None)
        base_kwargs.pop("bbox_image_indices", None)
        base_kwargs.pop("image_counts", None)
        base_kwargs.pop("compute_mask_loss", None)
        base_kwargs.pop("multi_patch_labels", None)
        base_kwargs.pop("visual_token_indices_of_coordinates", None)
        base_kwargs.pop("coordinates", None)
        base_kwargs.pop("labels", None)
        base_kwargs.pop("mask_target_counts", None)
        # Optional identifiers carried by the independent collator are
        # metadata only and must not be forwarded into Qwen's language model.
        for metadata_key in ("id", "sample_id", "record_id", "image_grid_hw"):
            base_kwargs.pop(metadata_key, None)
        base_kwargs["input_ids"] = input_ids
        base_kwargs["image_grid_thw"] = image_grid_thw
        base_kwargs["output_hidden_states"] = bool(output_hidden_states)
        base_kwargs["return_dict"] = True

        # Qwen3.5's official implementation requires modality ids whenever an
        # image grid is supplied.  The legacy formatter predates that API and
        # does not retain the field, so derive the equivalent ids locally.
        if base_kwargs.get("mm_token_type_ids") is None:
            modality_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            modality_ids.masked_fill_(input_ids == self.image_token_id, 1)
            video_token_id = _infer_model_id(self.base_model, "video_token_id", None)
            if video_token_id is not None:
                modality_ids.masked_fill_(input_ids == video_token_id, 2)
            base_kwargs["mm_token_type_ids"] = modality_ids

        # The official Qwen3.5 model always projects hidden states through its
        # language head, even when no language loss is requested.  Keeping one
        # logit is enough to preserve the standard output contract while
        # avoiding a [sequence, 248k-vocabulary] activation.
        base_kwargs["labels"] = None
        base_kwargs["logits_to_keep"] = 1
        base_outputs = self.base_model(**base_kwargs)
        hidden = _extract_final_hidden_states(base_outputs)
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError(
                "Base final hidden states must match input_ids in first two dimensions: "
                f"hidden={tuple(hidden.shape)}, input_ids={tuple(input_ids.shape)}"
            )

        mask_logits: List[torch.Tensor] = []
        collected_targets: Optional[List[torch.Tensor]] = (
            [] if (bbox_targets is not None or global_mask_targets is not None) else None
        )
        grid_shapes: List[Tuple[int, int]] = []
        sample_losses: List[torch.Tensor] = []
        for sample_index in range(batch_size):
            token_ids = input_ids[sample_index]
            visual_indices = torch.nonzero(
                token_ids == self.image_token_id, as_tuple=False
            ).flatten()
            query_indices = torch.nonzero(
                token_ids == self.pointer_pad_token_id, as_tuple=False
            ).flatten()
            if visual_indices.numel() == 0:
                raise ValueError(f"No image tokens found for sample {sample_index}")
            if query_indices.numel() == 0:
                raise ValueError(f"No pointer query tokens found for sample {sample_index}")

            temporal, patch_grid_h, patch_grid_w = [
                _as_positive_int(value, f"image_grid_thw[{sample_index}]")
                for value in grid_tensor[sample_index].tolist()
            ]
            if temporal != 1:
                raise ValueError("Phase 1 supports still images only (temporal grid must be 1)")
            grid_height = patch_grid_h // self.spatial_merge_size
            grid_width = patch_grid_w // self.spatial_merge_size
            if grid_height <= 0 or grid_width <= 0:
                raise ValueError(
                    f"Invalid post-merge grid {patch_grid_h}x{patch_grid_w} "
                    f"with merge={self.spatial_merge_size}"
                )
            expected_visual = grid_height * grid_width
            if visual_indices.numel() != expected_visual:
                raise ValueError(
                    "Image-token/grid mismatch for sample "
                    f"{sample_index}: tokens={visual_indices.numel()}, "
                    f"expected={expected_visual} ({grid_height}x{grid_width})"
                )

            logits = self.mask_decoder(
                hidden[sample_index, visual_indices],
                hidden[sample_index, query_indices],
                grid_height,
                grid_width,
            )
            mask_logits.append(logits)
            grid_shapes.append((grid_height, grid_width))

            if bbox_targets is not None:
                boxes = _normalise_boxes(
                    _unwrap_sample_boxes(bbox_targets, sample_index),
                    device=logits.device,
                )
                if boxes.shape[0] != query_indices.numel():
                    raise ValueError(
                        "bbox/query count mismatch for sample "
                        f"{sample_index}: boxes={boxes.shape[0]}, queries={query_indices.numel()}"
                    )
                targets = build_hard_bbox_targets(
                    boxes,
                    logits.shape[-2],
                    logits.shape[-1],
                    device=logits.device,
                )
                assert collected_targets is not None
                collected_targets.append(targets)
                if compute_mask_loss:
                    sample_losses.append(
                        bce_dice_loss(
                            logits,
                            targets,
                            bce_weight=self.bce_weight,
                            dice_weight=self.dice_weight,
                        )
                    )
            elif global_mask_targets is not None:
                # ``Phase1MaskDataCollator`` stores one global flattened mask
                # per query.  Select the local image slice and expand its
                # coarse cells to the decoder's fixed output resolution.  If
                # metadata is absent, the one-image-per-record contract makes
                # the first grid unambiguous.
                sample_global = _unwrap_sample_boxes(global_mask_targets, sample_index)
                sample_global = torch.as_tensor(
                    sample_global, dtype=torch.float32, device=logits.device
                )
                if sample_global.ndim == 1:
                    sample_global = sample_global.reshape(1, -1)
                if sample_global.ndim != 2 or sample_global.shape[0] != query_indices.numel():
                    raise ValueError(
                        "mask_targets must have shape [num_queries,total_visual_tokens] "
                        f"for sample {sample_index}, got {tuple(sample_global.shape)}"
                    )
                sample_indices = None
                if mask_target_image_indices is not None:
                    sample_indices = torch.as_tensor(
                        _unwrap_sample_boxes(mask_target_image_indices, sample_index),
                        dtype=torch.long,
                        device=logits.device,
                    ).reshape(-1)
                    if sample_indices.numel() != query_indices.numel():
                        raise ValueError(
                            "mask_target_image_indices/query count mismatch: "
                            f"{sample_indices.numel()} != {query_indices.numel()}"
                        )
                if sample_indices is not None and torch.any(sample_indices != 0):
                    raise ValueError(
                        "Phase 1 wrapper requires one image per sample; "
                        f"got image indices {sample_indices.tolist()}"
                    )
                coarse_tokens = grid_height * grid_width
                if sample_global.shape[1] != coarse_tokens:
                    raise ValueError(
                        "mask_targets visual-token count/grid mismatch for sample "
                        f"{sample_index}: {sample_global.shape[1]} != {coarse_tokens}"
                    )
                coarse = sample_global.reshape(query_indices.numel(), grid_height, grid_width)
                targets = F.interpolate(
                    coarse[:, None],
                    size=logits.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
                assert collected_targets is not None
                collected_targets.append(targets)
                if compute_mask_loss:
                    sample_losses.append(
                        bce_dice_loss(
                            logits,
                            targets,
                            bce_weight=self.bce_weight,
                            dice_weight=self.dice_weight,
                        )
                    )

        mask_loss = torch.stack(sample_losses).mean() if sample_losses else None
        if mask_loss is not None and not torch.isfinite(mask_loss):
            raise FloatingPointError("Phase 1 mask loss is non-finite")
        return Phase1MaskOutput(
            loss=mask_loss,
            mask_loss=mask_loss,
            mask_logits=mask_logits,
            mask_targets=collected_targets,
            base_outputs=base_outputs,
            hidden_states=hidden,
            grid_shapes=grid_shapes,
        )


def configure_phase1_trainable_parameters(
    model: nn.Module,
    *,
    mask_decoder_attr: str = "mask_decoder",
    require_lora: bool = True,
) -> Dict[str, int]:
    """Freeze the backbone and enable LoRA plus the complete FP32 decoder.

    This helper is deliberately explicit about the trainable boundary.  It
    works with either the wrapper itself or a parent module containing a PEFT
    model.  Pointer heads, embeddings, language heads and visual towers remain
    frozen unless they are LoRA parameters.
    """

    decoder = getattr(model, mask_decoder_attr, None)
    if decoder is None or not isinstance(decoder, nn.Module):
        raise ValueError(f"Model has no nn.Module named {mask_decoder_attr!r}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    lora_names: List[str] = []
    for name, parameter in model.named_parameters():
        if "lora_" in name.lower():
            parameter.requires_grad = True
            lora_names.append(name)
    for parameter in decoder.parameters():
        parameter.requires_grad = True
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    if require_lora and not lora_names:
        raise ValueError(
            "Phase 1 requires LoRA parameters, but none were found. "
            "Wrap the Qwen3.5 backbone with PEFT before configuring training."
        )
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable.values())),
        "lora_parameters": int(
            sum(
                parameter.numel()
                for name, parameter in trainable.items()
                if "lora_" in name.lower()
            )
        ),
        "mask_decoder_parameters": int(
            sum(parameter.numel() for parameter in decoder.parameters())
        ),
    }


def assert_phase1_decoder_fp32(
    model: nn.Module, *, mask_decoder_attr: str = "mask_decoder"
) -> None:
    """Fail if decoder parameters are not finite FP32 tensors."""

    decoder = getattr(model, mask_decoder_attr, None)
    if decoder is None:
        raise ValueError(f"Model has no {mask_decoder_attr!r}")
    wrong_dtype = [name for name, p in decoder.named_parameters() if p.dtype != torch.float32]
    nonfinite = [name for name, p in decoder.named_parameters() if not torch.isfinite(p).all()]
    if wrong_dtype:
        raise TypeError(f"Phase 1 mask decoder must be FP32, got non-FP32: {wrong_dtype}")
    if nonfinite:
        raise FloatingPointError(f"Phase 1 mask decoder contains NaN/Inf: {nonfinite}")


def save_phase1_decoder(
    model: nn.Module,
    output_dir: str | Path,
    *,
    base_model_name_or_path: Optional[str] = None,
    extra_manifest: Optional[Mapping[str, Any]] = None,
    mask_decoder_attr: str = "mask_decoder",
) -> Path:
    """Save the FP32 decoder and an auditable manifest.

    ``safetensors`` is preferred when installed; a Torch state dict fallback
    keeps local smoke tests usable without that optional package.
    """

    assert_phase1_decoder_fp32(model, mask_decoder_attr=mask_decoder_attr)
    decoder = getattr(model, mask_decoder_attr)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().float().contiguous()
        for name, tensor in decoder.state_dict().items()
    }
    weights_name = "phase1_mask_decoder.safetensors"
    try:
        from safetensors.torch import save_file

        save_file(
            state,
            str(output_path / weights_name),
            metadata={"format": "pt", "component": "phase1_mask_decoder"},
        )
    except Exception:
        weights_name = "phase1_mask_decoder.pt"
        torch.save(state, output_path / weights_name)
    manifest: Dict[str, Any] = {
        "schema_version": PHASE1_SCHEMA_VERSION,
        "method": "phase1_qwen35_bbox_hard_mask",
        "weights": weights_name,
        "base_model_name_or_path": str(base_model_name_or_path)
        if base_model_name_or_path
        else None,
        "hidden_size": int(decoder.hidden_size),
        "decoder_channels": int(decoder.decoder_channels),
        "pixel_shuffle_factor": int(decoder.pixel_shuffle_factor),
        "decoder_architecture": "dynamic_pixel_shuffle",
        "target": "bbox_hard_proxy_cell_overlap",
        "loss": {"type": "bce_plus_dice", "bce_weight": 0.5, "dice_weight": 0.5},
        "parameter_dtype": "float32",
    }
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    (output_path / "phase1_mask_decoder.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path / weights_name


__all__ = [
    "DEFAULT_DECODER_CHANNELS",
    "DEFAULT_PIXEL_SHUFFLE_FACTOR",
    "PHASE1_SCHEMA_VERSION",
    "Phase1MaskDecoder",
    "Phase1MaskOutput",
    "Phase1Qwen35MaskWrapper",
    "assert_phase1_decoder_fp32",
    "bce_dice_loss",
    "build_hard_bbox_targets",
    "configure_phase1_trainable_parameters",
    "dice_loss_from_logits",
    "save_phase1_decoder",
]
