"""Shared mask geometry, Dice objective and backbone-output helpers.

These operations are identical across the Qwen mask experiments. Model-specific
BCE loss messages, decoder layouts and checkpoint manifests stay in each model.
"""

from __future__ import annotations

import math

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _as_positive_int(value: Any, name: str) -> int:
    """Convert scalar-like values to a positive Python integer."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar-like, got shape {tuple(value.shape)}")
        value = value.detach().cpu().item()
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _normalise_boxes(
    bbox_targets: torch.Tensor | Sequence[Sequence[float]],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Validate and return normalized ``[num_targets, 4]`` boxes.

    Bboxes are represented as ``(x_min, y_min, x_max, y_max)`` in normalized
    image coordinates.  Silently clipping malformed boxes would hide data
    errors, so this helper deliberately fails on non-finite/out-of-range or
    degenerate boxes.
    """

    boxes = torch.as_tensor(bbox_targets, dtype=torch.float32, device=device)
    if boxes.ndim == 1:
        if boxes.numel() != 4:
            raise ValueError(
                f"A one-dimensional bbox target must contain four values, got {tuple(boxes.shape)}"
            )
        boxes = boxes.reshape(1, 4)
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"bbox_targets must have shape [num_targets, 4], got {tuple(boxes.shape)}")
    if boxes.shape[0] == 0:
        raise ValueError("bbox_targets must contain at least one box")
    if not torch.isfinite(boxes).all():
        raise ValueError("bbox_targets contain NaN or Inf")
    if torch.any(boxes < 0) or torch.any(boxes > 1):
        raise ValueError("bbox_targets must be normalized to [0, 1]")
    if torch.any(boxes[:, 0] >= boxes[:, 2]) or torch.any(boxes[:, 1] >= boxes[:, 3]):
        raise ValueError("bbox_targets must satisfy x_min < x_max and y_min < y_max")
    return boxes


def build_hard_bbox_targets(
    bbox_targets: torch.Tensor | Sequence[Sequence[float]],
    field_height: int,
    field_width: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Rasterize normalized boxes into a hard, cell-overlap proxy mask.

    The output has shape ``[num_targets, field_height, field_width]``.  A cell
    is positive when its half-open normalized area intersects the bbox.  This
    area-overlap rule guarantees at least one positive cell for every valid
    non-empty box, including tiny controls that are smaller than a cell.
    """

    height = _as_positive_int(field_height, "field_height")
    width = _as_positive_int(field_width, "field_width")
    boxes = _normalise_boxes(bbox_targets, device=device)

    x0 = torch.arange(width, device=boxes.device, dtype=torch.float32) / width
    x1 = (torch.arange(width, device=boxes.device, dtype=torch.float32) + 1.0) / width
    y0 = torch.arange(height, device=boxes.device, dtype=torch.float32) / height
    y1 = (torch.arange(height, device=boxes.device, dtype=torch.float32) + 1.0) / height

    # [N, W] and [N, H], then outer-product into [N, H, W].
    x_overlap = (x1[None, :] > boxes[:, 0:1]) & (x0[None, :] < boxes[:, 2:3])
    y_overlap = (y1[None, :] > boxes[:, 1:2]) & (y0[None, :] < boxes[:, 3:4])
    targets = y_overlap[:, :, None] & x_overlap[:, None, :]
    if torch.any(targets.flatten(1).sum(dim=1) == 0):
        # This should be unreachable after validation, but retaining an
        # explicit guard makes a changed rasterization rule fail loudly.
        raise RuntimeError("A valid bbox produced an empty hard proxy mask")
    return targets.to(dtype=torch.float32)


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute per-instance soft Dice loss from logits."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits/targets shape mismatch: {tuple(logits.shape)} != {tuple(targets.shape)}"
        )
    if logits.ndim < 2:
        raise ValueError("Mask logits must have at least two dimensions")
    if smooth <= 0 or not math.isfinite(float(smooth)):
        raise ValueError("smooth must be a finite positive value")
    probabilities = torch.sigmoid(logits.float())
    target_float = targets.to(device=probabilities.device, dtype=torch.float32)
    intersection = (probabilities * target_float).flatten(1).sum(dim=1)
    denominator = probabilities.flatten(1).sum(dim=1) + target_float.flatten(1).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def _first_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if obj is not None and hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _infer_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = _first_attr(text_config, ("hidden_size",), None)
    if value is None:
        value = _first_attr(config, ("hidden_size",), None)
    if value is None:
        # Last-resort inference from the language-model embedding matrix.
        embedding = model.get_input_embeddings()
        value = embedding.weight.shape[-1]
    return _as_positive_int(value, "hidden_size")


def _infer_model_id(model: nn.Module, name: str, default: Optional[int] = None) -> Optional[int]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    value = _first_attr(config, (name,), None)
    if value is None:
        value = _first_attr(text_config, (name,), None)
    return default if value is None else int(value)


def _infer_spatial_merge_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    value = _first_attr(vision_config, ("spatial_merge_size",), None)
    if value is None:
        visual = _first_attr(getattr(model, "model", None), ("visual",), None)
        value = _first_attr(visual, ("spatial_merge_size",), None)
    return _as_positive_int(2 if value is None else value, "spatial_merge_size")


def _extract_final_hidden_states(base_outputs: Any) -> torch.Tensor:
    """Extract a ``[batch, sequence, hidden]`` final-layer tensor."""

    hidden_states = getattr(base_outputs, "hidden_states", None)
    if hidden_states is not None:
        if isinstance(hidden_states, (tuple, list)):
            if not hidden_states:
                raise ValueError("Base model returned an empty hidden_states tuple")
            hidden_states = hidden_states[-1]
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
    # Some wrappers expose the final language representation directly.
    for name in ("last_hidden_state", "encoder_last_hidden_state"):
        value = getattr(base_outputs, name, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(base_outputs, Mapping):
        for name in ("hidden_states", "last_hidden_state"):
            value = base_outputs.get(name)
            if isinstance(value, (tuple, list)) and value:
                value = value[-1]
            if isinstance(value, torch.Tensor):
                return value
    raise ValueError(
        "Could not extract final hidden states from base model output. "
        "Call the wrapper with output_hidden_states=True."
    )


def _unwrap_sample_boxes(bbox_targets: Any, sample_index: int) -> Any:
    """Read one sample's boxes from common collator representations."""

    if isinstance(bbox_targets, torch.Tensor):
        if bbox_targets.ndim == 3:
            return bbox_targets[sample_index]
        if bbox_targets.ndim == 2:
            # A [Q,4] tensor is only unambiguous for a batch of one.
            if sample_index != 0:
                raise ValueError("A [num_queries,4] bbox tensor is valid only for batch size 1")
            return bbox_targets
        raise ValueError(f"Unsupported bbox_targets tensor shape {tuple(bbox_targets.shape)}")
    if isinstance(bbox_targets, (tuple, list)):
        if sample_index >= len(bbox_targets):
            raise ValueError("bbox_targets batch length is smaller than input batch")
        sample = bbox_targets[sample_index]
        # Dataset/collator code sometimes wraps each tensor in a one-element
        # list to preserve variable-length fields.
        if isinstance(sample, (tuple, list)) and len(sample) == 1:
            sample = sample[0]
        return sample
    raise ValueError(f"Unsupported bbox_targets type {type(bbox_targets).__name__}")
