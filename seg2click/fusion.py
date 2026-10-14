"""Seg2Click residual concat-MLP fusion and backbone-specific contracts.

The shared math preserves parameter names and initialization order.  Named
backbone variants retain their original hidden-size checks, decoder defaults,
and checkpoint metadata.  The 0.8B/2B manifest keeps a fixed shape string;
the 4B/9B and Qwen3-VL manifests record actual dimensions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from seg2click.pixel_shuffle import Qwen35PixelShuffleMaskDecoder
from seg2click.mask_model import (
    Phase1Qwen35MaskWrapper,
    save_phase1_decoder as _save_phase1_decoder,
)


EXPECTED_QWEN35_4B_HIDDEN_SIZE = 2560

EXPECTED_QWEN35_9B_HIDDEN_SIZE = 4096

EXPECTED_QWEN3VL_2B_HIDDEN_SIZE = 2048


class VisionConcatMlpPixelShuffleMaskDecoder(Qwen35PixelShuffleMaskDecoder):
    """Fuse language-conditioned and post-merge vision tokens with an MLP."""

    def __init__(
        self,
        hidden_size: int,
        decoder_channels: int = 128,
        *,
        pixel_shuffle_factor: int = 32,
    ) -> None:
        super().__init__(
            hidden_size,
            decoder_channels=decoder_channels,
            pixel_shuffle_factor=pixel_shuffle_factor,
        )
        fusion_size = 2 * hidden_size
        self.vision_norm = nn.LayerNorm(hidden_size)
        self.vision_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_norm = nn.LayerNorm(fusion_size)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        output_layer = self.fusion_mlp[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self._vision_tokens: list[torch.Tensor] | None = None
        self.to(dtype=torch.float32)

    def set_vision_tokens(self, features: Any) -> None:
        if isinstance(features, torch.Tensor):
            features = [features]
        if not isinstance(features, (tuple, list)):
            raise TypeError(
                f"Vision features must be a tensor or sequence, got {type(features).__name__}"
            )
        self._vision_tokens = [torch.as_tensor(value) for value in features]

    def clear_vision_tokens(self) -> None:
        self._vision_tokens = None

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        if self._vision_tokens is None or not self._vision_tokens:
            raise RuntimeError("Vision features were not set before decoder forward")
        vision_tokens = self._vision_tokens.pop(0).to(
            device=visual_tokens.device,
            dtype=torch.float32,
        )
        if vision_tokens.shape != visual_tokens.shape:
            raise ValueError(
                "Vision/image-token feature shape mismatch: "
                f"vision={tuple(vision_tokens.shape)}, "
                f"image_tokens={tuple(visual_tokens.shape)}"
            )

        visual = visual_tokens.float()
        vision = self.vision_proj(self.vision_norm(vision_tokens))
        fusion_input = torch.cat((visual, vision), dim=-1)
        residual = self.fusion_mlp(self.fusion_norm(fusion_input))
        fused = visual + residual
        return super().forward(fused, query_tokens, grid_height, grid_width)


class _VisionFeatureWrapper(Phase1Qwen35MaskWrapper):
    """Cache one vision pass per batch and always release its feature list."""

    _vision_input_error = "Vision concat-MLP Phase 1 requires pixel_values and image_grid_thw"

    def _vision_model(self) -> Any:
        model = self.base_model
        getter = getattr(model, "get_base_model", None)
        if callable(getter):
            model = getter()
        return getattr(model, "model", model)

    def _cache_vision_features(self, vision_output: Any, image_grid_thw: torch.Tensor) -> None:
        self.mask_decoder.set_vision_tokens(vision_output.pooler_output)

    def forward(self, *args: Any, **kwargs: Any):
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(self._vision_input_error)
        vision_output = self._vision_model().get_image_features(
            pixel_values,
            image_grid_thw,
            return_dict=True,
        )
        self._cache_vision_features(vision_output, image_grid_thw)
        try:
            return super().forward(*args, **kwargs)
        finally:
            self.mask_decoder.clear_vision_tokens()


class VisionConcatMlpPhase1Qwen35MaskWrapper(_VisionFeatureWrapper):
    """Combine final LLM image states with post-merge vision features."""

    decoder_type = VisionConcatMlpPixelShuffleMaskDecoder
    expected_hidden_size: Optional[int] = None
    backbone_variant = "Qwen3.5"

    def __init__(
        self,
        base_model: nn.Module,
        decoder_channels: int = 128,
        *,
        pixel_shuffle_factor: int = 32,
        image_token_id: Optional[int] = None,
        pointer_pad_token_id: Optional[int] = None,
    ) -> None:
        # Keep the base decoder construction before replacement: it consumes
        # RNG state, so removing it would change initialization for a given seed.
        super().__init__(
            base_model,
            decoder_channels=decoder_channels,
            pixel_shuffle_factor=pixel_shuffle_factor,
            image_token_id=image_token_id,
            pointer_pad_token_id=pointer_pad_token_id,
        )
        hidden_size = int(self.mask_decoder.hidden_size)
        if self.expected_hidden_size is not None and hidden_size != self.expected_hidden_size:
            raise ValueError(
                f"The {self.backbone_variant} backbone must expose hidden_size={self.expected_hidden_size}, "
                f"got {hidden_size}"
            )
        self.mask_decoder = self.decoder_type(
            self.mask_decoder.hidden_size,
            decoder_channels=decoder_channels,
            pixel_shuffle_factor=pixel_shuffle_factor,
        )


class _BackboneConcatMlpDecoder(VisionConcatMlpPixelShuffleMaskDecoder):
    """Keep the C=1024 default and explicit backbone width validation."""

    def __init__(
        self,
        hidden_size: int,
        decoder_channels: int = 1024,
        *,
        pixel_shuffle_factor: int = 32,
    ) -> None:
        if int(hidden_size) != self.expected_hidden_size:
            raise ValueError(
                f"The {self.backbone_variant} Seg2Click decoder requires hidden_size={self.expected_hidden_size}, "
                f"got {hidden_size}"
            )
        super().__init__(
            hidden_size,
            decoder_channels=decoder_channels,
            pixel_shuffle_factor=pixel_shuffle_factor,
        )


class _BackboneConcatMlpWrapper(VisionConcatMlpPhase1Qwen35MaskWrapper):
    """Use the original C=1024 default for backbone-specific experiments."""

    def __init__(
        self,
        base_model: nn.Module,
        decoder_channels: int = 1024,
        *,
        pixel_shuffle_factor: int = 32,
        image_token_id: Optional[int] = None,
        pointer_pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__(
            base_model,
            decoder_channels=decoder_channels,
            pixel_shuffle_factor=pixel_shuffle_factor,
            image_token_id=image_token_id,
            pointer_pad_token_id=pointer_pad_token_id,
        )


class Qwen35FourBVisionConcatMlpPixelShuffleMaskDecoder(_BackboneConcatMlpDecoder):
    """Residual concat-MLP decoder for Qwen3.5-4B."""

    expected_hidden_size = EXPECTED_QWEN35_4B_HIDDEN_SIZE
    backbone_variant = "Qwen3.5-4B"


class Qwen35FourBVisionConcatMlpMaskWrapper(_BackboneConcatMlpWrapper):
    """Mask wrapper preserving the Qwen3.5-4B feature contract."""

    decoder_type = Qwen35FourBVisionConcatMlpPixelShuffleMaskDecoder
    expected_hidden_size = EXPECTED_QWEN35_4B_HIDDEN_SIZE
    backbone_variant = "Qwen3.5-4B"


class Qwen35NineBVisionConcatMlpPixelShuffleMaskDecoder(_BackboneConcatMlpDecoder):
    """Residual concat-MLP decoder for Qwen3.5-9B."""

    expected_hidden_size = EXPECTED_QWEN35_9B_HIDDEN_SIZE
    backbone_variant = "Qwen3.5-9B"


class Qwen35NineBVisionConcatMlpMaskWrapper(_BackboneConcatMlpWrapper):
    """Mask wrapper preserving the Qwen3.5-9B feature contract."""

    decoder_type = Qwen35NineBVisionConcatMlpPixelShuffleMaskDecoder
    expected_hidden_size = EXPECTED_QWEN35_9B_HIDDEN_SIZE
    backbone_variant = "Qwen3.5-9B"


class Qwen3VLTwoBVisionConcatMlpPixelShuffleMaskDecoder(_BackboneConcatMlpDecoder):
    """Residual concat-MLP decoder for Qwen3-VL-2B-Instruct."""

    expected_hidden_size = EXPECTED_QWEN3VL_2B_HIDDEN_SIZE
    backbone_variant = "Qwen3-VL-2B-Instruct"


class Qwen3VLTwoBVisionConcatMlpMaskWrapper(_BackboneConcatMlpWrapper):
    """Mask wrapper preserving the Qwen3-VL-2B-Instruct feature contract."""

    decoder_type = Qwen3VLTwoBVisionConcatMlpPixelShuffleMaskDecoder
    expected_hidden_size = EXPECTED_QWEN3VL_2B_HIDDEN_SIZE
    backbone_variant = "Qwen3-VL-2B-Instruct"

    def _cache_vision_features(self, vision_output: Any, image_grid_thw: torch.Tensor) -> None:
        deepstack = getattr(vision_output, "deepstack_features", None)
        if not isinstance(deepstack, (tuple, list)) or len(deepstack) != 3:
            raise RuntimeError("Qwen3-VL-2B must return three native DeepStack feature levels")
        super()._cache_vision_features(vision_output, image_grid_thw)


def _save_concat_decoder(
    model: nn.Module,
    output_dir: str | Path,
    *,
    expected_type: Optional[type] = None,
    backbone_variant: Optional[str] = None,
    native_deepstack: bool = False,
    **kwargs: Any,
) -> Path:
    if expected_type is not None:
        decoder = getattr(model, "mask_decoder", None)
        if not isinstance(decoder, expected_type):
            raise TypeError(f"Expected a {expected_type.__name__}, got {type(decoder).__name__}")
        hidden_size = int(decoder.hidden_size)
        fusion_size = 2 * hidden_size
        fusion_mlp = f"LayerNorm({fusion_size})-Linear({fusion_size},{hidden_size})-GELU-Linear({hidden_size},{hidden_size})"
    else:
        fusion_mlp = "LayerNorm(2048)-Linear(2048,1024)-GELU-Linear(1024,1024)"
    path = _save_phase1_decoder(model, output_dir, **kwargs)
    manifest_path = Path(output_dir) / "phase1_mask_decoder.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decoder_architecture"] = "vision_concat_mlp_dynamic_pixel_shuffle"
    if backbone_variant is not None:
        manifest["backbone_family"] = "qwen3_vl" if native_deepstack else "qwen3_5"
        manifest["backbone_variant"] = backbone_variant
    if native_deepstack:
        manifest["native_deepstack_visual_indexes"] = [5, 11, 17]
        manifest["native_deepstack_preserved"] = True
    manifest["vision_encoder_features"] = {
        "source": (
            "Qwen3-VL merged vision tower pooler_output"
            if native_deepstack
            else "Qwen3.5 merged vision tower pooler_output"
        ),
        "fusion": "residual concat MLP",
        "fusion_input": "LLM final image-token state concatenated with projected post-merge vision token",
        "fusion_mlp": fusion_mlp,
        "initialization": "final fusion linear initialized to zero",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def save_vision_concat_mlp_phase1_decoder(
    model: nn.Module,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Save the original 0.8B LoRA fusion manifest, including its shape string."""
    return _save_concat_decoder(model, output_dir, **kwargs)


def save_qwen35_4b_vision_concat_mlp_decoder(
    model: nn.Module,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Save Qwen3.5-4B fusion with its original type check and manifest."""
    return _save_concat_decoder(
        model,
        output_dir,
        expected_type=Qwen35FourBVisionConcatMlpPixelShuffleMaskDecoder,
        backbone_variant="Qwen3.5-4B",
        **kwargs,
    )


def save_qwen35_9b_vision_concat_mlp_decoder(
    model: nn.Module,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Save Qwen3.5-9B fusion with its original type check and manifest."""
    return _save_concat_decoder(
        model,
        output_dir,
        expected_type=Qwen35NineBVisionConcatMlpPixelShuffleMaskDecoder,
        backbone_variant="Qwen3.5-9B",
        **kwargs,
    )


def save_qwen3vl_2b_vision_concat_mlp_decoder(
    model: nn.Module,
    output_dir: str | Path,
    **kwargs: Any,
) -> Path:
    """Save Qwen3-VL-2B-Instruct fusion with its original type check and manifest."""
    return _save_concat_decoder(
        model,
        output_dir,
        expected_type=Qwen3VLTwoBVisionConcatMlpPixelShuffleMaskDecoder,
        backbone_variant="Qwen3-VL-2B-Instruct",
        native_deepstack=True,
        **kwargs,
    )


__all__ = [
    "VisionConcatMlpPhase1Qwen35MaskWrapper",
    "VisionConcatMlpPixelShuffleMaskDecoder",
    "save_vision_concat_mlp_phase1_decoder",
    "EXPECTED_QWEN35_4B_HIDDEN_SIZE",
    "Qwen35FourBVisionConcatMlpPixelShuffleMaskDecoder",
    "Qwen35FourBVisionConcatMlpMaskWrapper",
    "save_qwen35_4b_vision_concat_mlp_decoder",
    "EXPECTED_QWEN35_9B_HIDDEN_SIZE",
    "Qwen35NineBVisionConcatMlpPixelShuffleMaskDecoder",
    "Qwen35NineBVisionConcatMlpMaskWrapper",
    "save_qwen35_9b_vision_concat_mlp_decoder",
    "EXPECTED_QWEN3VL_2B_HIDDEN_SIZE",
    "Qwen3VLTwoBVisionConcatMlpPixelShuffleMaskDecoder",
    "Qwen3VLTwoBVisionConcatMlpMaskWrapper",
    "save_qwen3vl_2b_vision_concat_mlp_decoder",
]
