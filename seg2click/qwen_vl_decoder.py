"""Shared Qwen2 / Qwen2.5-VL mask decoders, including the 7B variant.

The model families use the same FP32 residual fusion and dynamic pixel shuffle.
Their explicit class names remain available for existing model artifacts.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PIXEL_SHUFFLE_FACTOR = 28
DEFAULT_DECODER_CHANNELS = 1024


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar-like, got shape {tuple(value.shape)}")
        value = value.detach().cpu().item()
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False


class Qwen25VLPixelShuffleDecoder(nn.Module):
    """Query-conditioned mask decoder that outputs the processed image size.

    For the official Qwen2.5-VL configuration, one post-merge visual cell maps to
    ``14 * 2 = 28`` processed pixels in each direction.  The output head
    therefore predicts ``28**2`` sub-pixel channels and ``pixel_shuffle``
    rearranges them into ``[num_queries, H', W']`` logits.
    """

    def __init__(
        self,
        hidden_size: int,
        decoder_channels: int = DEFAULT_DECODER_CHANNELS,
        *,
        pixel_shuffle_factor: int = DEFAULT_PIXEL_SHUFFLE_FACTOR,
    ) -> None:
        super().__init__()
        self.hidden_size = _as_positive_int(hidden_size, "hidden_size")
        self.decoder_channels = _as_positive_int(decoder_channels, "decoder_channels")
        self.pixel_shuffle_factor = _as_positive_int(pixel_shuffle_factor, "pixel_shuffle_factor")
        if self.decoder_channels < 8:
            raise ValueError("decoder_channels must be at least 8")

        group_count = min(32, self.decoder_channels)
        while self.decoder_channels % group_count:
            group_count -= 1

        self.visual_norm = nn.LayerNorm(self.hidden_size)
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.visual_proj = nn.Linear(self.hidden_size, self.decoder_channels)
        self.query_scale = nn.Linear(self.hidden_size, self.decoder_channels)
        self.query_bias = nn.Linear(self.hidden_size, self.decoder_channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(self.decoder_channels, self.decoder_channels, 3, padding=1),
            nn.GroupNorm(group_count, self.decoder_channels),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.decoder_channels, self.decoder_channels, 3, padding=1),
            nn.GroupNorm(group_count, self.decoder_channels),
            nn.GELU(),
        )
        self.output = nn.Conv2d(
            self.decoder_channels,
            self.pixel_shuffle_factor * self.pixel_shuffle_factor,
            1,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                    if module.weight is not None:
                        nn.init.ones_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.to(dtype=torch.float32)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 2 or visual_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                "visual_tokens must have shape [grid_height*grid_width, hidden_size], "
                f"got {tuple(visual_tokens.shape)}"
            )
        if query_tokens.ndim != 2 or query_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                "query_tokens must have shape [num_queries, hidden_size], "
                f"got {tuple(query_tokens.shape)}"
            )
        height = _as_positive_int(grid_height, "grid_height")
        width = _as_positive_int(grid_width, "grid_width")
        if visual_tokens.shape[0] != height * width:
            raise ValueError(
                "visual token count does not match grid: "
                f"{visual_tokens.shape[0]} != {height}*{width}"
            )
        if query_tokens.shape[0] == 0:
            raise ValueError("At least one query token is required")

        device_type = visual_tokens.device.type
        autocast_enabled = device_type in {"cuda", "cpu", "xpu", "hpu", "mps"}
        context = (
            torch.autocast(device_type=device_type, enabled=False)
            if autocast_enabled
            else _NullContext()
        )
        with context:
            visual_tokens = visual_tokens.to(dtype=torch.float32)
            query_tokens = query_tokens.to(dtype=torch.float32)
            if not torch.isfinite(visual_tokens).all():
                raise FloatingPointError("Non-finite visual features passed to mask decoder")
            if not torch.isfinite(query_tokens).all():
                raise FloatingPointError("Non-finite query features passed to mask decoder")

            visual = self.visual_proj(self.visual_norm(visual_tokens))
            visual = visual.transpose(0, 1).reshape(1, self.decoder_channels, height, width)
            queries = self.query_norm(query_tokens)
            scale = torch.tanh(self.query_scale(queries))[:, :, None, None]
            bias = self.query_bias(queries)[:, :, None, None]
            features = visual * (1.0 + scale) + bias
            features = self.block1(features)
            features = self.block2(features)
            subpixel_logits = self.output(features)
            logits = F.pixel_shuffle(subpixel_logits, self.pixel_shuffle_factor).squeeze(1).float()

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite mask decoder logits")
        return logits


class _ResidualVisionFusion:
    """Fuse final LLM image states and post-merge vision tokens with a residual MLP."""

    def __init__(
        self,
        hidden_size: int,
        decoder_channels: int = 1024,
        *,
        pixel_shuffle_factor: int = 28,
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

        with torch.autocast(device_type=visual_tokens.device.type, enabled=False):
            visual = visual_tokens.float()
            vision = self.vision_proj(self.vision_norm(vision_tokens))
            fusion_input = torch.cat((visual, vision), dim=-1)
            residual = self.fusion_mlp(self.fusion_norm(fusion_input))
            fused = visual + residual
            return super().forward(fused, query_tokens, grid_height, grid_width)


class Qwen2VL2BPixelShuffleDecoder(Qwen25VLPixelShuffleDecoder):
    """Qwen2-VL spelling of the shared dynamic pixel-shuffle decoder."""


class Seg2ClickQwen25VLDecoder(_ResidualVisionFusion, Qwen25VLPixelShuffleDecoder):
    """Residual vision fusion used by Qwen2.5-VL (3B and 7B)."""


class Seg2ClickQwen2VL2BDecoder(_ResidualVisionFusion, Qwen2VL2BPixelShuffleDecoder):
    """Residual vision fusion used by Qwen2-VL-2B."""


__all__ = [
    "DEFAULT_PIXEL_SHUFFLE_FACTOR",
    "DEFAULT_DECODER_CHANNELS",
    "Qwen25VLPixelShuffleDecoder",
    "Qwen2VL2BPixelShuffleDecoder",
    "Seg2ClickQwen25VLDecoder",
    "Seg2ClickQwen2VL2BDecoder",
]
