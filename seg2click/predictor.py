"""Seg2Click inference: load a trained checkpoint and predict a click mask.

A Seg2Click checkpoint directory contains the merged backbone
(``config.json`` / ``model.safetensors``), its tokenizer and processor, and
the FP32 mask decoder (``phase1_mask_decoder.json`` / ``.safetensors``)::

    from seg2click.predictor import Seg2ClickPredictor

    predictor = Seg2ClickPredictor.from_pretrained("path/to/checkpoint")
    result = predictor.predict(image, "close this window")
    x, y = result.point  # normalized to the original screenshot

The mask is predicted at the processed-image resolution.  Because the
processor only resizes (it never crops or pads), normalized mask coordinates
are also normalized coordinates on the original screenshot.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from seg2click.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    chat_template,
    grounding_system_message,
)

DECODER_MANIFEST = "phase1_mask_decoder.json"
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 5720064
ASSISTANT_PREFIX = (
    "<|im_start|>assistant<|recipient|>os\npyautogui.click("
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN})"
)
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

SEG2CLICK_DECODER = "vision_concat_mlp_dynamic_pixel_shuffle"


def _model_class(model_type: str):
    if model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration
    if model_type == "qwen3_vl":
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    raise ValueError(f"Unsupported backbone model_type: {model_type!r}")


def build_decoder(manifest: dict[str, Any], model_type: str) -> torch.nn.Module:
    """Construct the (untrained) mask decoder described by a checkpoint manifest."""

    if manifest.get("decoder_architecture") != SEG2CLICK_DECODER:
        raise ValueError(f"Not a Seg2Click decoder manifest: {manifest.get('decoder_architecture')!r}")
    hidden = int(manifest["hidden_size"])
    channels = int(manifest["decoder_channels"])
    factor = int(manifest["pixel_shuffle_factor"])
    if model_type in {"qwen2_vl", "qwen2_5_vl"}:
        from seg2click.qwen_vl_decoder import Seg2ClickQwen25VLDecoder

        return Seg2ClickQwen25VLDecoder(hidden, channels, pixel_shuffle_factor=factor)
    from seg2click.fusion import VisionConcatMlpPixelShuffleMaskDecoder

    return VisionConcatMlpPixelShuffleMaskDecoder(hidden, channels, pixel_shuffle_factor=factor)


def mask_to_point_centroid(
    logits: torch.Tensor, *, threshold: float = 0.5
) -> tuple[float, float, str]:
    """Centroid of the largest 4-connected component of ``sigmoid(logits) >= threshold``.

    Ties between equally large components go to the one containing the
    smallest row-major pixel.  An empty foreground falls back to the
    probability-weighted centroid, then to the image centre.
    """

    from scipy import ndimage

    field = torch.as_tensor(logits).detach().float().cpu()
    if field.ndim != 2 or not torch.isfinite(field).all():
        raise ValueError("mask logits must be a finite [height, width] tensor")
    probabilities = field.sigmoid()
    labels, count = ndimage.label((probabilities >= threshold).numpy(), structure=_FOUR_CONNECTED)
    height, width = field.shape
    if count:
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        sizes[0] = 0
        largest = np.flatnonzero(sizes == sizes.max())
        first = np.flatnonzero(np.isin(labels.ravel(), largest))[0]
        ys, xs = np.nonzero(labels == labels.ravel()[first])
        return float((xs.mean() + 0.5) / width), float((ys.mean() + 0.5) / height), "component_centroid"
    mass = probabilities.clamp_min(0)
    if float(mass.sum()) > 0:
        xs = torch.arange(width, dtype=torch.float32)
        ys = torch.arange(height, dtype=torch.float32)
        center_x = (mass * xs[None, :]).sum() / mass.sum()
        center_y = (mass * ys[:, None]).sum() / mass.sum()
        return float((center_x + 0.5) / width), float((center_y + 0.5) / height), "weighted_centroid"
    return 0.5, 0.5, "image_center"


def build_inputs(processor, image: Image.Image, instruction: str, *, patch_size: int):
    """Tokenize one screenshot/instruction pair with the training prompt."""

    from qwen_vl_utils import process_vision_info

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=chat_template
    )
    images, videos = process_vision_info(conversation, image_patch_size=patch_size)
    return processor(text=[text + ASSISTANT_PREFIX], images=images, videos=videos, padding=True, return_tensors="pt")


@dataclass
class Prediction:
    point: tuple[float, float]
    """Normalized ``(x, y)`` click on the original screenshot."""
    logits: torch.Tensor
    """``[H, W]`` mask logits at the processed-image resolution (CPU, float32)."""
    decode_reason: str

    @property
    def probability(self) -> torch.Tensor:
        return self.logits.sigmoid()


class Seg2ClickPredictor:
    """Backbone + mask decoder wrapper for single-image click prediction."""

    def __init__(self, model, decoder, tokenizer, processor, manifest: dict[str, Any], device: torch.device):
        self.model = model
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.manifest = manifest
        self.device = device
        self.patch_size = int(processor.image_processor.patch_size)
        self.merge_size = int(model.config.vision_config.spatial_merge_size)
        self.image_token_id = int(model.config.image_token_id)
        self.video_token_id = getattr(model.config, "video_token_id", None)
        self.pointer_pad_id = int(tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN))

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device | None = None,
        attn_implementation: Optional[str] = None,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Seg2ClickPredictor":
        from safetensors.torch import load_file
        from transformers import AutoProcessor, AutoTokenizer

        checkpoint = Path(checkpoint).resolve()
        device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        manifest = json.loads((checkpoint / DECODER_MANIFEST).read_text(encoding="utf-8"))
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        model_type = config["model_type"]
        if attn_implementation is None:
            attn_implementation = "sdpa"
            if model_type in {"qwen2_vl", "qwen2_5_vl"} and device.type == "cuda":
                try:
                    import flash_attn  # noqa: F401

                    attn_implementation = "flash_attention_2"
                except ImportError:
                    pass

        # Training saves the LoRA adapter next to the merged weights.  Expose
        # only the merged files so Transformers does not re-apply the adapter.
        with tempfile.TemporaryDirectory(prefix="seg2click-load-") as temporary:
            load_dir = Path(temporary)
            for path in checkpoint.iterdir():
                if path.name in {"config.json", "generation_config.json"} or (
                    path.name.startswith("model") and path.suffix in {".safetensors", ".json"}
                ):
                    (load_dir / path.name).symlink_to(path)
            model, loading_info = _model_class(model_type).from_pretrained(
                load_dir,
                dtype=dtype,
                attn_implementation=attn_implementation,
                device_map={"": str(device)},
                output_loading_info=True,
            )
        errors = {key: value for key, value in loading_info.items() if value}
        if errors:
            raise RuntimeError(f"Incomplete checkpoint loading: {errors}")
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
        for token in ADDITIONAL_SPECIAL_TOKENS[2:]:
            if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
                raise ValueError(f"Checkpoint tokenizer is missing the Seg2Click token {token}")
        if tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN) != model.config.pointer_pad_token_id:
            raise ValueError("Tokenizer and model config disagree on the pointer token id")
        processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)
        processor.tokenizer = tokenizer

        if int(manifest["hidden_size"]) != int(getattr(model.config, "text_config", model.config).hidden_size):
            raise ValueError("Decoder hidden size differs from the backbone")
        decoder = build_decoder(manifest, model_type)
        decoder.load_state_dict(load_file(str(checkpoint / manifest["weights"])), strict=True)
        decoder.to(device=device, dtype=torch.float32).eval()
        return cls(model, decoder, tokenizer, processor, manifest, device)

    @torch.inference_mode()
    def predict_batch(self, images: Sequence[Image.Image], instructions: Sequence[str]) -> list[Prediction]:
        """Predict one mask per (image, instruction); images may differ in size."""

        prepared = [
            build_inputs(self.processor, image.convert("RGB"), text, patch_size=self.patch_size)
            for image, text in zip(images, instructions)
        ]
        # Right padding keeps every real token's causal context unchanged.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"][0] for item in prepared], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"][0] for item in prepared], batch_first=True, padding_value=0
        )
        token_types = torch.zeros_like(input_ids, dtype=torch.int32)
        token_types.masked_fill_(input_ids == self.image_token_id, 1)
        if self.video_token_id is not None:
            token_types.masked_fill_(input_ids == int(self.video_token_id), 2)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": token_types,
            "pixel_values": torch.cat([item["pixel_values"] for item in prepared]),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in prepared]),
        }
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Capture the vision tower output of the single backbone forward pass.
        captured: list[Any] = []
        handle = self.model.model.visual.register_forward_hook(lambda _m, _a, output: captured.append(output))
        try:
            outputs = self.model.model(**inputs, return_dict=True, use_cache=False)
        finally:
            handle.remove()
        hidden = outputs.last_hidden_state
        grids = inputs["image_grid_thw"].tolist()
        # get_image_features replaces pooler_output with per-image splits on
        # the captured object after the hook has run.
        vision = captured[0].pooler_output
        if isinstance(vision, torch.Tensor):
            vision = vision.split([t * h * w // self.merge_size**2 for t, h, w in grids])

        results = []
        for index, (_, grid_h, grid_w) in enumerate(grids):
            ids = inputs["input_ids"][index]
            visual = torch.nonzero(ids == self.image_token_id).flatten()
            query = torch.nonzero(ids == self.pointer_pad_id).flatten()
            height, width = grid_h // self.merge_size, grid_w // self.merge_size
            if visual.numel() != height * width or query.numel() != 1:
                raise RuntimeError("Image-token/grid or pointer-token mismatch")
            self.decoder.set_vision_tokens([vision[index]])
            try:
                logits = self.decoder(hidden[index, visual].float(), hidden[index, query].float(), height, width)
            finally:
                self.decoder.clear_vision_tokens()
            field = logits[0].float().cpu()
            x, y, reason = mask_to_point_centroid(field)
            results.append(Prediction((x, y), field, reason))
        return results

    def predict(self, image: Image.Image, instruction: str) -> Prediction:
        return self.predict_batch([image], [instruction])[0]


__all__ = [
    "Prediction",
    "Seg2ClickPredictor",
    "build_decoder",
    "build_inputs",
    "mask_to_point_centroid",
]
