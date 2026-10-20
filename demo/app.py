"""Interactive Seg2Click demo for Hugging Face Spaces."""

# ruff: noqa: I001 -- spaces must be imported before torch and its transitive importers.

import spaces  # Import before torch so ZeroGPU can configure the runtime.

import time

import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw

from seg2click.predictor import Seg2ClickPredictor

MODEL_ID = "jcwang0602/Seg2Click-Qwen3.5-9B"
MASK_THRESHOLD = 0.5

print(f"Downloading {MODEL_ID} ...", flush=True)
checkpoint_dir = snapshot_download(MODEL_ID)
print(f"Loading model from {checkpoint_dir} ...", flush=True)
predictor = Seg2ClickPredictor.from_pretrained(
    checkpoint_dir,
    device="cuda",
    attn_implementation="sdpa",
    dtype=torch.bfloat16,
)
print("Seg2Click is ready.", flush=True)


def render_overlay(image: Image.Image, probability: np.ndarray, x: float, y: float) -> Image.Image:
    """Overlay the action-mask heatmap and predicted click on a screenshot."""
    canvas = image.convert("RGB")
    width, height = canvas.size
    probability_image = Image.fromarray(
        (np.clip(probability, 0, 1) * 255).astype(np.uint8), mode="L"
    ).resize((width, height), resample=Image.Resampling.BILINEAR)
    values = np.asarray(probability_image, dtype=np.float32) / 255.0

    heatmap = np.stack(
        [
            np.clip(1.5 - np.abs(4.0 * values - 3.0), 0, 1),
            np.clip(1.5 - np.abs(4.0 * values - 2.0), 0, 1),
            np.clip(1.5 - np.abs(4.0 * values - 1.0), 0, 1),
        ],
        axis=-1,
    )
    base = np.asarray(canvas, dtype=np.float32)
    blended = base * (1 - 0.55 * values[..., None]) + heatmap * 255 * (0.55 * values[..., None])
    foreground = (values >= MASK_THRESHOLD).astype(np.float32)
    blended *= 0.75 + 0.25 * foreground[..., None]
    canvas = Image.fromarray(blended.astype(np.uint8), mode="RGB")

    px = round(x * width)
    py = round(y * height)
    marker = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(marker)
    draw.ellipse((px - 14, py - 14, px + 14, py + 14), fill=(255, 0, 0, 190))
    draw.ellipse((px - 26, py - 26, px + 26, py + 26), outline=(255, 215, 0, 255), width=5)
    return Image.alpha_composite(canvas.convert("RGBA"), marker).convert("RGB")


@spaces.GPU(duration=30)
def predict_click(screenshot: Image.Image, instruction: str):
    """Predict an action mask and executable click point."""
    if screenshot is None:
        raise gr.Error("Please upload a screenshot first.")
    instruction = (instruction or "").strip()
    if not instruction:
        raise gr.Error("Please enter an instruction, such as 'click the Submit button'.")

    image = screenshot.convert("RGB")
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = predictor.predict(image, instruction)
    elapsed = time.perf_counter() - started

    x, y = prediction.point
    width, height = image.size
    overlay = render_overlay(image, prediction.probability.numpy(), x, y)
    report = (
        f"**Click position:** `({x:.4f}, {y:.4f})` (normalized)  \n"
        f"**Pixel position:** `({round(x * width)}, {round(y * height)})`  \n"
        f"**Decode:** `{prediction.decode_reason}`  \n"
        f"**Inference time:** `{elapsed:.1f}s`"
    )
    return overlay, report


CSS = "#app-container { max-width: 1200px; margin: 0 auto; }"

with gr.Blocks(title="Seg2Click — GUI Visual Grounding", css=CSS) as demo:
    gr.Markdown(
        """
        # 🖱️ Seg2Click: Pixel-Level Action Masks for GUI Visual Grounding

        Upload a GUI screenshot and describe what you want to click. Seg2Click predicts the complete
        executable region as a pixel-level action mask, then returns its centroid as the click point.

        This demo runs [Seg2Click-Qwen3.5-9B](https://huggingface.co/jcwang0602/Seg2Click-Qwen3.5-9B).
        [Paper](https://arxiv.org/abs/2610.16234) · [Code](https://github.com/jcwang0602/Seg2Click)
        """
    )
    with gr.Row(elem_id="app-container"):
        with gr.Column():
            screenshot_input = gr.Image(
                type="pil",
                label="GUI screenshot",
                height=440,
                sources=["upload", "clipboard"],
            )
            instruction_input = gr.Textbox(
                label="Instruction",
                placeholder="e.g. click the refresh button in the sidebar",
            )
            predict_button = gr.Button("Predict click", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Action mask and click point", height=440)
            output_report = gr.Markdown()

    predict_button.click(
        predict_click,
        inputs=[screenshot_input, instruction_input],
        outputs=[output_image, output_report],
        api_name="predict_click",
    )
    instruction_input.submit(
        predict_click,
        inputs=[screenshot_input, instruction_input],
        outputs=[output_image, output_report],
    )

demo.launch(theme=gr.themes.Citrus())
