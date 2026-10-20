---
title: Seg2Click
emoji: 🖱️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Pixel-level action masks for GUI visual grounding
startup_duration_timeout: 30m
suggested_hardware: zero-a10g
---

# Seg2Click

[Seg2Click](https://github.com/jcwang0602/Seg2Click) turns a GUI screenshot and a natural-language
instruction into a pixel-level executable action mask and click point. This Space runs the lightweight public
[Seg2Click-Qwen3.5-0.8B](https://huggingface.co/jcwang0602/Seg2Click-Qwen3.5-0.8B) checkpoint with the official
inference path.

Seg2Click-0.8B achieves **87.4 / 89.9 / 44.3** Hit@1 on ScreenSpot / ScreenSpot-v2 / ScreenSpot-Pro.

The example screenshots come from [ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
and are evaluated with fresh model inference whenever an example is clicked.

Read the [paper](https://arxiv.org/abs/2610.16234) for method and evaluation details.
