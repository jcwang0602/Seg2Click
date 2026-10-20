---
title: Seg2Click Demo
emoji: 🖱️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
python_version: "3.12"
short_description: Pixel-level action masks for GUI visual grounding
startup_duration_timeout: 30m
---

# Seg2Click Demo

[Seg2Click](https://github.com/jcwang0602/Seg2Click) turns a GUI screenshot and a natural-language
instruction into a pixel-level executable action mask and click point. This Space runs the public
[Seg2Click-Qwen3.5-9B](https://huggingface.co/jcwang0602/Seg2Click-Qwen3.5-9B) checkpoint with the official
inference path.

Seg2Click-9B achieves **92.1 / 94.1 / 55.2** Hit@1 on ScreenSpot / ScreenSpot-v2 / ScreenSpot-Pro.

Read the [paper](https://arxiv.org/abs/2610.16234) for method and evaluation details.
