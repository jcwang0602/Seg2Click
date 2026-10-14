#!/bin/bash
# Seg2Click on Qwen2.5-VL-3B/7B-Instruct with LoRA (FlashAttention 2 required).
# Gradient checkpointing is controlled separately for the vision tower and LLM:
# 3B uses vision off / LLM on; 7B uses both on to fit in 80 GB.
#
#   MODEL_SIZE (3b | 7b, default 3b)
#   BATCH_SIZE (1), GRAD_ACCUM (4), LR (1e-4), SEED (42), DECODER_CHANNELS (1024)
#   VISION_GC (False for 3b, True for 7b), NUM_GPUS (8)
#   BASE_MODEL (checkpoints/Qwen2.5-VL-<SIZE>-Instruct)
#   OUTPUT_DIR (outputs/seg2click_qwen25vl_<SIZE>)
#   RESUME_FROM_CHECKPOINT  optional checkpoint-* inside OUTPUT_DIR
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"

model_size="${MODEL_SIZE:-3b}"
case "${model_size}" in
  3b) default_vision_gc=False ;;
  7b) default_vision_gc=True; export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" ;;
  *) echo "Unsupported MODEL_SIZE: ${model_size}" >&2; exit 2 ;;
esac
base_model="${BASE_MODEL:-checkpoints/Qwen2.5-VL-${model_size^^}-Instruct}"
output_dir="${OUTPUT_DIR:-outputs/seg2click_qwen25vl_${model_size}}"
setup_run_dir "${output_dir}"
add_resume_args resume_args

torchrun --nproc_per_node="${NUM_GPUS}" training/train_qwen25vl.py \
  --model_size "${model_size}" \
  --checkpointing_mode split \
  --attn_implementation flash_attention_2 \
  --model_name_or_path "${base_model}" \
  --data_path data/data_config.yaml --image_folder "" \
  --output_dir "${output_dir}" \
  --seed "${SEED:-42}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --learning_rate "${LR:-1e-4}" --weight_decay 0. --lr_scheduler_type cosine \
  --save_steps 2000 --save_total_limit 2 --logging_steps 10 \
  --dataloader_num_workers 8 --model_max_length 24576 --max_pixels 5720064 \
  --decoder_channels "${DECODER_CHANNELS:-1024}" --pixel_shuffle_factor 28 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bf16 True --tf32 True \
  --vision_gradient_checkpointing "${VISION_GC:-${default_vision_gc}}" \
  --llm_gradient_checkpointing True \
  "${resume_args[@]}" 2>&1 | tee -a "${output_dir}/train.log"
