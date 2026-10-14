#!/bin/bash
# Seg2Click on Qwen2-VL-2B-Instruct with LoRA (FlashAttention 2 required).
# Vision-tower gradient checkpointing is off; LLM checkpointing is on.
#
#   BATCH_SIZE (2), GRAD_ACCUM (2), LR (1e-4), SEED (42), DECODER_CHANNELS (1024)
#   NUM_GPUS (8), BASE_MODEL (checkpoints/Qwen2-VL-2B-Instruct)
#   OUTPUT_DIR (outputs/seg2click_qwen2vl_2b)
#   RESUME_FROM_CHECKPOINT  optional checkpoint-* inside OUTPUT_DIR
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"

base_model="${BASE_MODEL:-checkpoints/Qwen2-VL-2B-Instruct}"
output_dir="${OUTPUT_DIR:-outputs/seg2click_qwen2vl_2b}"
setup_run_dir "${output_dir}"
add_resume_args resume_args

torchrun --nproc_per_node="${NUM_GPUS}" training/train_qwen2vl.py \
  --model_name_or_path "${base_model}" \
  --data_path data/data_config.yaml --image_folder "" \
  --output_dir "${output_dir}" \
  --seed "${SEED:-42}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-2}" \
  --learning_rate "${LR:-1e-4}" --weight_decay 0. --lr_scheduler_type cosine \
  --save_steps 2000 --save_total_limit 2 --logging_steps 10 \
  --dataloader_num_workers 8 --model_max_length 24576 --max_pixels 5720064 \
  --decoder_channels "${DECODER_CHANNELS:-1024}" --pixel_shuffle_factor 28 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bf16 True --tf32 True \
  --vision_gradient_checkpointing False --llm_gradient_checkpointing True \
  "${resume_args[@]}" 2>&1 | tee -a "${output_dir}/train.log"
