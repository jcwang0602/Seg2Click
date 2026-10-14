#!/bin/bash
# Train Seg2Click on Qwen3.5 with LoRA.
#
#   MODEL_SIZE   0.8b (default) | 2b | 4b | 9b
#   BATCH_SIZE / GRAD_ACCUM  per-GPU batch / accumulation; defaults
#                0.8b: 4/1, 2b: 2/2, 4b and 9b: 1/4 (effective 32 on 8 GPUs)
#   LR (1e-4), SEED (42), DECODER_CHANNELS (1024), NUM_GPUS (8)
#   BASE_MODEL   default checkpoints/Qwen3.5-<SIZE>
#   OUTPUT_DIR   default outputs/seg2click_qwen35_<SIZE>
#   RESUME_FROM_CHECKPOINT  optional checkpoint-* inside OUTPUT_DIR
#
# Examples:
#   bash scripts/train/qwen35_seg2click.sh
#   MODEL_SIZE=9b bash scripts/train/qwen35_seg2click.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"

model_size="${MODEL_SIZE:-0.8b}"
case "${model_size}" in
  0.8b) default_batch=4; default_accum=1 ;;
  2b)   default_batch=2; default_accum=2 ;;
  4b|9b) default_batch=1; default_accum=4 ;;
  *) echo "Unsupported MODEL_SIZE: ${model_size}" >&2; exit 2 ;;
esac
gradient_checkpointing=False
[[ "${model_size}" == 9b ]] && gradient_checkpointing=True

base_model="${BASE_MODEL:-checkpoints/Qwen3.5-${model_size^^}}"
output_dir="${OUTPUT_DIR:-outputs/seg2click_qwen35_${model_size}}"
setup_run_dir "${output_dir}"
add_resume_args resume_args

torchrun --nproc_per_node="${NUM_GPUS}" training/train_qwen35.py \
  --model_size "${model_size}" \
  --model_name_or_path "${base_model}" \
  --data_path data/data_config.yaml --image_folder "" \
  --output_dir "${output_dir}" \
  --seed "${SEED:-42}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size "${BATCH_SIZE:-${default_batch}}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-${default_accum}}" \
  --learning_rate "${LR:-1e-4}" --weight_decay 0. --lr_scheduler_type cosine \
  --save_steps 2000 --save_total_limit 2 --logging_steps 10 \
  --dataloader_num_workers 8 --model_max_length 24576 --max_pixels 5720064 \
  --decoder_channels "${DECODER_CHANNELS:-1024}" --pixel_shuffle_factor 32 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --bf16 True --tf32 True --gradient_checkpointing "${gradient_checkpointing}" \
  "${resume_args[@]}" 2>&1 | tee -a "${output_dir}/train.log"
