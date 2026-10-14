#!/bin/bash
# Evaluate a Seg2Click checkpoint on ScreenSpot, ScreenSpot-v2 and ScreenSpot-Pro.
# The three benchmarks run as three independent processes, one per GPU.
#
#   CHECKPOINT   Seg2Click checkpoint directory (required)
#   OUTPUT_DIR   default ${CHECKPOINT}/evaluations
#   GPUS         three comma-separated GPU ids (default 0,1,2)
#   BENCHMARKS   default "screenspot screenspot_v2 screenspot_pro"
#   BATCH_SIZE   default 1; MAX_SAMPLES default 0 (= full benchmark)
#
# Example:
#   CHECKPOINT=checkpoints/Seg2Click-Qwen3.5-0.8B bash scripts/eval/evaluate_seg2click.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../env.sh"

: "${CHECKPOINT:?Set CHECKPOINT to a Seg2Click checkpoint directory}"
output_dir="${OUTPUT_DIR:-${CHECKPOINT}/evaluations}"
read -r -a benchmarks <<< "${BENCHMARKS:-screenspot screenspot_v2 screenspot_pro}"
IFS=, read -r -a gpus <<< "${GPUS:-0,1,2}"
mkdir -p "${output_dir}"

pids=()
for i in "${!benchmarks[@]}"; do
  benchmark="${benchmarks[$i]}"
  CUDA_VISIBLE_DEVICES="${gpus[$((i % ${#gpus[@]}))]}" python -u eval/evaluate_seg2click.py \
    --checkpoint "${CHECKPOINT}" --benchmark "${benchmark}" \
    --data_root data/eval --output_dir "${output_dir}" \
    --batch_size "${BATCH_SIZE:-1}" --max_samples "${MAX_SAMPLES:-0}" \
    > "${output_dir}/${benchmark}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
for benchmark in "${benchmarks[@]}"; do tail -n 6 "${output_dir}/${benchmark}.log"; done
exit "${status}"
