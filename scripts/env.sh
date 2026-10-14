# Shared environment for Seg2Click launch scripts. Source it from a script in
# scripts/<subdir>/ after `set -euo pipefail`; it changes to the repository root.
# Activate your Python environment before running any script.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
# Training and evaluation read local checkpoints and datasets only.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# Keep Hugging Face Hub kernelization off; Qwen3.5 uses the locally installed
# flash-linear-attention / causal-conv1d kernels through Transformers.
export USE_HUB_KERNELS=NO
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
NUM_GPUS="${NUM_GPUS:-8}"

# Keep compiler caches next to one run's outputs. Delete only these cache
# directories (never checkpoints or logs) before retrying a failed run.
setup_run_dir() {
  local output_dir="$1"
  mkdir -p "${output_dir}"
  export TRITON_CACHE_DIR="${output_dir}/triton_cache"
  export TILELANG_CACHE_DIR="${output_dir}/tilelang_cache"
  export TILELANG_TMP_DIR="${output_dir}/tilelang_tmp"
  export TORCHINDUCTOR_CACHE_DIR="${output_dir}/torchinductor_cache"
}

# Usage: resume_args=(); add_resume_args resume_args
add_resume_args() {
  local -n _target="$1"
  _target=()
  if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    _target=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  fi
}
