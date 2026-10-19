#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export AQA_CPU_THREADS="${AQA_CPU_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$AQA_CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$AQA_CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export USE_HF=1 ENABLE_AUDIO_OUTPUT=0 PYTHONUNBUFFERED=1
aqa_python="${AQA_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$aqa_python" ]]; then
  echo 'Python environment missing. Run bash setup_server.sh first, or set AQA_PYTHON.' >&2
  exit 1
fi
exec "$aqa_python" -u server.py "$@"
