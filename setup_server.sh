#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required. Install it using https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
# Optionally choose the CUDA wheel index compatible with the server driver.
if [[ -n "${AQA_TORCH_INDEX:-}" ]]; then
  uv pip install --python .venv/bin/python 'torch>=2.7,<3' --index-url "$AQA_TORCH_INDEX"
  uv pip install --python .venv/bin/python -r requirements.txt
else
  uv pip install --python .venv/bin/python --torch-backend auto -r requirements.txt
fi
.venv/bin/python -c 'import torch, transformers, swift; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 GPU unavailable"'
