#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="$(command -v uv || true)"

if [[ -z "${uv_bin}" && -x /home/ubuntu/.local/bin/uv ]]; then
  uv_bin=/home/ubuntu/.local/bin/uv
fi

if [[ -z "${uv_bin}" ]]; then
  echo "uv is required but was not found" >&2
  exit 1
fi

cd "${repo_dir}"
export UV_CACHE_DIR="${repo_dir}/.cache/uv"
export UV_LINK_MODE=copy

"${uv_bin}" venv --python /usr/bin/python3 .venv
"${uv_bin}" pip install \
  --python .venv/bin/python \
  torch==2.10.0 \
  torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126
"${uv_bin}" pip install \
  --python .venv/bin/python \
  --requirement environment/requirements-v100.txt

.venv/bin/python scripts/smoke-v100.py --cuda-only
