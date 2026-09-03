#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
surface_root="${EDGEMED_LOCATOR_SURFACE:-/home/ubuntu/data/external/surfaces/slake-train-locator-balanced32-v1}"
data_root="${EDGEMED_SLAKE_DATA_ROOT:-/home/ubuntu/data/external/slake-a9083ce6/extracted/imgs}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
run_dir="${EDGEMED_TRAIN_RUN_DIR:-${repo_root}/runs/qwen35-4b-slake-locator-qlora-smoke2-s20260904}"

cd "${repo_root}"
test ! -e "${run_dir}"
test -f "${surface_root}/inference.jsonl"
test -f "${surface_root}/targets.jsonl"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_locator_qlora \
  --manifest "${surface_root}/inference.jsonl" \
  --targets "${surface_root}/targets.jsonl" \
  --data-root "${data_root}" \
  --model-path "${model_path}" \
  --model-source-manifest "${model_receipt}" \
  --run-dir "${run_dir}" \
  --max-steps 2 \
  --gradient-accumulation 2 \
  --learning-rate 1e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-image-pixels 786432 \
  --grad-scaler-init-scale 1 \
  --seed 20260904
