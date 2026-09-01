#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
train_surface="${EDGEMED_PMC_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
seed="${EDGEMED_SEED:-20260901}"
train_run="${repo_root}/runs/qwen35-4b-pmc-t1a-qlora-pilot128-s${seed}"
eval_run="${repo_root}/runs/qwen35-4b-pmc-t1a-pilot128-dev-s${seed}"
direct_predictions="${repo_root}/runs/qwen35-4b-pmc-vqa-dev-direct-px786432-20260901/predictions.jsonl"

cd "${repo_root}"
test ! -e "${train_run}"
test ! -e "${eval_run}"
test -f "${train_surface}/inference.jsonl"
test -f "${train_surface}/references.jsonl"
test -f "${dev_surface}/inference.jsonl"
test -f "${dev_surface}/references.jsonl"
test -f "${direct_predictions}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_qlora \
  --manifest "${train_surface}/inference.jsonl" \
  --references "${train_surface}/references.jsonl" \
  --data-root "${data_base}/extracted/figures" \
  --model-path "${model_path}" \
  --model-source-manifest "${model_receipt}" \
  --run-dir "${train_run}" \
  --max-steps 128 \
  --gradient-accumulation 2 \
  --learning-rate 1e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-image-pixels 786432 \
  --grad-scaler-init-scale 1 \
  --seed "${seed}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
  --kind mcq \
  --manifest "${dev_surface}/inference.jsonl" \
  --data-root "${data_base}/extracted/figures" \
  --model-path "${model_path}" \
  --model-source-manifest "${model_receipt}" \
  --adapter-path "${train_run}/adapter" \
  --adapter-source-manifest "${train_run}/run_manifest.json" \
  --run-dir "${eval_run}" \
  --prompt-variant direct \
  --max-new-tokens 64 \
  --max-image-pixels 786432 \
  --sync-every 10

PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
  --manifest "${dev_surface}/inference.jsonl" \
  --references "${dev_surface}/references.jsonl" \
  --predictions "${eval_run}/predictions.jsonl" \
  --output "${eval_run}/metrics.json"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${dev_surface}/references.jsonl" \
  --predictions-a "${direct_predictions}" \
  --predictions-b "${eval_run}/predictions.jsonl" \
  --output "${eval_run}/paired-vs-direct.json" \
  --bootstrap-repetitions 10000 \
  --seed "${seed}"
