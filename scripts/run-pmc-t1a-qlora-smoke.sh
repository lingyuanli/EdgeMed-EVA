#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
manifest="${EDGEMED_PMC_TRAIN_MANIFEST:-/home/ubuntu/data/external/manifests/pmc-vqa-train-admitted-1968.jsonl}"
gate_report="${EDGEMED_PMC_TRAIN_GATE:-${repo_root}/runs/data-build/pmc-vqa-train-final-gate.json}"
surface_root="${EDGEMED_PMC_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
run_dir="${EDGEMED_TRAIN_RUN_DIR:-${repo_root}/runs/qwen35-4b-pmc-t1a-qlora-smoke-v3-20260901}"

cd "${repo_root}"
test "$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${gate_report}")" = passed
mkdir -p "${surface_root}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.prepare_external split-surfaces \
  --manifest "${manifest}" \
  --kind mcq \
  --inference-output "${surface_root}/inference.jsonl" \
  --references-output "${surface_root}/references.jsonl" \
  --report "${surface_root}/surface-report.json"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_qlora \
  --manifest "${surface_root}/inference.jsonl" \
  --references "${surface_root}/references.jsonl" \
  --data-root "${data_base}/extracted/figures" \
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
  --seed 20260901
