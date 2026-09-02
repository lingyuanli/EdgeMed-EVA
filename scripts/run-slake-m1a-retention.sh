#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_SLAKE_DATA_BASE:-/home/ubuntu/data/external/slake-a9083ce6}"
manifest="${EDGEMED_SLAKE_MANIFEST:-/home/ubuntu/data/external/manifests/slake-validation-en.jsonl}"
surface_root="${EDGEMED_SLAKE_SURFACES:-/home/ubuntu/data/external/surfaces/slake-validation-en-1053}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
adapter_run="${repo_root}/runs/qwen35-4b-pmc-t1a-qlora-pilot128-s20260903"
direct_run="${repo_root}/runs/qwen35-4b-slake-validation-direct-20260902"
adapter_eval_run="${repo_root}/runs/qwen35-4b-slake-validation-m1a-s20260903-20260902"
comparison="${repo_root}/runs/slake-validation-direct-vs-m1a-s20260903-20260902.json"

cd "${repo_root}"
test -f "${manifest}"
test -d "${data_base}/extracted/imgs"
test -f "${adapter_run}/adapter/adapter_config.json"
test -f "${adapter_run}/run_manifest.json"
mkdir -p "${surface_root}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.prepare_external split-surfaces \
  --manifest "${manifest}" \
  --kind open \
  --inference-output "${surface_root}/inference.jsonl" \
  --references-output "${surface_root}/references.jsonl" \
  --report "${surface_root}/surface-report.json"

if test ! -e "${direct_run}"; then
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind open \
    --manifest "${surface_root}/inference.jsonl" \
    --data-root "${data_base}/extracted/imgs" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --run-dir "${direct_run}" \
    --prompt-variant direct \
    --max-new-tokens 64 \
    --max-image-pixels 786432 \
    --sync-every 10
fi

PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_open \
  --manifest "${surface_root}/inference.jsonl" \
  --references "${surface_root}/references.jsonl" \
  --predictions "${direct_run}/predictions.jsonl" \
  --output "${direct_run}/metrics.json"

if test ! -e "${adapter_eval_run}"; then
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind open \
    --manifest "${surface_root}/inference.jsonl" \
    --data-root "${data_base}/extracted/imgs" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --adapter-path "${adapter_run}/adapter" \
    --adapter-source-manifest "${adapter_run}/run_manifest.json" \
    --run-dir "${adapter_eval_run}" \
    --prompt-variant direct \
    --max-new-tokens 64 \
    --max-image-pixels 786432 \
    --sync-every 10
fi

PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_open \
  --manifest "${surface_root}/inference.jsonl" \
  --references "${surface_root}/references.jsonl" \
  --predictions "${adapter_eval_run}/predictions.jsonl" \
  --output "${adapter_eval_run}/metrics.json"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_open \
  --references "${surface_root}/references.jsonl" \
  --predictions-a "${direct_run}/predictions.jsonl" \
  --predictions-b "${adapter_eval_run}/predictions.jsonl" \
  --output "${comparison}" \
  --bootstrap-repetitions 10000 \
  --seed 20260902
