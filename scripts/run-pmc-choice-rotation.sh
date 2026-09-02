#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
surface_root="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
rotated_root="${EDGEMED_PMC_ROTATED_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512-rotate1}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
adapter_run="${repo_root}/runs/qwen35-4b-pmc-t1a-qlora-pilot128-s20260903"
original_b0="${repo_root}/runs/qwen35-4b-pmc-vqa-dev-direct-px786432-20260901/predictions.jsonl"
original_m1a="${repo_root}/runs/qwen35-4b-pmc-t1a-pilot128-dev-s20260903/predictions.jsonl"
b0_run="${repo_root}/runs/qwen35-4b-pmc-vqa-dev-rotate1-direct-20260902"
m1a_run="${repo_root}/runs/qwen35-4b-pmc-vqa-dev-rotate1-m1a-s20260903-20260902"

cd "${repo_root}"
test -f "${surface_root}/inference.jsonl"
test -f "${surface_root}/references.jsonl"
test -f "${original_b0}"
test -f "${original_m1a}"
test -f "${adapter_run}/adapter/adapter_config.json"
mkdir -p "${rotated_root}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.rotate_mcq \
  --manifest "${surface_root}/inference.jsonl" \
  --references "${surface_root}/references.jsonl" \
  --output-manifest "${rotated_root}/inference.jsonl" \
  --output-references "${rotated_root}/references.jsonl" \
  --report "${rotated_root}/rotation-report.json" \
  --shift 1

for run_spec in "b0:${b0_run}" "m1a:${m1a_run}"; do
  model_kind="${run_spec%%:*}"
  run_dir="${run_spec#*:}"
  adapter_args=()
  if [[ "${model_kind}" == "m1a" ]]; then
    adapter_args+=(--adapter-path "${adapter_run}/adapter")
    adapter_args+=(--adapter-source-manifest "${adapter_run}/run_manifest.json")
  fi
  test ! -e "${run_dir}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind mcq \
    --manifest "${rotated_root}/inference.jsonl" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --run-dir "${run_dir}" \
    --prompt-variant direct \
    --max-new-tokens 64 \
    --max-image-pixels 786432 \
    --sync-every 10 \
    "${adapter_args[@]}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
    --manifest "${rotated_root}/inference.jsonl" \
    --references "${rotated_root}/references.jsonl" \
    --predictions "${run_dir}/predictions.jsonl" \
    --output "${run_dir}/metrics.json"
done

PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_choice_invariance \
  --rotated-manifest "${rotated_root}/inference.jsonl" \
  --original-predictions "${original_b0}" \
  --rotated-predictions "${b0_run}/predictions.jsonl" \
  --output "${b0_run}/choice-invariance.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_choice_invariance \
  --rotated-manifest "${rotated_root}/inference.jsonl" \
  --original-predictions "${original_m1a}" \
  --rotated-predictions "${m1a_run}/predictions.jsonl" \
  --output "${m1a_run}/choice-invariance.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${rotated_root}/references.jsonl" \
  --predictions-a "${b0_run}/predictions.jsonl" \
  --predictions-b "${m1a_run}/predictions.jsonl" \
  --output "${repo_root}/runs/pmc-vqa-dev-rotate1-direct-vs-m1a-s20260903-20260902.json" \
  --bootstrap-repetitions 10000 \
  --seed 20260902
