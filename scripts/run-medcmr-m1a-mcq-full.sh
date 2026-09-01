#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

run_dir="${EDGEMED_RUN_DIR:-${project_dir}/runs/qwen35-4b-medcmr-m1a-sft128-s20260903-mcq-full-20260901}"
adapter_run="${project_dir}/runs/qwen35-4b-pmc-t1a-qlora-pilot128-s20260903"
base_run="${project_dir}/runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z"
resume_args=()
if [[ "${EDGEMED_EXACT_RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi
mkdir -p "${run_dir}"

set +e
PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
  --kind mcq \
  --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --data-root /home/ubuntu/data/medcmr/release_a9b2d6e6/raw \
  --model-path /home/ubuntu/models/Qwen3.5-4B \
  --model-source-manifest baselines/local/qwen35-4b-medcmr-b0/source_manifest.json \
  --adapter-path "${adapter_run}/adapter" \
  --adapter-source-manifest "${adapter_run}/run_manifest.json" \
  --run-dir "${run_dir}" \
  --sync-every 10 \
  "${resume_args[@]}" \
  2>&1 | tee -a "${run_dir}/console.log"
runner_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${runner_status}" > "${run_dir}/process_exit_code"
if [[ "${runner_status}" -ne 0 ]]; then
  exit "${runner_status}"
fi

PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
  --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --references /home/ubuntu/data/medcmr/release_a9b2d6e6/references/mcq.jsonl \
  --predictions "${run_dir}/predictions.jsonl" \
  --output "${run_dir}/metrics.json"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references /home/ubuntu/data/medcmr/release_a9b2d6e6/references/mcq.jsonl \
  --predictions-a "${base_run}/predictions.jsonl" \
  --predictions-b "${run_dir}/predictions.jsonl" \
  --output "${run_dir}/paired-vs-b0.json" \
  --bootstrap-repetitions 10000 \
  --seed 20260903
