#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

run_dir="${EDGEMED_RUN_DIR:-$project_dir/runs/qwen35-4b-medcmr-b0-mcq-full-20260831T0427Z}"
resume_args=()
if [[ "${EDGEMED_EXACT_RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi
mkdir -p "$run_dir"

set +e
PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
  --kind mcq \
  --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --data-root /home/ubuntu/data/medcmr/release_a9b2d6e6/raw \
  --model-path /home/ubuntu/models/Qwen3.5-4B \
  --model-source-manifest baselines/local/qwen35-4b-medcmr-b0/source_manifest.json \
  --run-dir "$run_dir" \
  --sync-every 10 \
  "${resume_args[@]}" \
  2>&1 | tee -a "$run_dir/console.log"
runner_status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$runner_status" > "$run_dir/process_exit_code"
exit "$runner_status"
