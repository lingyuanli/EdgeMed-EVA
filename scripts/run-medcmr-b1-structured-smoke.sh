#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

run_dir="${EDGEMED_RUN_DIR:-$project_dir/runs/qwen35-4b-medcmr-b1-structured-smoke-20260901T0615Z}"
selection="/home/ubuntu/EdgeMed-EVA/runs/_selections/medcmr-mcq-2-per-task.txt"
mkdir -p "$run_dir"

# Operational smoke only: this command intentionally has no references path and
# never invokes score_mcq. It may validate schema, parsing, memory, and latency,
# but must not expose correctness while the B1 prompt remains under development.
set +e
PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
  --kind mcq \
  --prompt-variant structured_evidence \
  --manifest /home/ubuntu/data/medcmr/release_a9b2d6e6/manifests/mcq.jsonl \
  --data-root /home/ubuntu/data/medcmr/release_a9b2d6e6/raw \
  --model-path /home/ubuntu/models/Qwen3.5-4B \
  --model-source-manifest baselines/local/qwen35-4b-medcmr-b0/source_manifest.json \
  --run-dir "$run_dir" \
  --sample-id-file "$selection" \
  --max-new-tokens 128 \
  --sync-every 1 \
  2>&1 | tee -a "$run_dir/console.log"
runner_status=${PIPESTATUS[0]}
set -e

printf '%s\n' "$runner_status" > "$run_dir/process_exit_code"
exit "$runner_status"
