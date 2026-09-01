#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
manifest="${EDGEMED_PMC_DEV_MANIFEST:-/home/ubuntu/data/external/manifests/pmc-vqa-mcq-dev-512.jsonl}"
gate_report="${EDGEMED_PMC_DEV_GATE:-${repo_root}/runs/data-build/pmc-vqa-dev-overlap-gate.json}"
surface_root="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"

cd "${repo_root}"
test "$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${gate_report}")" = passed
mkdir -p "${surface_root}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.prepare_external split-surfaces \
  --manifest "${manifest}" \
  --kind mcq \
  --inference-output "${surface_root}/inference.jsonl" \
  --references-output "${surface_root}/references.jsonl" \
  --report "${surface_root}/surface-report.json"

for variant in direct evidence_answer_v2; do
  run_dir="${repo_root}/runs/qwen35-4b-pmc-vqa-dev-${variant}-20260901"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind mcq \
    --manifest "${surface_root}/inference.jsonl" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --run-dir "${run_dir}" \
    --prompt-variant "${variant}" \
    --max-new-tokens 64 \
    --sync-every 10
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
    --manifest "${surface_root}/inference.jsonl" \
    --references "${surface_root}/references.jsonl" \
    --predictions "${run_dir}/predictions.jsonl" \
    --output "${run_dir}/metrics.json"
done

