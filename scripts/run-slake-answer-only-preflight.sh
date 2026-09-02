#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_SLAKE_DATA_BASE:-/home/ubuntu/data/external/slake-a9083ce6}"
surface_root="${EDGEMED_SLAKE_SURFACES:-/home/ubuntu/data/external/surfaces/slake-validation-en-1053}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
adapter_run="${repo_root}/runs/qwen35-4b-pmc-t1a-qlora-pilot128-s20260903"
b0_run="${repo_root}/runs/qwen35-4b-slake-answer-only32-preflight-b0-parser2-20260902"
m1a_run="${repo_root}/runs/qwen35-4b-slake-answer-only32-preflight-m1a-parser2-20260902"
receipt="${repo_root}/runs/slake-answer-only32-preflight-parser2-20260902.json"

cd "${repo_root}"
test -f "${surface_root}/inference.jsonl"
test -d "${data_base}/extracted/imgs"
test -f "${adapter_run}/adapter/adapter_config.json"

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
    --kind open \
    --manifest "${surface_root}/inference.jsonl" \
    --data-root "${data_base}/extracted/imgs" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --run-dir "${run_dir}" \
    --prompt-variant answer_only \
    --max-new-tokens 32 \
    --max-image-pixels 786432 \
    --limit 32 \
    --sync-every 1 \
    "${adapter_args[@]}"
done

PYTHONPATH=src .venv/bin/python - "${b0_run}/predictions.jsonl" "${m1a_run}/predictions.jsonl" "${receipt}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from edgemed_bench.io import read_jsonl, sha256_file, write_json

result = {"schema_version": "edgemed-open-operational-preflight/v1", "required_parseable": 31}
passed = True
for label, value in zip(("b0", "m1a"), sys.argv[1:3]):
    path = Path(value)
    rows = read_jsonl(path)
    counts = Counter(str(row.get("parse_status")) for row in rows)
    parseable = len(rows) - counts["invalid"]
    result[label] = {
        "observed": len(rows),
        "parseable": parseable,
        "parse_status_counts": dict(sorted(counts.items())),
        "predictions_sha256": sha256_file(path),
    }
    passed = passed and len(rows) == 32 and parseable >= 31
result["status"] = "passed" if passed else "failed"
write_json(Path(sys.argv[3]), result)
print(Path(sys.argv[3]).read_text())
raise SystemExit(0 if passed else 1)
PY
