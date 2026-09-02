#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
train_run="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-pilot128-s20260903"
b0_run="${repo_root}/runs/qwen35-4b-pmc-semantic-option-parser2-smoke32-b0"
m2a_run="${repo_root}/runs/qwen35-4b-pmc-semantic-option-parser2-smoke32-m2a"
receipt="${repo_root}/runs/pmc-semantic-option-parser2-smoke32.json"

cd "${repo_root}"
for run_spec in "b0:${b0_run}" "m2a:${m2a_run}"; do
  label="${run_spec%%:*}"
  run_dir="${run_spec#*:}"
  adapter_args=()
  if [[ "${label}" == "m2a" ]]; then
    adapter_args+=(--adapter-path "${train_run}/adapter")
    adapter_args+=(--adapter-source-manifest "${train_run}/run_manifest.json")
  fi
  test ! -e "${run_dir}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind mcq --manifest "${dev_surface}/inference.jsonl" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" --model-source-manifest "${model_receipt}" \
    --run-dir "${run_dir}" --prompt-variant semantic_option \
    --max-new-tokens 64 --max-image-pixels 786432 --limit 32 --sync-every 1 \
    "${adapter_args[@]}"
done

PYTHONPATH=src .venv/bin/python - "${b0_run}/predictions.jsonl" "${m2a_run}/predictions.jsonl" "${receipt}" <<'PY'
import sys
from collections import Counter
from pathlib import Path

from edgemed_bench.io import read_jsonl, sha256_file, write_json

result = {"schema_version": "edgemed-semantic-option-parser-preflight/v1"}
passed = True
for label, value, required in (("b0", sys.argv[1], 24), ("m2a", sys.argv[2], 31)):
    path = Path(value)
    rows = read_jsonl(path)
    counts = Counter(str(row.get("parse_status")) for row in rows)
    parseable = len(rows) - counts["invalid_option_text"]
    result[label] = {
        "observed": len(rows), "parseable": parseable, "required": required,
        "parse_status_counts": dict(sorted(counts.items())),
        "predictions_sha256": sha256_file(path),
    }
    passed = passed and len(rows) == 32 and parseable >= required
result["status"] = "passed" if passed else "failed"
write_json(Path(sys.argv[3]), result)
print(Path(sys.argv[3]).read_text())
raise SystemExit(0 if passed else 1)
PY
