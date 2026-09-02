#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
rotate1_surface="${EDGEMED_PMC_ROTATE1_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512-rotate1}"
rotate2_surface="${EDGEMED_PMC_ROTATE2_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512-rotate2}"
rotate3_surface="${EDGEMED_PMC_ROTATE3_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512-rotate3}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
m2a_train="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-pilot128-s20260903"
b0_original="${repo_root}/runs/qwen35-4b-pmc-semantic-option-b0-original-s20260903"
m2a_original="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-original-s20260903"
m2a_rotate1="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-rotate1-s20260903"
smoke2="${repo_root}/runs/qwen35-4b-pmc-a1-rotate2-smoke8"
smoke3="${repo_root}/runs/qwen35-4b-pmc-a1-rotate3-smoke8"
full2="${repo_root}/runs/qwen35-4b-pmc-a1-rotate2-full"
full3="${repo_root}/runs/qwen35-4b-pmc-a1-rotate3-full"
agent_run="${repo_root}/runs/qwen35-4b-pmc-a1-semantic-consensus"
reverse_run="${repo_root}/runs/qwen35-4b-pmc-a1-semantic-consensus-reversed"
vs_b0="${repo_root}/runs/pmc-a1-consensus-vs-b0.json"
vs_m2a="${repo_root}/runs/pmc-a1-consensus-vs-m2a.json"
gate_receipt="${repo_root}/runs/pmc-a1-semantic-consensus-gate.json"
seed="20260903"

cd "${repo_root}"

make_rotation() {
  local shift="$1"
  local surface="$2"
  if [[ ! -e "${surface}/report.json" ]]; then
    test ! -e "${surface}/inference.jsonl"
    test ! -e "${surface}/references.jsonl"
    PYTHONPATH=src .venv/bin/python -m edgemed_bench.rotate_mcq \
      --manifest "${dev_surface}/inference.jsonl" \
      --references "${dev_surface}/references.jsonl" \
      --output-manifest "${surface}/inference.jsonl" \
      --output-references "${surface}/references.jsonl" \
      --report "${surface}/report.json" --shift "${shift}"
  fi
}

make_rotation 2 "${rotate2_surface}"
make_rotation 3 "${rotate3_surface}"

PYTHONPATH=src .venv/bin/python - \
  "${b0_original}/predictions.jsonl" \
  "${m2a_original}/run_manifest.json" "${m2a_original}/predictions.jsonl" \
  "${m2a_rotate1}/run_manifest.json" "${m2a_rotate1}/predictions.jsonl" \
  "${m2a_train}/run_manifest.json" "${m2a_train}/adapter/adapter_model.safetensors" <<'PY'
import sys
from pathlib import Path

from edgemed_bench.io import sha256_file

expected = (
    "46df2dc2f4b0de680f8aac76859b580d4509dd2094f3ce37c74170d54bf3add7",
    "9f3e823c5e22d0616f7141f8ff4a0c5636457dc83b7e31fe3ebde047ca56844f",
    "bb94e5a80084cc16a6a0fae730fa68f9f59067743e1998154edad2fb62ce7648",
    "934db1d7c10255a388ba9dae3003c6b600ff381aecd5274eaf49270da84b2ace",
    "3bd87bee4df3ded0da09d540068776fc4163a8fab3626748fd79c93016a62880",
    "cd7e0c69ac1fd381d5b3de7b913f3e162238b2e556a0b1769ea9529e7668086e",
    "d41ac00e2357099955a539f4980a698da2f36c1515338023014b5d278681c6c0",
)
observed = tuple(sha256_file(Path(path)) for path in sys.argv[1:])
assert observed == expected, {"expected": expected, "observed": observed}
PY

run_view() {
  local surface="$1"
  local run_dir="$2"
  local limit="${3:-}"
  local limit_args=()
  if [[ -n "${limit}" ]]; then
    limit_args+=(--limit "${limit}")
  fi
  test ! -e "${run_dir}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind mcq --manifest "${surface}/inference.jsonl" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" --model-source-manifest "${model_receipt}" \
    --adapter-path "${m2a_train}/adapter" \
    --adapter-source-manifest "${m2a_train}/run_manifest.json" \
    --run-dir "${run_dir}" --prompt-variant semantic_option \
    --max-new-tokens 64 --max-image-pixels 786432 --sync-every 10 \
    "${limit_args[@]}"
}

run_view "${rotate2_surface}" "${smoke2}" 8
run_view "${rotate3_surface}" "${smoke3}" 8

PYTHONPATH=src .venv/bin/python - "${smoke2}/predictions.jsonl" "${smoke3}/predictions.jsonl" <<'PY'
import sys
from pathlib import Path

from edgemed_bench.io import read_jsonl

for value in sys.argv[1:]:
    rows = read_jsonl(Path(value))
    parseable = sum(row.get("parsed_answer") is not None for row in rows)
    assert len(rows) == 8 and len({row["sample_id"] for row in rows}) == 8
    assert parseable >= 7, {"path": value, "parseable": parseable}
PY

run_view "${rotate2_surface}" "${full2}"
run_view "${rotate3_surface}" "${full3}"

mkdir -p "${agent_run}" "${reverse_run}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.semantic_consensus \
  --original-manifest "${dev_surface}/inference.jsonl" \
  --original-predictions "${m2a_original}/predictions.jsonl" \
  --rotated-manifest "${rotate1_surface}/inference.jsonl" \
  --rotated-predictions "${m2a_rotate1}/predictions.jsonl" \
  --rotated-manifest "${rotate2_surface}/inference.jsonl" \
  --rotated-predictions "${full2}/predictions.jsonl" \
  --rotated-manifest "${rotate3_surface}/inference.jsonl" \
  --rotated-predictions "${full3}/predictions.jsonl" \
  --output "${agent_run}/predictions.jsonl" --report "${agent_run}/agent-report.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.semantic_consensus \
  --original-manifest "${dev_surface}/inference.jsonl" \
  --original-predictions "${m2a_original}/predictions.jsonl" \
  --rotated-manifest "${rotate3_surface}/inference.jsonl" \
  --rotated-predictions "${full3}/predictions.jsonl" \
  --rotated-manifest "${rotate2_surface}/inference.jsonl" \
  --rotated-predictions "${full2}/predictions.jsonl" \
  --rotated-manifest "${rotate1_surface}/inference.jsonl" \
  --rotated-predictions "${m2a_rotate1}/predictions.jsonl" \
  --output "${reverse_run}/predictions.jsonl" --report "${reverse_run}/agent-report.json"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
  --manifest "${dev_surface}/inference.jsonl" --references "${dev_surface}/references.jsonl" \
  --predictions "${agent_run}/predictions.jsonl" --output "${agent_run}/metrics.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${dev_surface}/references.jsonl" \
  --predictions-a "${b0_original}/predictions.jsonl" \
  --predictions-b "${agent_run}/predictions.jsonl" \
  --output "${vs_b0}" --bootstrap-repetitions 10000 --seed "${seed}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${dev_surface}/references.jsonl" \
  --predictions-a "${m2a_original}/predictions.jsonl" \
  --predictions-b "${agent_run}/predictions.jsonl" \
  --output "${vs_m2a}" --bootstrap-repetitions 10000 --seed "${seed}"

PYTHONPATH=src .venv/bin/python - \
  "${agent_run}/agent-report.json" "${agent_run}/metrics.json" \
  "${agent_run}/predictions.jsonl" "${reverse_run}/predictions.jsonl" \
  "${vs_b0}" "${vs_m2a}" "${gate_receipt}" <<'PY'
import json
import sys
from pathlib import Path

from edgemed_bench.io import read_jsonl, sha256_file, write_json

report_path, metrics_path, predictions_path, reverse_path, b0_path, m2a_path, output_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
metrics = json.loads(metrics_path.read_text())
vs_b0 = json.loads(b0_path.read_text())
vs_m2a = json.loads(m2a_path.read_text())
predictions = [(row["sample_id"], row.get("parsed_answer")) for row in read_jsonl(predictions_path)]
reverse = [(row["sample_id"], row.get("parsed_answer")) for row in read_jsonl(reverse_path)]
checks = {
    "vs_b0_ci_lower_gt_zero": vs_b0["paired_bootstrap_95_percent"][0] > 0,
    "vs_m2a_noninferiority_lower_ge_minus_1": vs_m2a["paired_bootstrap_95_percent"][0] >= -1,
    "invalid_count_le_m2a_original_10": metrics["invalid_parse"]["count"] <= 10,
    "view_argument_order_invariant": predictions == reverse,
}
result = {
    "schema_version": "edgemed-a1-semantic-consensus-gate/v1",
    "status": "passed" if all(checks.values()) else "failed",
    "checks": checks,
    "agent_report": report,
    "agent_metrics": metrics,
    "vs_b0": vs_b0,
    "vs_m2a": vs_m2a,
    "compute": {"model_calls_per_question": 4, "incremental_pilot_calls_per_question": 2},
    "source_hashes": {
        "predictions_sha256": sha256_file(predictions_path),
        "reverse_predictions_sha256": sha256_file(reverse_path),
        "vs_b0_sha256": sha256_file(b0_path),
        "vs_m2a_sha256": sha256_file(m2a_path),
    },
}
write_json(output_path, result)
print(output_path.read_text())
PY
