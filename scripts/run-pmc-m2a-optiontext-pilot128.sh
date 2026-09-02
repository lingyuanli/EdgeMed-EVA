#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
train_surface="${EDGEMED_PMC_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
rotated_surface="${EDGEMED_PMC_ROTATED_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512-rotate1}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
seed="${EDGEMED_SEED:-20260903}"
train_run="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-pilot128-s${seed}"
b0_original="${repo_root}/runs/qwen35-4b-pmc-semantic-option-b0-original-s${seed}"
b0_rotated="${repo_root}/runs/qwen35-4b-pmc-semantic-option-b0-rotate1-s${seed}"
m2a_original="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-original-s${seed}"
m2a_rotated="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-rotate1-s${seed}"
original_pair="${repo_root}/runs/pmc-semantic-option-original-b0-vs-m2a-s${seed}.json"
rotated_pair="${repo_root}/runs/pmc-semantic-option-rotate1-b0-vs-m2a-s${seed}.json"
invariance_pair="${repo_root}/runs/pmc-semantic-option-invariance-b0-vs-m2a-s${seed}.json"
gate_receipt="${repo_root}/runs/pmc-m2a-optiontext-pilot128-gate-s${seed}.json"
skip_train="${EDGEMED_SKIP_TRAIN:-0}"

cd "${repo_root}"
for path in "${b0_original}" "${b0_rotated}" "${m2a_original}" "${m2a_rotated}"; do
  test ! -e "${path}"
done

if [[ "${skip_train}" == "1" ]]; then
  test "$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${train_run}/run_manifest.json")" = completed
else
  test ! -e "${train_run}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_qlora \
    --manifest "${train_surface}/inference.jsonl" \
    --references "${train_surface}/references.jsonl" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --run-dir "${train_run}" \
    --target-mode option_text \
    --max-steps 128 \
    --gradient-accumulation 2 \
    --learning-rate 1e-4 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --max-image-pixels 786432 \
    --grad-scaler-init-scale 1 \
    --seed "${seed}"
fi

run_eval() {
  local manifest="$1"
  local references="$2"
  local run_dir="$3"
  local use_adapter="$4"
  local adapter_args=()
  if [[ "${use_adapter}" == "yes" ]]; then
    adapter_args+=(--adapter-path "${train_run}/adapter")
    adapter_args+=(--adapter-source-manifest "${train_run}/run_manifest.json")
  fi
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind mcq \
    --manifest "${manifest}" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" \
    --model-source-manifest "${model_receipt}" \
    --run-dir "${run_dir}" \
    --prompt-variant semantic_option \
    --max-new-tokens 64 \
    --max-image-pixels 786432 \
    --sync-every 10 \
    "${adapter_args[@]}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
    --manifest "${manifest}" \
    --references "${references}" \
    --predictions "${run_dir}/predictions.jsonl" \
    --output "${run_dir}/metrics.json"
}

run_eval "${dev_surface}/inference.jsonl" "${dev_surface}/references.jsonl" "${b0_original}" no
run_eval "${rotated_surface}/inference.jsonl" "${rotated_surface}/references.jsonl" "${b0_rotated}" no
run_eval "${dev_surface}/inference.jsonl" "${dev_surface}/references.jsonl" "${m2a_original}" yes
run_eval "${rotated_surface}/inference.jsonl" "${rotated_surface}/references.jsonl" "${m2a_rotated}" yes

PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${dev_surface}/references.jsonl" \
  --predictions-a "${b0_original}/predictions.jsonl" \
  --predictions-b "${m2a_original}/predictions.jsonl" \
  --output "${original_pair}" --bootstrap-repetitions 10000 --seed "${seed}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${rotated_surface}/references.jsonl" \
  --predictions-a "${b0_rotated}/predictions.jsonl" \
  --predictions-b "${m2a_rotated}/predictions.jsonl" \
  --output "${rotated_pair}" --bootstrap-repetitions 10000 --seed "${seed}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_choice_invariance \
  --rotated-manifest "${rotated_surface}/inference.jsonl" \
  --original-predictions "${b0_original}/predictions.jsonl" \
  --rotated-predictions "${b0_rotated}/predictions.jsonl" \
  --output "${b0_rotated}/choice-invariance.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_choice_invariance \
  --rotated-manifest "${rotated_surface}/inference.jsonl" \
  --original-predictions "${m2a_original}/predictions.jsonl" \
  --rotated-predictions "${m2a_rotated}/predictions.jsonl" \
  --output "${m2a_rotated}/choice-invariance.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_choice_invariance \
  --rotated-manifest "${rotated_surface}/inference.jsonl" \
  --original-a "${b0_original}/predictions.jsonl" \
  --rotated-a "${b0_rotated}/predictions.jsonl" \
  --original-b "${m2a_original}/predictions.jsonl" \
  --rotated-b "${m2a_rotated}/predictions.jsonl" \
  --output "${invariance_pair}" --bootstrap-repetitions 10000 --seed "${seed}"

PYTHONPATH=src .venv/bin/python - "${original_pair}" "${rotated_pair}" "${invariance_pair}" "${gate_receipt}" <<'PY'
import json
import sys
from pathlib import Path

from edgemed_bench.io import sha256_file, write_json

original_path, rotated_path, invariance_path, output_path = map(Path, sys.argv[1:])
original = json.loads(original_path.read_text())
rotated = json.loads(rotated_path.read_text())
invariance = json.loads(invariance_path.read_text())
checks = {
    "original_delta_positive": original["delta_accuracy_points"] > 0,
    "rotated_noninferiority_lower_ge_minus_1": rotated["paired_bootstrap_95_percent"][0] >= -1,
    "invariance_noninferiority_lower_ge_minus_1": invariance["paired_bootstrap_95_percent"][0] >= -1,
}
result = {
    "schema_version": "edgemed-m2a-pilot-gate/v1",
    "status": "passed" if all(checks.values()) else "failed",
    "checks": checks,
    "original_pair": original,
    "rotated_pair": rotated,
    "invariance_pair": invariance,
    "source_hashes": {
        "original_pair_sha256": sha256_file(original_path),
        "rotated_pair_sha256": sha256_file(rotated_path),
        "invariance_pair_sha256": sha256_file(invariance_path),
    },
}
write_json(output_path, result)
print(output_path.read_text())
PY
