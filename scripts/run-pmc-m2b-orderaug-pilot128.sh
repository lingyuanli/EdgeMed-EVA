#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
aug_surface="${EDGEMED_PMC_M2B_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968-m2b-order-s20260903}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
rotated_surface="${EDGEMED_PMC_ROTATED_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512-rotate1}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
seed="${EDGEMED_SEED:-20260903}"
train_run="${repo_root}/runs/qwen35-4b-pmc-m2b-orderaug-pilot128-s${seed}"
b0_original="${repo_root}/runs/qwen35-4b-pmc-semantic-option-b0-original-s${seed}"
b0_rotated="${repo_root}/runs/qwen35-4b-pmc-semantic-option-b0-rotate1-s${seed}"
m2b_original="${repo_root}/runs/qwen35-4b-pmc-m2b-orderaug-original-s${seed}"
m2b_rotated="${repo_root}/runs/qwen35-4b-pmc-m2b-orderaug-rotate1-s${seed}"
original_pair="${repo_root}/runs/pmc-semantic-option-original-b0-vs-m2b-s${seed}.json"
rotated_pair="${repo_root}/runs/pmc-semantic-option-rotate1-b0-vs-m2b-s${seed}.json"
invariance_pair="${repo_root}/runs/pmc-semantic-option-invariance-b0-vs-m2b-s${seed}.json"
gate_receipt="${repo_root}/runs/pmc-m2b-orderaug-pilot128-gate-s${seed}.json"
skip_train="${EDGEMED_SKIP_TRAIN:-0}"

cd "${repo_root}"
test -f "${aug_surface}/report.json"
for path in "${m2b_original}" "${m2b_rotated}" "${original_pair}" "${rotated_pair}" "${invariance_pair}" "${gate_receipt}"; do
  test ! -e "${path}"
done

PYTHONPATH=src .venv/bin/python - \
  "${b0_original}/run_manifest.json" "${b0_original}/predictions.jsonl" \
  "${b0_rotated}/run_manifest.json" "${b0_rotated}/predictions.jsonl" \
  "${dev_surface}/inference.jsonl" "${dev_surface}/references.jsonl" \
  "${rotated_surface}/inference.jsonl" "${rotated_surface}/references.jsonl" <<'PY'
import sys
from pathlib import Path

from edgemed_bench.io import sha256_file

expected = (
    "49dbc78d57dbeedfb4d93f1dbfd6b7a447aaae1e1e19a17dda3e22ac96de7a58",
    "46df2dc2f4b0de680f8aac76859b580d4509dd2094f3ce37c74170d54bf3add7",
    "efb75b338a08b943c01d63dc2b171027f05e0cdee415b50984d310595f2a1aff",
    "4bb8690ef72383be557db8834e4fd763bbf5dbc2ee8e746baf193fd2af706c9d",
    "78c6d2a5c0790eaf3c66db523774f4cfcfeb93e27d4cf743e540f8bbfdff5e75",
    "9a7e03cfdfa258b02eecace8c805dfceeeb247ee12cc63b635aeb003b4b3b0f6",
    "682eaf51fad657d5b25c1fa2985b8dd7690e388c26bde158f0ee90ff48645b75",
    "05ca829bfb9995ed90c2f06ad6d807c5f1c4e702bc5ce8e8a5fd596871f3f6be",
)
observed = tuple(sha256_file(Path(value)) for value in sys.argv[1:])
assert observed == expected, {"expected": expected, "observed": observed}
PY

if [[ "${skip_train}" == "1" ]]; then
  test "$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${train_run}/run_manifest.json")" = completed
else
  test ! -e "${train_run}"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_qlora \
    --manifest "${aug_surface}/inference.jsonl" \
    --references "${aug_surface}/references.jsonl" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" --model-source-manifest "${model_receipt}" \
    --run-dir "${train_run}" --target-mode option_text \
    --max-steps 128 --gradient-accumulation 2 --learning-rate 1e-4 \
    --lora-rank 16 --lora-alpha 32 --max-image-pixels 786432 \
    --grad-scaler-init-scale 1 --seed "${seed}"
fi

run_eval() {
  local manifest="$1"
  local references="$2"
  local run_dir="$3"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
    --kind mcq --manifest "${manifest}" \
    --data-root "${data_base}/extracted/figures" \
    --model-path "${model_path}" --model-source-manifest "${model_receipt}" \
    --adapter-path "${train_run}/adapter" \
    --adapter-source-manifest "${train_run}/run_manifest.json" \
    --run-dir "${run_dir}" --prompt-variant semantic_option \
    --max-new-tokens 64 --max-image-pixels 786432 --sync-every 10
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_mcq \
    --manifest "${manifest}" --references "${references}" \
    --predictions "${run_dir}/predictions.jsonl" --output "${run_dir}/metrics.json"
}

run_eval "${dev_surface}/inference.jsonl" "${dev_surface}/references.jsonl" "${m2b_original}"
run_eval "${rotated_surface}/inference.jsonl" "${rotated_surface}/references.jsonl" "${m2b_rotated}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${dev_surface}/references.jsonl" \
  --predictions-a "${b0_original}/predictions.jsonl" \
  --predictions-b "${m2b_original}/predictions.jsonl" \
  --output "${original_pair}" --bootstrap-repetitions 10000 --seed "${seed}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_mcq \
  --references "${rotated_surface}/references.jsonl" \
  --predictions-a "${b0_rotated}/predictions.jsonl" \
  --predictions-b "${m2b_rotated}/predictions.jsonl" \
  --output "${rotated_pair}" --bootstrap-repetitions 10000 --seed "${seed}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.score_choice_invariance \
  --rotated-manifest "${rotated_surface}/inference.jsonl" \
  --original-predictions "${m2b_original}/predictions.jsonl" \
  --rotated-predictions "${m2b_rotated}/predictions.jsonl" \
  --output "${m2b_rotated}/choice-invariance.json"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.compare_choice_invariance \
  --rotated-manifest "${rotated_surface}/inference.jsonl" \
  --original-a "${b0_original}/predictions.jsonl" \
  --rotated-a "${b0_rotated}/predictions.jsonl" \
  --original-b "${m2b_original}/predictions.jsonl" \
  --rotated-b "${m2b_rotated}/predictions.jsonl" \
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
    "schema_version": "edgemed-m2b-pilot-gate/v1",
    "status": "passed" if all(checks.values()) else "failed",
    "checks": checks,
    "original_pair": original,
    "rotated_pair": rotated,
    "invariance_pair": invariance,
    "reused_b0_prediction_hashes": {
        "original": "46df2dc2f4b0de680f8aac76859b580d4509dd2094f3ce37c74170d54bf3add7",
        "rotate1": "4bb8690ef72383be557db8834e4fd763bbf5dbc2ee8e746baf193fd2af706c9d",
    },
    "source_hashes": {
        "original_pair_sha256": sha256_file(original_path),
        "rotated_pair_sha256": sha256_file(rotated_path),
        "invariance_pair_sha256": sha256_file(invariance_path),
    },
}
write_json(output_path, result)
print(output_path.read_text())
PY
