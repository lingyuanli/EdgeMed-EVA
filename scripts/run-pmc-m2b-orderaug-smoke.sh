#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
source_surface="${EDGEMED_PMC_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968}"
aug_surface="${EDGEMED_PMC_M2B_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968-m2b-order-s20260903}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
seed="${EDGEMED_SEED:-20260903}"
train_run="${repo_root}/runs/qwen35-4b-pmc-m2b-orderaug-qlora-smoke2-s${seed}"
reload_run="${repo_root}/runs/qwen35-4b-pmc-m2b-orderaug-reload4-s${seed}"

cd "${repo_root}"
if [[ ! -e "${aug_surface}/report.json" ]]; then
  test ! -e "${aug_surface}/inference.jsonl"
  test ! -e "${aug_surface}/references.jsonl"
  PYTHONPATH=src .venv/bin/python -m edgemed_bench.randomize_training_options \
    --manifest "${source_surface}/inference.jsonl" \
    --references "${source_surface}/references.jsonl" \
    --output-manifest "${aug_surface}/inference.jsonl" \
    --output-references "${aug_surface}/references.jsonl" \
    --report "${aug_surface}/report.json" \
    --order-seed "${seed}" --selection-seed "${seed}" --selected-examples 256
fi

PYTHONPATH=src .venv/bin/python - "${aug_surface}/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
counts = report["selected_training_examples"]["answer_position_counts"]
assert report["count"] == 1968
assert report["leakage_boundary"]["inference_has_reference_fields"] is False
assert set(report["audit"]["shift_counts"]) == {"0", "1", "2", "3"}
assert sum(counts.values()) == 256
assert all(0.20 <= value / 256 <= 0.30 for value in counts.values()), counts
PY

test ! -e "${train_run}"
test ! -e "${reload_run}"
PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_qlora \
  --manifest "${aug_surface}/inference.jsonl" \
  --references "${aug_surface}/references.jsonl" \
  --data-root "${data_base}/extracted/figures" \
  --model-path "${model_path}" --model-source-manifest "${model_receipt}" \
  --run-dir "${train_run}" --target-mode option_text \
  --max-steps 2 --gradient-accumulation 2 --learning-rate 1e-4 \
  --lora-rank 16 --lora-alpha 32 --max-image-pixels 786432 \
  --grad-scaler-init-scale 1 --seed "${seed}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
  --kind mcq --manifest "${dev_surface}/inference.jsonl" \
  --data-root "${data_base}/extracted/figures" \
  --model-path "${model_path}" --model-source-manifest "${model_receipt}" \
  --adapter-path "${train_run}/adapter" \
  --adapter-source-manifest "${train_run}/run_manifest.json" \
  --run-dir "${reload_run}" --prompt-variant semantic_option \
  --max-new-tokens 64 --max-image-pixels 786432 --limit 4 --sync-every 1

PYTHONPATH=src .venv/bin/python - "${aug_surface}/report.json" "${train_run}" "${reload_run}" <<'PY'
import json
import math
import sys
from pathlib import Path

from edgemed_bench.io import read_jsonl, sha256_file

surface = json.load(open(sys.argv[1]))
train = Path(sys.argv[2])
reload = Path(sys.argv[3])
manifest = json.loads((train / "run_manifest.json").read_text())
predictions = read_jsonl(reload / "predictions.jsonl")
assert manifest["status"] == "completed"
assert manifest["contract"]["manifest_sha256"] == surface["source_hashes"]["output_manifest_sha256"]
assert manifest["contract"]["references_sha256"] == surface["source_hashes"]["output_references_sha256"]
assert manifest["optimizer_steps"] == 2 and manifest["loss"]["finite"] is True
assert math.isfinite(manifest["loss"]["first"]) and math.isfinite(manifest["loss"]["last"])
for relative, expected in manifest["adapter_hashes"].items():
    assert sha256_file(train / relative) == expected
assert len(predictions) == 4 and len({row["sample_id"] for row in predictions}) == 4
print(json.dumps({
    "status": "passed", "optimizer_steps": 2, "loss": manifest["loss"],
    "peak_cuda_mib": manifest["peak_cuda_mib"],
    "adapter_hashes": manifest["adapter_hashes"],
    "reload_parse_statuses": [row["parse_status"] for row in predictions],
}, sort_keys=True))
PY
