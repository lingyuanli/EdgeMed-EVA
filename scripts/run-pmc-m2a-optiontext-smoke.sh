#!/usr/bin/env bash
set -euo pipefail

repo_root="${EDGEMED_REPO_ROOT:-/home/ubuntu/EdgeMed-EVA}"
data_base="${EDGEMED_PMC_DATA_BASE:-/home/ubuntu/data/external/pmc-vqa-b56ae594}"
train_surface="${EDGEMED_PMC_TRAIN_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-train-admitted-1968}"
dev_surface="${EDGEMED_PMC_DEV_SURFACES:-/home/ubuntu/data/external/surfaces/pmc-vqa-mcq-dev-512}"
model_path="${EDGEMED_MODEL_PATH:-/home/ubuntu/models/Qwen3.5-4B}"
model_receipt="${repo_root}/baselines/local/qwen35-4b-medcmr-b0/source_manifest.json"
seed="${EDGEMED_SEED:-20260903}"
train_run="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-qlora-smoke2-s${seed}"
reload_run="${repo_root}/runs/qwen35-4b-pmc-m2a-optiontext-reload4-s${seed}"

cd "${repo_root}"
test ! -e "${train_run}"
test ! -e "${reload_run}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.train_qlora \
  --manifest "${train_surface}/inference.jsonl" \
  --references "${train_surface}/references.jsonl" \
  --data-root "${data_base}/extracted/figures" \
  --model-path "${model_path}" \
  --model-source-manifest "${model_receipt}" \
  --run-dir "${train_run}" \
  --target-mode option_text \
  --max-steps 2 \
  --gradient-accumulation 2 \
  --learning-rate 1e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --max-image-pixels 786432 \
  --grad-scaler-init-scale 1 \
  --seed "${seed}"

PYTHONPATH=src .venv/bin/python -m edgemed_bench.run \
  --kind mcq \
  --manifest "${dev_surface}/inference.jsonl" \
  --data-root "${data_base}/extracted/figures" \
  --model-path "${model_path}" \
  --model-source-manifest "${model_receipt}" \
  --adapter-path "${train_run}/adapter" \
  --adapter-source-manifest "${train_run}/run_manifest.json" \
  --run-dir "${reload_run}" \
  --prompt-variant semantic_option \
  --max-new-tokens 64 \
  --max-image-pixels 786432 \
  --limit 4 \
  --sync-every 1

PYTHONPATH=src .venv/bin/python - "${train_run}" "${reload_run}" <<'PY'
import json
import math
import sys
from pathlib import Path

from edgemed_bench.io import read_jsonl, sha256_file

train = Path(sys.argv[1])
reload = Path(sys.argv[2])
manifest = json.loads((train / "run_manifest.json").read_text())
predictions = read_jsonl(reload / "predictions.jsonl")
assert manifest["status"] == "completed"
assert manifest["contract"]["target_mode"] == "option_text"
assert manifest["optimizer_steps"] == 2
assert manifest["loss"]["finite"] is True
assert math.isfinite(manifest["loss"]["first"])
assert math.isfinite(manifest["loss"]["last"])
assert manifest["adapter_hashes"]
for relative, expected in manifest["adapter_hashes"].items():
    assert sha256_file(train / relative) == expected
assert len(predictions) == 4
assert len({row["sample_id"] for row in predictions}) == 4
print(json.dumps({
    "status": "passed",
    "optimizer_steps": manifest["optimizer_steps"],
    "loss": manifest["loss"],
    "peak_cuda_mib": manifest["peak_cuda_mib"],
    "adapter_hashes": manifest["adapter_hashes"],
    "reload_parse_statuses": [row["parse_status"] for row in predictions],
}, sort_keys=True))
PY
