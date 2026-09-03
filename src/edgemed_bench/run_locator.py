"""Reference-blind locator-only inference with durable, exact-bound outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .io import append_jsonl, read_jsonl, reject_reference_fields, sha256_file, write_json
from .medical_agent import SYSTEM_CONTRACT
from .medical_agent_tools import MedicalToolExecutor, TOOL_SCHEMAS
from .run import git_commit, select_rows, utc_now
from .run_medical_agent import normalize_agent_sample


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _processed_ids(path: Path, selected_ids: set[str], contract_sha256: str) -> set[str]:
    if not path.exists():
        return set()
    processed = set()
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id"))
        if sample_id not in selected_ids or sample_id in processed:
            raise RuntimeError(f"Invalid existing locator prediction: {sample_id}")
        if row.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"Locator prediction contract mismatch: {sample_id}")
        processed.add(sample_id)
    return processed


def build_localizer_messages(
    row: dict[str, Any], data_root: Path, artifact_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = normalize_agent_sample(row)
    executor = MedicalToolExecutor(
        sample,
        data_root,
        artifact_dir,
        allowed_tools=("inspect_overview",),
        max_calls=1,
    )
    initial_call = {"name": "inspect_overview", "arguments": {"sample_count": 1}}
    overview, trace = executor.execute(initial_call["name"], initial_call["arguments"])
    if trace["status"] != "completed":
        raise RuntimeError(f"Overview failed: {trace.get('error')}")
    messages = [
        {"role": "system", "content": SYSTEM_CONTRACT},
        {
            "role": "user",
            "content": {
                "sample_id": sample["sample_id"],
                "question_type": sample["question_type"],
                "question": sample["question"],
                "options": sample.get("options"),
                "clinical_context": sample.get("clinical_context", ""),
                "media": sample["media"],
            },
        },
        {
            "role": "assistant",
            "content": "Acquire a low-resolution overview before planning targeted evidence.",
            "tool_call": initial_call,
            "policy_intervention": "initial_overview_required",
        },
        {
            "role": "tool",
            "tool_call_id": trace["trace_id"],
            "name": trace["tool_name"],
            "content": overview,
        },
    ]
    return messages, trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-manifest", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-source-manifest", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-image-pixels", type=int, default=786_432)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.adapter_path is None) != (args.adapter_source_manifest is None):
        raise ValueError("adapter path and source manifest must be provided together")
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.sample_id_file)
    reject_reference_fields(rows)
    if not rows or any(row.get("kind") != "localization" for row in rows):
        raise ValueError("Locator runner requires a non-empty localization surface")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate locator sample ids")

    from .qwen_agent_backend import Qwen35MedicalAgentBackend

    backend = Qwen35MedicalAgentBackend(
        args.model_path,
        args.model_source_manifest,
        decision_max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        adapter_path=args.adapter_path,
        adapter_source_manifest=args.adapter_source_manifest,
    )
    contract = {
        "schema_version": "edgemed-locator-inference-contract/v1",
        "manifest_sha256": sha256_file(args.manifest),
        "model_source_manifest_sha256": sha256_file(args.model_source_manifest),
        "prompt_contract_sha256": backend.receipt["prompt_contract_sha256"],
        "selected_count": len(rows),
        "selected_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "max_image_pixels": args.max_image_pixels,
        "do_sample": False,
        "thinking_mode": False,
        "quantization": "nf4-double-quant",
        "compute_dtype": "float16",
        "attention": "eager",
    }
    if "adapter_source" in backend.receipt:
        contract["adapter_source"] = backend.receipt["adapter_source"]
    contract_sha256 = _canonical_hash(contract)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    events_path = run_dir / "events.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"Locator predictions already exist: {predictions_path}")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("contract_sha256") != contract_sha256:
            raise RuntimeError("Locator resume contract differs")
        if existing.get("code_commit") != git_commit():
            raise RuntimeError("Locator resume code commit differs")
    processed = _processed_ids(predictions_path, set(sample_ids), contract_sha256)
    manifest = {
        "schema_version": "edgemed-locator-run/v1",
        "run_id": run_dir.name,
        "status": "running",
        "scientific_result": False,
        "started_at": utc_now(),
        "code_commit": git_commit(),
        "contract": contract,
        "contract_sha256": contract_sha256,
        "backend_receipt": backend.receipt,
    }
    write_json(manifest_path, manifest)
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "run_resumed" if processed else "run_started",
                "time": utc_now(),
                "selected": len(rows),
                "already_processed": len(processed),
                "contract_sha256": contract_sha256,
            },
            sync=True,
        )

    started = time.perf_counter()
    with predictions_path.open("a", encoding="utf-8") as output:
        for position, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            if sample_id in processed:
                continue
            sample_started = time.perf_counter()
            record: dict[str, Any] = {
                "schema_version": "edgemed-locator-prediction/v1",
                "sample_id": sample_id,
                "contract_sha256": contract_sha256,
            }
            try:
                messages, overview_trace = build_localizer_messages(
                    row, args.data_root, run_dir / "overview_artifacts"
                )
                turn = backend.localize(
                    messages, {"region_inspect": TOOL_SCHEMAS["region_inspect"]}
                )
                model_call = turn.pop("_model_call", None)
                record.update(
                    {
                        "status": "completed",
                        "content": turn.get("content"),
                        "tool_call": turn.get("tool_call"),
                        "model_call": model_call,
                        "overview_trace": overview_trace,
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            record["latency_seconds"] = time.perf_counter() - sample_started
            append_jsonl(output, record, sync=True)
            processed.add(sample_id)
            if position == 1 or position % 10 == 0 or position == len(rows):
                print(
                    f"PROGRESS processed={len(processed)}/{len(rows)} sample={sample_id}",
                    flush=True,
                )

    predictions = read_jsonl(predictions_path)
    failed = sum(row.get("status") != "completed" for row in predictions)
    manifest.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "processed_total": len(predictions),
            "failed_total": failed,
            "inference_seconds": time.perf_counter() - started,
            "predictions_sha256": sha256_file(predictions_path),
            "backend_runtime": backend.runtime_summary(),
        }
    )
    write_json(manifest_path, manifest)
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "run_completed",
                "time": utc_now(),
                "processed_total": len(predictions),
                "failed_total": failed,
            },
            sync=True,
        )
    os.chmod(predictions_path, 0o644)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
