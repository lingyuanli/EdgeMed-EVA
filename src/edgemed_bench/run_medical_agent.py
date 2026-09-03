"""Reference-blind, exact-resume batch runner for the medical multimodal Agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import append_jsonl, read_jsonl, reject_reference_fields, sha256_file, write_json
from .medical_agent import AgentBackend, FINAL_SCHEMA, run_medical_agent
from .medical_agent_tools import TOOL_SCHEMAS
from .run import git_commit, select_rows


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_agent_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an existing single-image benchmark row without adding references."""
    reject_reference_fields([row])
    if isinstance(row.get("media"), list):
        normalized = dict(row)
        normalized.setdefault("question_type", normalized.pop("kind", "mcq"))
        return normalized
    required = {"sample_id", "question", "image_path", "image_sha256"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Agent inference row is missing fields: {sorted(missing)}")
    question_type = str(row.get("question_type", row.get("kind", "mcq")))
    return {
        "sample_id": str(row["sample_id"]),
        "question_type": question_type,
        "question": row["question"],
        "options": row.get("options"),
        "clinical_context": row.get("clinical_context", ""),
        "media": [
            {
                "media_id": "image-0",
                "kind": "image",
                "path": row["image_path"],
                "sha256": row["image_sha256"],
                "modality": row.get("modality", "unknown"),
                "view": row.get("view", "unknown"),
                "timepoint": row.get("timepoint", "unknown"),
            }
        ],
        "task": row.get("task", question_type),
        "source_dataset": row.get("source_dataset"),
        "source_version": row.get("source_version"),
        "source_record_id": row.get("source_record_id"),
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                append_jsonl(handle, row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_path(checkpoint_dir: Path, sample_id: str) -> Path:
    return checkpoint_dir / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json"


def _backend_runtime_summary(backend: AgentBackend) -> dict[str, Any] | None:
    summarize = getattr(backend, "runtime_summary", None)
    if not callable(summarize):
        return None
    summary = summarize()
    if not isinstance(summary, dict):
        raise TypeError("Backend runtime_summary() must return a dictionary")
    return summary


def _load_checkpoints(
    rows: list[dict[str, Any]], checkpoint_dir: Path, contract_sha256: str
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = _checkpoint_path(checkpoint_dir, row["sample_id"])
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("sample_id") != row["sample_id"]:
            raise RuntimeError(f"Checkpoint sample mismatch: {path}")
        if checkpoint.get("contract_sha256") != contract_sha256:
            raise RuntimeError(f"Checkpoint contract mismatch: {row['sample_id']}")
        if checkpoint.get("status") != "completed":
            raise RuntimeError(f"Checkpoint is not completed: {row['sample_id']}")
        completed[row["sample_id"]] = checkpoint
    return completed


def _materialize(
    rows: list[dict[str, Any]], checkpoints: dict[str, dict[str, Any]], run_dir: Path
) -> dict[str, str]:
    predictions = [checkpoints[row["sample_id"]]["prediction"] for row in rows]
    trajectories = [checkpoints[row["sample_id"]]["trajectory"] for row in rows]
    traces = [
        trace
        for row in rows
        for trace in checkpoints[row["sample_id"]]["tool_traces"]
    ]
    paths = {
        "predictions": run_dir / "predictions.jsonl",
        "trajectories": run_dir / "trajectories.jsonl",
        "tool_traces": run_dir / "tool_traces.jsonl",
    }
    _atomic_jsonl(paths["predictions"], predictions)
    _atomic_jsonl(paths["trajectories"], trajectories)
    _atomic_jsonl(paths["tool_traces"], traces)
    return {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}


def run_agent_batch(
    rows: list[dict[str, Any]],
    backend: AgentBackend,
    *,
    data_root: Path,
    run_dir: Path,
    contract: dict[str, Any],
    allowed_tools: tuple[str, ...],
    max_steps: int,
    initial_visual_policy: str = "none",
    resume: bool = False,
    interrupt_after: int | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No Agent rows selected")
    normalized = [normalize_agent_sample(row) for row in rows]
    sample_ids = [row["sample_id"] for row in normalized]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Duplicate sample ids")
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    inference_path = run_dir / "inference_manifest.jsonl"
    events_path = run_dir / "events.jsonl"
    checkpoint_dir = run_dir / "sample_checkpoints"
    contract_sha256 = _canonical_hash(contract)
    current_code_commit = git_commit()
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Run already exists; pass resume=True: {run_dir}")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("code_commit") != current_code_commit:
            raise RuntimeError("Resume code commit differs from existing run")
        if previous.get("contract_sha256") != contract_sha256:
            raise RuntimeError("Resume contract differs from existing run")
        if not inference_path.is_file() or sha256_file(inference_path) != previous["source_hashes"]["inference_manifest_sha256"]:
            raise RuntimeError("Resume inference manifest differs from existing run")
    else:
        _atomic_jsonl(inference_path, normalized)
        previous = {}

    checkpoints = _load_checkpoints(normalized, checkpoint_dir, contract_sha256)
    manifest = {
        "schema_version": "edgemed-medical-agent-run/v1",
        "run_id": run_dir.name,
        "status": "running",
        "scientific_result": False,
        "started_at": previous.get("started_at", utc_now()),
        "resume_count": int(previous.get("resume_count", 0)) + int(bool(previous)),
        "code_commit": current_code_commit,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "quality_gates": {
            "metric_minimums": {
                "e0_structure.schema_valid_rate": 1.0,
                "e0_structure.citation_valid_rate": 1.0,
                "e0_structure.tool_trace_bound_rate": 1.0,
            },
            "max_failed_tool_calls": 0,
        },
        "source_hashes": {"inference_manifest_sha256": sha256_file(inference_path)},
        "backend_receipt": getattr(backend, "receipt", {"backend": type(backend).__name__}),
        "environment": {"python": platform.python_version()},
    }
    _atomic_json(manifest_path, manifest)
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "run_resumed" if checkpoints else "run_started",
                "time": utc_now(),
                "selected": len(normalized),
                "already_completed": len(checkpoints),
                "contract_sha256": contract_sha256,
            },
            sync=True,
        )
    completed_this_process = 0
    try:
        for row in normalized:
            sample_id = row["sample_id"]
            if sample_id in checkpoints:
                continue
            started = time.perf_counter()
            result = run_medical_agent(
                row,
                backend,
                data_root,
                run_dir / "tool_artifacts",
                allowed_tools=allowed_tools,
                max_steps=max_steps,
                initial_visual_policy=initial_visual_policy,
            )
            checkpoint = {
                "schema_version": "edgemed-medical-agent-sample-checkpoint/v1",
                "sample_id": sample_id,
                "status": "completed",
                "contract_sha256": contract_sha256,
                "latency_seconds": time.perf_counter() - started,
                "prediction": result.prediction,
                "trajectory": result.trajectory,
                "tool_traces": result.tool_traces,
            }
            _atomic_json(_checkpoint_path(checkpoint_dir, sample_id), checkpoint)
            checkpoints[sample_id] = checkpoint
            completed_this_process += 1
            with events_path.open("a", encoding="utf-8") as events:
                append_jsonl(
                    events,
                    {
                        "event": "sample_completed",
                        "time": utc_now(),
                        "sample_id": sample_id,
                        "completed_total": len(checkpoints),
                    },
                    sync=True,
                )
            if interrupt_after is not None and completed_this_process >= interrupt_after:
                raise InterruptedError("Injected interruption after complete sample checkpoint")
    except BaseException as exc:
        backend_runtime = _backend_runtime_summary(backend)
        manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now(),
                "error_type": type(exc).__name__,
                "completed_total": len(checkpoints),
            }
        )
        if backend_runtime is not None:
            manifest["backend_runtime"] = backend_runtime
        _atomic_json(manifest_path, manifest)
        with events_path.open("a", encoding="utf-8") as events:
            append_jsonl(
                events,
                {
                    "event": "run_failed",
                    "time": utc_now(),
                    "error_type": type(exc).__name__,
                    "completed_total": len(checkpoints),
                },
                sync=True,
            )
        raise
    output_hashes = _materialize(normalized, checkpoints, run_dir)
    backend_runtime = _backend_runtime_summary(backend)
    manifest.update(
        {
            "status": "inference_completed",
            "finished_at": utc_now(),
            "completed_total": len(checkpoints),
            "output_hashes": output_hashes,
        }
    )
    if backend_runtime is not None:
        manifest["backend_runtime"] = backend_runtime
    _atomic_json(manifest_path, manifest)
    with events_path.open("a", encoding="utf-8") as events:
        append_jsonl(
            events,
            {
                "event": "inference_completed",
                "time": utc_now(),
                "completed_total": len(checkpoints),
                "contract_sha256": contract_sha256,
            },
            sync=True,
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-source-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id-file", type=Path)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--initial-visual-policy", choices=("none", "overview"), default="none"
    )
    parser.add_argument("--decision-max-new-tokens", type=int, default=192)
    parser.add_argument("--final-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-image-pixels", type=int, default=786_432)
    parser.add_argument("--tools", nargs="+", choices=tuple(TOOL_SCHEMAS), default=["inspect_overview", "region_inspect"])
    parser.add_argument("--skip-weight-verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and (args.limit < 1 or args.limit > 16):
        raise ValueError("Operational M1 runner limit must be in [1,16]")
    rows = select_rows(read_jsonl(args.manifest), args.limit, args.sample_id_file)
    if len(rows) > 16:
        raise ValueError("Operational M1 runner cannot select more than 16 samples")
    normalized = [normalize_agent_sample(row) for row in rows]
    selected_ids_sha256 = hashlib.sha256("\n".join(row["sample_id"] for row in normalized).encode()).hexdigest()
    from .qwen_agent_backend import Qwen35MedicalAgentBackend

    backend = Qwen35MedicalAgentBackend(
        args.model_path,
        args.model_source_manifest,
        decision_max_new_tokens=args.decision_max_new_tokens,
        final_max_new_tokens=args.final_max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        verify_weights=not args.skip_weight_verification,
    )
    contract = {
        "stage": "m1-operational-smoke",
        "input_manifest_sha256": sha256_file(args.manifest),
        "selected_count": len(rows),
        "selected_ids_sha256": selected_ids_sha256,
        "data_root": str(args.data_root.resolve()),
        "model_source_manifest_sha256": sha256_file(args.model_source_manifest),
        "backend": backend.receipt["backend"],
        "prompt_contract_sha256": backend.receipt.get("prompt_contract_sha256", "test-or-external"),
        "final_schema_sha256": _canonical_hash(FINAL_SCHEMA),
        "tool_schemas_sha256": _canonical_hash(
            {name: TOOL_SCHEMAS[name] for name in args.tools}
        ),
        "generation": backend.receipt["generation"],
        "allowed_tools": list(args.tools),
        "max_steps": args.max_steps,
        "initial_visual_policy": args.initial_visual_policy,
    }
    manifest = run_agent_batch(
        rows,
        backend,
        data_root=args.data_root,
        run_dir=args.run_dir,
        contract=contract,
        allowed_tools=tuple(args.tools),
        max_steps=args.max_steps,
        initial_visual_policy=args.initial_visual_policy,
        resume=args.resume,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
