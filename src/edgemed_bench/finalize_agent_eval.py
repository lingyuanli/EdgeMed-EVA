"""Attach scorer-only references after inference, score, and verify an Agent run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .io import append_jsonl, read_jsonl, sha256_file, write_json
from .score_agent import score_agent_rows
from .verify_agent_run import verify_agent_run


def _select_references(
    inference_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in reference_rows}
    if len(by_id) != len(reference_rows):
        raise ValueError("Duplicate reference sample ids")
    missing = [row["sample_id"] for row in inference_rows if row["sample_id"] not in by_id]
    if missing:
        raise KeyError(f"Missing references: {missing[:10]}")
    return [by_id[row["sample_id"]] for row in inference_rows]


def _write_references(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        if read_jsonl(path) != rows:
            raise RuntimeError("Existing scorer-only references differ from requested references")
        return
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row, sync=True)
    os.chmod(path, 0o600)


def finalize_agent_evaluation(run_dir: Path, references_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "inference_completed":
        raise RuntimeError("Agent inference must be completed before references are attached")
    inference = read_jsonl(run_dir / "inference_manifest.jsonl")
    references = _select_references(inference, read_jsonl(references_path))
    run_references = run_dir / "references.jsonl"
    _write_references(run_references, references)
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    traces = read_jsonl(run_dir / "tool_traces.jsonl")
    metrics = score_agent_rows(inference, references, predictions, traces)
    metrics["source_hashes"] = {
        "inference_manifest_sha256": sha256_file(run_dir / "inference_manifest.jsonl"),
        "references_sha256": sha256_file(run_references),
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "tool_traces_sha256": sha256_file(run_dir / "tool_traces.jsonl"),
    }
    write_json(run_dir / "metrics.json", metrics)
    manifest["status"] = "completed"
    manifest["source_hashes"]["references_sha256"] = sha256_file(run_references)
    manifest["output_hashes"] = {
        "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        "tool_traces_sha256": sha256_file(run_dir / "tool_traces.jsonl"),
        "trajectories_sha256": sha256_file(run_dir / "trajectories.jsonl"),
        "metrics_sha256": sha256_file(run_dir / "metrics.json"),
    }
    write_json(manifest_path, manifest)
    report = verify_agent_run(run_dir)
    write_json(run_dir / "verifier_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_agent_evaluation(args.run_dir, args.references)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
