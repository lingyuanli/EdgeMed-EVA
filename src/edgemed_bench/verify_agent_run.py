"""Deterministic integrity and metric-recomputation verifier for Agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, reject_reference_fields, sha256_file, write_json
from .score_agent import score_agent_rows


def _check(name: str, passed: bool, **evidence: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "BLOCK", **evidence}


def _metric_at_path(metrics: dict[str, Any], dotted_path: str) -> Any:
    value: Any = metrics
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _evaluate_quality_gates(
    metrics: dict[str, Any], traces: list[dict[str, Any]], gates: Any
) -> tuple[bool, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    if not isinstance(gates, dict):
        return False, [{"gate": "quality_gates", "error": "missing_or_not_an_object"}]
    unknown = set(gates) - {"metric_minimums", "max_failed_tool_calls"}
    if unknown:
        failures.append({"gate": "quality_gates", "error": f"unknown_fields:{sorted(unknown)}"})
    minimums = gates.get("metric_minimums")
    if not isinstance(minimums, dict) or not minimums:
        failures.append({"gate": "metric_minimums", "error": "missing_or_empty"})
    else:
        for path, threshold in minimums.items():
            try:
                actual = _metric_at_path(metrics, str(path))
                passed = isinstance(actual, (int, float)) and not isinstance(actual, bool)
                passed = passed and float(actual) >= float(threshold)
            except (KeyError, TypeError, ValueError) as exc:
                failures.append({"gate": path, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not passed:
                failures.append({"gate": path, "minimum": threshold, "actual": actual})
    maximum_failed = gates.get("max_failed_tool_calls")
    if not isinstance(maximum_failed, int) or isinstance(maximum_failed, bool) or maximum_failed < 0:
        failures.append({"gate": "max_failed_tool_calls", "error": "missing_or_invalid"})
    else:
        failed_count = sum(trace.get("status") != "completed" for trace in traces)
        if failed_count > maximum_failed:
            failures.append(
                {"gate": "max_failed_tool_calls", "maximum": maximum_failed, "actual": failed_count}
            )
    return not failures, failures


def verify_agent_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    required = {
        "run_manifest": run_dir / "run_manifest.json",
        "inference_manifest": run_dir / "inference_manifest.jsonl",
        "references": run_dir / "references.jsonl",
        "predictions": run_dir / "predictions.jsonl",
        "tool_traces": run_dir / "tool_traces.jsonl",
        "trajectories": run_dir / "trajectories.jsonl",
        "metrics": run_dir / "metrics.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "schema_version": "edgemed-medical-agent-verification/v1",
            "overall": "BLOCK",
            "checks": [_check("required_files", False, missing=missing)],
        }

    manifest = json.loads(required["run_manifest"].read_text())
    metrics = json.loads(required["metrics"].read_text())
    inference = read_jsonl(required["inference_manifest"])
    references = read_jsonl(required["references"])
    predictions = read_jsonl(required["predictions"])
    traces = read_jsonl(required["tool_traces"])
    trajectories = read_jsonl(required["trajectories"])
    checks = []
    try:
        reject_reference_fields(inference)
        boundary_ok = True
        boundary_error = None
    except Exception as exc:
        boundary_ok = False
        boundary_error = str(exc)
    checks.append(_check("reference_isolation", boundary_ok, error=boundary_error))

    expected_sources = manifest.get("source_hashes", {})
    actual_sources = {
        "inference_manifest_sha256": sha256_file(required["inference_manifest"]),
        "references_sha256": sha256_file(required["references"]),
    }
    checks.append(
        _check("source_hash_binding", expected_sources == actual_sources, expected=expected_sources, actual=actual_sources)
    )
    expected_outputs = manifest.get("output_hashes", {})
    actual_outputs = {
        "predictions_sha256": sha256_file(required["predictions"]),
        "tool_traces_sha256": sha256_file(required["tool_traces"]),
        "trajectories_sha256": sha256_file(required["trajectories"]),
        "metrics_sha256": sha256_file(required["metrics"]),
    }
    checks.append(
        _check("output_hash_binding", expected_outputs == actual_outputs, expected=expected_outputs, actual=actual_outputs)
    )
    ids = [{str(row["sample_id"]) for row in rows} for rows in (inference, references, predictions, trajectories)]
    coverage_ok = bool(ids[0]) and all(item == ids[0] for item in ids[1:])
    checks.append(_check("sample_coverage", coverage_ok, sample_count=len(ids[0])))

    artifact_failures = []
    for trace in traces:
        if trace.get("status") != "completed":
            continue
        path = Path(str(trace.get("output_artifact", ""))).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError:
            artifact_failures.append({"trace_id": trace.get("trace_id"), "error": "outside_run_dir"})
            continue
        if not path.is_file() or sha256_file(path) != trace.get("output_sha256"):
            artifact_failures.append({"trace_id": trace.get("trace_id"), "error": "missing_or_hash_mismatch"})
    checks.append(_check("tool_artifact_integrity", not artifact_failures, failures=artifact_failures))

    try:
        recomputed = score_agent_rows(inference, references, predictions, traces)
        recorded_core = {key: value for key, value in metrics.items() if key != "source_hashes"}
        recompute_ok = recomputed == recorded_core
        recompute_error = None
    except Exception as exc:
        recompute_ok = False
        recompute_error = f"{type(exc).__name__}: {exc}"
    checks.append(_check("metric_recompute", recompute_ok, error=recompute_error))
    quality_ok, quality_failures = _evaluate_quality_gates(
        metrics, traces, manifest.get("quality_gates")
    )
    checks.append(
        _check(
            "declared_quality_gates",
            quality_ok,
            gates=manifest.get("quality_gates"),
            failures=quality_failures,
        )
    )
    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "BLOCK"
    return {
        "schema_version": "edgemed-medical-agent-verification/v1",
        "run_id": manifest.get("run_id", run_dir.name),
        "overall": overall,
        "checks": checks,
        "medical_correctness": {
            "status": "DEFER",
            "reason": "Integrity checks and references do not replace expert adjudication.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = verify_agent_run(args.run_dir)
    write_json(args.run_dir / "verifier_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
