"""Reference-free operational analysis for medical Agent runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .io import read_jsonl, sha256_file, write_json
from .score_agent import _schema_valid


def _unique(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key}")
    return result


def _region_area(trace: dict[str, Any]) -> float | None:
    if trace.get("tool_name") != "region_inspect" or trace.get("status") != "completed":
        return None
    box = trace.get("request", {}).get("region_xyxy_1000")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / 1_000_000


def analyze_agent_operations(
    run_dir: Path,
    *,
    target_area_min: float = 0.01,
    target_area_max: float = 0.64,
    target_rate_min: float = 0.30,
) -> dict[str, Any]:
    if not 0 <= target_area_min <= target_area_max <= 1:
        raise ValueError("Invalid target area interval")
    if not 0 <= target_rate_min <= 1:
        raise ValueError("target_rate_min must be in [0,1]")
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    inference = read_jsonl(run_dir / "inference_manifest.jsonl")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    traces = read_jsonl(run_dir / "tool_traces.jsonl")
    trajectories = read_jsonl(run_dir / "trajectories.jsonl")
    samples = _unique(inference, "sample_id")
    by_prediction = _unique(predictions, "sample_id")
    by_trajectory = _unique(trajectories, "sample_id")
    trace_map = _unique(traces, "trace_id")
    coverage_ok = set(samples) == set(by_prediction) == set(by_trajectory)

    schema_valid = citation_valid = trace_bound = 0
    for sample_id, sample in samples.items():
        prediction = by_prediction.get(sample_id, {})
        valid, cited, bound = _schema_valid(sample, prediction, trace_map)
        schema_valid += int(valid)
        citation_valid += int(cited)
        trace_bound += int(bound)
    count = len(samples)

    areas = [area for trace in traces if (area := _region_area(trace)) is not None]
    targeted_samples = {
        str(trace["sample_id"])
        for trace in traces
        if (area := _region_area(trace)) is not None
        and target_area_min <= area <= target_area_max
    }
    artifact_pixels = 0
    artifact_failures = []
    for trace in traces:
        if trace.get("status") != "completed":
            continue
        path = Path(str(trace.get("output_artifact", ""))).resolve()
        try:
            path.relative_to(run_dir)
        except ValueError:
            artifact_failures.append(
                {"trace_id": trace.get("trace_id"), "error": "outside_run_dir"}
            )
            continue
        if not path.is_file() or sha256_file(path) != trace.get("output_sha256"):
            artifact_failures.append(
                {"trace_id": trace.get("trace_id"), "error": "missing_or_hash_mismatch"}
            )
            continue
        with Image.open(path) as image:
            artifact_pixels += image.width * image.height

    model_calls = []
    interventions = Counter()
    normalizations = Counter()
    for trajectory in trajectories:
        for message in trajectory.get("messages", []):
            if isinstance(message.get("model_call"), dict):
                model_calls.append(message["model_call"])
            if isinstance(message.get("policy_intervention"), str):
                interventions[message["policy_intervention"]] += 1
            for item in message.get("policy_normalizations", []):
                if isinstance(item, dict) and isinstance(item.get("rule"), str):
                    normalizations[item["rule"]] += 1

    failed = [trace for trace in traces if trace.get("status") != "completed"]
    e0 = {
        "schema_valid_rate": schema_valid / count if count else 0.0,
        "citation_valid_rate": citation_valid / count if count else 0.0,
        "tool_trace_bound_rate": trace_bound / count if count else 0.0,
    }
    targeted_rate = len(targeted_samples) / count if count else 0.0
    gates = {
        "inference_completed": manifest.get("status") == "inference_completed",
        "sample_coverage": coverage_ok and count == manifest.get("completed_total"),
        "e0_all_one": all(value == 1.0 for value in e0.values()),
        "zero_failed_tools": not failed,
        "tool_artifact_integrity": not artifact_failures,
        "targeted_roi_rate": targeted_rate >= target_rate_min,
    }
    return {
        "schema_version": "edgemed-medical-agent-operational-analysis/v1",
        "run_id": manifest.get("run_id", run_dir.name),
        "overall": "PASS" if all(gates.values()) else "BLOCK",
        "gates": gates,
        "gate_contract": {
            "target_area_interval": [target_area_min, target_area_max],
            "targeted_sample_rate_minimum": target_rate_min,
        },
        "sample_count": count,
        "e0_structure": e0,
        "tools": {
            "calls": len(traces),
            "completed": len(traces) - len(failed),
            "failed": len(failed),
            "by_name": dict(sorted(Counter(str(trace.get("tool_name")) for trace in traces).items())),
            "region_areas": areas,
            "targeted_sample_count": len(targeted_samples),
            "targeted_sample_rate": targeted_rate,
            "artifact_pixels_total": artifact_pixels,
            "artifact_failures": artifact_failures,
        },
        "model_calls": {
            "count": len(model_calls),
            "input_tokens": sum(int(call.get("input_tokens", 0)) for call in model_calls),
            "output_tokens": sum(int(call.get("output_tokens", 0)) for call in model_calls),
            "latency_seconds": sum(float(call.get("latency_seconds", 0)) for call in model_calls),
        },
        "policy_interventions": dict(sorted(interventions.items())),
        "policy_normalizations": dict(sorted(normalizations.items())),
        "backend_runtime": manifest.get("backend_runtime"),
        "source_hashes": {
            "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
            "inference_manifest_sha256": sha256_file(run_dir / "inference_manifest.jsonl"),
            "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
            "tool_traces_sha256": sha256_file(run_dir / "tool_traces.jsonl"),
            "trajectories_sha256": sha256_file(run_dir / "trajectories.jsonl"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-area-min", type=float, default=0.01)
    parser.add_argument("--target-area-max", type=float, default=0.64)
    parser.add_argument("--target-rate-min", type=float, default=0.30)
    args = parser.parse_args()
    report = analyze_agent_operations(
        args.run_dir,
        target_area_min=args.target_area_min,
        target_area_max=args.target_area_max,
        target_rate_min=args.target_rate_min,
    )
    output = args.output or args.run_dir / "operational_analysis.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
