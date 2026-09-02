"""Run a synthetic, non-scientific fixture through the complete Agent artifact loop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .io import append_jsonl, sha256_file, write_json
from .medical_agent import run_medical_agent
from .score_agent import score_agent_rows
from .verify_agent_run import verify_agent_run


class FixtureBackend:
    """A deterministic scripted backend used only to test plumbing, never efficacy."""

    def __init__(self) -> None:
        self.step = 0

    def decide(self, messages: list[dict[str, Any]], tools: dict[str, Any]) -> dict[str, Any]:
        calls = [
            ("inspect_overview", {"sample_count": 3}),
            ("temporal_skim", {"media_id": "study", "start_time": 0, "end_time": 2, "sample_count": 3}),
            (
                "region_inspect",
                {
                    "media_id": "study",
                    "timestamp": 1,
                    "region_xyxy_1000": [300, 300, 700, 700],
                    "target": "check whether the central region changes at the middle timepoint",
                },
            ),
        ]
        if self.step >= len(calls):
            return {"content": "The visual evidence is sufficient for the fixture."}
        name, arguments = calls[self.step]
        self.step += 1
        if name not in tools:
            raise RuntimeError(f"Fixture requested disabled tool: {name}")
        return {"content": "Acquire the next bounded visual observation.", "tool_call": {"name": name, "arguments": arguments}}

    def finalize(self, messages: list[dict[str, Any]], output_schema: dict[str, Any]) -> dict[str, Any]:
        trace_ids = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
        return {
            "sample_id": "fixture-001",
            "hypotheses": [
                {"id": "H1", "label": "central finding appears at middle timepoint", "status": "supported"},
                {"id": "H2", "label": "central finding is absent throughout", "status": "refuted"},
            ],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "media_id": "study",
                    "view_or_time": "1.0s",
                    "region_xyxy_1000": [300, 300, 700, 700],
                    "observation": "A colored central square is visible at the middle timepoint.",
                    "supports": ["H1"],
                    "contradicts": ["H2"],
                    "acquisition": "region_inspect",
                    "confidence": 1.0,
                }
            ],
            "answer": "A",
            "answer_text": "The central finding appears only at the middle timepoint.",
            "answer_evidence_ids": ["E1"],
            "confidence": 1.0,
            "insufficient_evidence": False,
            "tool_trace_ids": trace_ids,
        }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row, sync=True)


def run_fixture(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Fixture output must be empty: {output_dir}")
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(3):
        path = data_dir / f"frame-{index}.png"
        image = Image.new("RGB", (96, 96), "black")
        if index == 1:
            ImageDraw.Draw(image).rectangle((29, 29, 67, 67), fill="red")
        image.save(path, format="PNG", optimize=False)
        frames.append({
            "path": path.relative_to(output_dir).as_posix(),
            "timestamp": float(index),
            "sha256": sha256_file(path),
        })
    inference = [
        {
            "sample_id": "fixture-001",
            "question_type": "mcq",
            "question": "When is the central colored finding visible?",
            "options": {"A": "middle only", "B": "all timepoints", "C": "never", "D": "final only"},
            "clinical_context": "Synthetic plumbing fixture; no clinical meaning.",
            "media": [{"media_id": "study", "kind": "image_sequence", "modality": "synthetic", "view": "axial", "frames": frames}],
        }
    ]
    references = [
        {
            "sample_id": "fixture-001",
            "answer": "A",
            "evidence": [{"media_id": "study", "timestamp": 1.0, "region_xyxy_1000": [300, 300, 700, 700]}],
        }
    ]
    inference_path = output_dir / "inference_manifest.jsonl"
    references_path = output_dir / "references.jsonl"
    _write_jsonl(inference_path, inference)
    _write_jsonl(references_path, references)
    result = run_medical_agent(
        inference[0], FixtureBackend(), output_dir, output_dir / "tool_artifacts", max_steps=4
    )
    predictions_path = output_dir / "predictions.jsonl"
    traces_path = output_dir / "tool_traces.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    _write_jsonl(predictions_path, [result.prediction])
    _write_jsonl(traces_path, result.tool_traces)
    _write_jsonl(trajectories_path, [result.trajectory])
    metrics = score_agent_rows(inference, references, [result.prediction], result.tool_traces)
    metrics["source_hashes"] = {
        "inference_manifest_sha256": sha256_file(inference_path),
        "references_sha256": sha256_file(references_path),
        "predictions_sha256": sha256_file(predictions_path),
        "tool_traces_sha256": sha256_file(traces_path),
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "edgemed-medical-agent-run/v1",
            "run_id": output_dir.name,
            "status": "completed",
            "scientific_result": False,
            "backend": "deterministic_fixture",
            "quality_gates": {
                "metric_minimums": {
                    "e0_structure.schema_valid_rate": 1.0,
                    "e0_structure.citation_valid_rate": 1.0,
                    "e0_structure.tool_trace_bound_rate": 1.0,
                },
                "max_failed_tool_calls": 0,
            },
            "source_hashes": {
                "inference_manifest_sha256": sha256_file(inference_path),
                "references_sha256": sha256_file(references_path),
            },
            "output_hashes": {
                "predictions_sha256": sha256_file(predictions_path),
                "tool_traces_sha256": sha256_file(traces_path),
                "trajectories_sha256": sha256_file(trajectories_path),
                "metrics_sha256": sha256_file(metrics_path),
            },
        },
    )
    event_time = datetime.now(timezone.utc).isoformat()
    _write_jsonl(
        output_dir / "events.jsonl",
        [
            {"seq": 1, "time": event_time, "event": "run_started", "sample_cursor": 0},
            {"seq": 2, "time": event_time, "event": "sample_completed", "sample_cursor": 1},
            {"seq": 3, "time": event_time, "event": "evaluation_completed", "sample_cursor": 1},
        ],
    )
    report = verify_agent_run(output_dir)
    write_json(output_dir / "verifier_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_fixture(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
