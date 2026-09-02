import json
from pathlib import Path

import pytest
from PIL import Image

from edgemed_bench.medical_agent import run_medical_agent
from edgemed_bench.medical_agent_fixture import FixtureBackend, run_fixture
from edgemed_bench.medical_agent_tools import MedicalToolExecutor
from edgemed_bench.score_agent import score_agent_rows
from edgemed_bench.verify_agent_run import _evaluate_quality_gates, verify_agent_run


def _sample(tmp_path: Path) -> dict:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "gray").save(image_path)
    return {
        "sample_id": "s1",
        "question_type": "mcq",
        "question": "Fixture?",
        "options": {"A": "yes", "B": "no"},
        "media": [{"media_id": "m1", "kind": "image", "path": "image.png"}],
    }


def test_tool_executor_enforces_allowlist_and_records_failure(tmp_path: Path) -> None:
    executor = MedicalToolExecutor(_sample(tmp_path), tmp_path, tmp_path / "artifacts", allowed_tools=("inspect_overview",))
    result, trace = executor.execute(
        "region_inspect",
        {"media_id": "m1", "region_xyxy_1000": [0, 0, 500, 500], "target": "fixture"},
    )
    assert result["status"] == "failed"
    assert trace["status"] == "failed"
    assert "not enabled" in trace["error"]


def test_controller_blocks_finalization_without_visual_evidence(tmp_path: Path) -> None:
    class PrematureBackend:
        def decide(self, messages, tools):
            return {"content": "ready"}

        def finalize(self, messages, output_schema):
            raise AssertionError("finalizer must not run")

    with pytest.raises(RuntimeError, match="no successful visual evidence"):
        run_medical_agent(
            _sample(tmp_path), PrematureBackend(), tmp_path, tmp_path / "artifacts",
            allowed_tools=("region_inspect",),
        )


def test_controller_forces_auditable_overview_when_backend_stops_before_evidence(
    tmp_path: Path,
) -> None:
    class PrematureThenFinalBackend:
        def decide(self, messages, tools):
            return {"content": "evidence is sufficient", "tool_call": None}

        def finalize(self, messages, output_schema):
            return {"sample_id": "s1", "answer": "A"}

    result = run_medical_agent(
        _sample(tmp_path),
        PrematureThenFinalBackend(),
        tmp_path,
        tmp_path / "artifacts",
        allowed_tools=("inspect_overview", "region_inspect"),
        max_steps=2,
    )
    interventions = [
        message for message in result.trajectory["messages"]
        if message.get("policy_intervention") == "first_visual_acquisition_required"
    ]
    assert len(interventions) == 1
    assert interventions[0]["tool_call"] == {
        "name": "inspect_overview", "arguments": {"sample_count": 1}
    }
    assert result.tool_traces[0]["status"] == "completed"
    assert result.tool_traces[0]["tool_name"] == "inspect_overview"


def test_controller_records_non_region_box_canonicalization(tmp_path: Path) -> None:
    class OverviewBackend:
        def decide(self, messages, tools):
            if any(message["role"] == "tool" for message in messages):
                return {"content": "ready", "tool_call": None}
            return {
                "content": "overview",
                "tool_call": {"name": "inspect_overview", "arguments": {"sample_count": 1}},
            }

        def finalize(self, messages, output_schema):
            return {
                "sample_id": "s1",
                "evidence": [{
                    "evidence_id": "E1",
                    "acquisition": "inspect_overview",
                    "region_xyxy_1000": [0, 0, 1000, 1000],
                }],
                "answer": "A",
            }

    result = run_medical_agent(
        _sample(tmp_path), OverviewBackend(), tmp_path, tmp_path / "artifacts",
        allowed_tools=("inspect_overview",), max_steps=2,
    )
    assert result.prediction["agent_output"]["evidence"][0]["region_xyxy_1000"] is None
    assert result.prediction["policy_normalizations"] == [{
        "rule": "non_region_acquisition_requires_null_region",
        "evidence_index": 0,
        "before": [0, 0, 1000, 1000],
        "after": None,
    }]
    final_message = result.trajectory["messages"][-1]
    assert final_message["policy_normalizations"] == result.prediction["policy_normalizations"]


def test_inference_row_rejects_reference_fields(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    sample["answer"] = "A"
    with pytest.raises(ValueError, match="forbidden reference fields"):
        run_medical_agent(sample, FixtureBackend(), tmp_path, tmp_path / "artifacts")


def test_scorer_keeps_structure_answer_evidence_and_causality_separate() -> None:
    inference = [{
        "sample_id": "s1", "question_type": "mcq", "question": "?",
        "options": {"A": "yes", "B": "no"},
        "media": [{"media_id": "m1", "kind": "image", "path": "unused.png"}],
    }]
    references = [{"sample_id": "s1", "answer": "A", "evidence": [{"media_id": "m1", "region_xyxy_1000": [100, 100, 500, 500]}]}]
    traces = [{
        "trace_id": "s1:T1", "sample_id": "s1", "status": "completed",
        "tool_name": "region_inspect",
        "request": {"media_id": "m1", "region_xyxy_1000": [100, 100, 500, 500]},
        "selected_frames": [{"media_id": "m1"}],
    }]
    output = {
        "sample_id": "s1", "hypotheses": [],
        "evidence": [{
            "evidence_id": "E1", "media_id": "m1", "view_or_time": "current",
            "region_xyxy_1000": [100, 100, 500, 500], "observation": "visible fixture",
            "acquisition": "region_inspect", "confidence": 0.9, "supports": [], "contradicts": [],
        }],
        "answer": "A", "answer_evidence_ids": ["E1"], "confidence": 0.9,
        "insufficient_evidence": False, "tool_trace_ids": ["s1:T1"],
    }
    predictions = [{"sample_id": "s1", "parsed_answer": "A", "agent_output": output, "tool_trace_ids": ["s1:T1"]}]
    metrics = score_agent_rows(inference, references, predictions, traces)
    assert metrics["e0_structure"]["schema_valid_rate"] == 1.0
    assert metrics["e1_answer"]["accuracy"] == 1.0
    assert metrics["e2_evidence"]["mean_best_region_iou"] == 1.0
    assert metrics["e3_causal"]["status"] == "DEFER"


def test_malformed_prediction_is_scored_invalid_instead_of_crashing() -> None:
    inference = [{
        "sample_id": "s1", "question_type": "mcq", "question": "?",
        "options": {"A": "yes", "B": "no"},
        "media": [{"media_id": "m1", "kind": "image", "path": "unused.png"}],
    }]
    metrics = score_agent_rows(
        inference,
        [{"sample_id": "s1", "answer": "A"}],
        [{"sample_id": "s1", "parsed_answer": None, "agent_output": None}],
        [],
    )
    assert metrics["e0_structure"]["schema_valid_rate"] == 0.0
    assert metrics["e1_answer"]["accuracy"] == 0.0
    assert metrics["e3_causal"]["status"] == "DEFER"


def test_complete_synthetic_fixture_verifies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    report = run_fixture(run_dir)
    assert report["overall"] == "PASS"
    assert all(item["status"] == "PASS" for item in report["checks"])
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["e0_structure"] == {
        "citation_valid_rate": 1.0,
        "schema_valid_rate": 1.0,
        "tool_trace_bound_rate": 1.0,
    }
    traces = [json.loads(line) for line in (run_dir / "tool_traces.jsonl").read_text().splitlines()]
    assert len(traces) == 3
    assert all(trace["status"] == "completed" for trace in traces)


def test_verifier_blocks_when_run_does_not_declare_quality_gates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert run_fixture(run_dir)["overall"] == "PASS"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("quality_gates")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = verify_agent_run(run_dir)
    checks = {item["name"]: item["status"] for item in report["checks"]}
    assert checks["declared_quality_gates"] == "BLOCK"
    assert report["overall"] == "BLOCK"


def test_quality_gate_rejects_the_original_false_pass_shape() -> None:
    metrics = {
        "e0_structure": {
            "schema_valid_rate": 0.0,
            "citation_valid_rate": 1.0,
            "tool_trace_bound_rate": 0.0,
        }
    }
    gates = {
        "metric_minimums": {
            "e0_structure.schema_valid_rate": 1.0,
            "e0_structure.citation_valid_rate": 1.0,
            "e0_structure.tool_trace_bound_rate": 1.0,
        },
        "max_failed_tool_calls": 0,
    }
    passed, failures = _evaluate_quality_gates(metrics, [{"status": "failed"}], gates)
    assert not passed
    assert {failure["gate"] for failure in failures} == {
        "e0_structure.schema_valid_rate",
        "e0_structure.tool_trace_bound_rate",
        "max_failed_tool_calls",
    }
