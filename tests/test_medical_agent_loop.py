from pathlib import Path

import pytest
from PIL import Image

from edgemed_bench.medical_agent import run_medical_agent
from edgemed_bench.medical_agent_fixture import FixtureBackend, run_fixture
from edgemed_bench.medical_agent_tools import MedicalToolExecutor
from edgemed_bench.score_agent import score_agent_rows


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
        run_medical_agent(_sample(tmp_path), PrematureBackend(), tmp_path, tmp_path / "artifacts")


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
    }]
    output = {
        "sample_id": "s1", "hypotheses": [],
        "evidence": [{"evidence_id": "E1", "media_id": "m1", "region_xyxy_1000": [100, 100, 500, 500], "observation": "visible fixture"}],
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
    report = run_fixture(tmp_path / "run")
    assert report["overall"] == "PASS"
    assert all(item["status"] == "PASS" for item in report["checks"])
