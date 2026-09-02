import json
from pathlib import Path

import pytest
from PIL import Image

from edgemed_bench.finalize_agent_eval import finalize_agent_evaluation
from edgemed_bench.io import read_jsonl
from edgemed_bench.qwen_agent_backend import parse_json_object
from edgemed_bench.run_medical_agent import normalize_agent_sample, run_agent_batch


class FakeBackend:
    receipt = {"backend": "fake-agent-backend/v1"}

    def __init__(self) -> None:
        self.finalized: list[str] = []

    def decide(self, messages, tools):
        if not any(message["role"] == "tool" for message in messages):
            return {
                "content": "Need an overview.",
                "tool_call": {"name": "inspect_overview", "arguments": {"sample_count": 1}},
            }
        return {"content": "Evidence is sufficient.", "tool_call": None}

    def finalize(self, messages, output_schema):
        sample = messages[1]["content"]
        sample_id = sample["sample_id"]
        self.finalized.append(sample_id)
        trace_ids = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
        return {
            "sample_id": sample_id,
            "hypotheses": [{"id": "H1", "label": "fixture", "status": "supported"}],
            "evidence": [{
                "evidence_id": "E1",
                "media_id": sample["media"][0]["media_id"],
                "region_xyxy_1000": None,
                "observation": "A synthetic fixture image is visible.",
                "acquisition": "inspect_overview",
                "confidence": 1.0,
                "supports": ["H1"],
                "contradicts": [],
            }],
            "answer": "A",
            "answer_evidence_ids": ["E1"],
            "confidence": 1.0,
            "insufficient_evidence": False,
            "tool_trace_ids": trace_ids,
        }


def _rows(root: Path) -> list[dict]:
    rows = []
    for index in range(2):
        path = root / f"image-{index}.png"
        Image.new("RGB", (32, 32), (index * 50, 0, 0)).save(path)
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({
            "sample_id": f"s{index}",
            "kind": "mcq",
            "question": "Fixture?",
            "options": {"A": "yes", "B": "no"},
            "image_path": path.name,
            "image_sha256": digest,
        })
    return rows


def _contract() -> dict:
    return {
        "stage": "test",
        "allowed_tools": ["inspect_overview"],
        "max_steps": 2,
    }


def test_batch_exact_resume_and_reference_isolated_finalize(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    run_dir = tmp_path / "run"
    first = FakeBackend()
    with pytest.raises(InterruptedError):
        run_agent_batch(
            rows, first, data_root=tmp_path, run_dir=run_dir,
            contract=_contract(), allowed_tools=("inspect_overview",), max_steps=2,
            interrupt_after=1,
        )
    assert first.finalized == ["s0"]
    second = FakeBackend()
    manifest = run_agent_batch(
        rows, second, data_root=tmp_path, run_dir=run_dir,
        contract=_contract(), allowed_tools=("inspect_overview",), max_steps=2,
        resume=True,
    )
    assert second.finalized == ["s1"]
    assert manifest["status"] == "inference_completed"
    assert [row["sample_id"] for row in read_jsonl(run_dir / "predictions.jsonl")] == ["s0", "s1"]
    assert not (run_dir / "references.jsonl").exists()

    full_references = tmp_path / "references.jsonl"
    full_references.write_text(
        "".join(json.dumps({"sample_id": row["sample_id"], "answer": "A"}) + "\n" for row in rows)
    )
    report = finalize_agent_evaluation(run_dir, full_references)
    assert report["overall"] == "PASS"
    assert (run_dir / "references.jsonl").stat().st_mode & 0o777 == 0o600
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["e0_structure"]["schema_valid_rate"] == 1.0
    assert metrics["e1_answer"]["accuracy"] == 1.0


def test_batch_resume_rejects_contract_change(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    run_dir = tmp_path / "run"
    with pytest.raises(InterruptedError):
        run_agent_batch(
            rows, FakeBackend(), data_root=tmp_path, run_dir=run_dir,
            contract=_contract(), allowed_tools=("inspect_overview",), max_steps=2,
            interrupt_after=1,
        )
    with pytest.raises(RuntimeError, match="Resume contract differs"):
        run_agent_batch(
            rows, FakeBackend(), data_root=tmp_path, run_dir=run_dir,
            contract={**_contract(), "max_steps": 3},
            allowed_tools=("inspect_overview",), max_steps=3, resume=True,
        )


def test_normalize_agent_sample_rejects_reference_bearing_row(tmp_path: Path) -> None:
    row = _rows(tmp_path)[0]
    row["answer"] = "A"
    with pytest.raises(ValueError, match="forbidden reference fields"):
        normalize_agent_sample(row)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"tool_call":null}', {"tool_call": None}),
        ('```json\n{"answer":"A"}\n```', {"answer": "A"}),
    ],
)
def test_parse_json_object_accepts_exact_object(raw: str, expected: dict) -> None:
    assert parse_json_object(raw) == expected


@pytest.mark.parametrize("raw", ['prefix {"answer":"A"}', '{"answer":"A"} trailing', "not json"])
def test_parse_json_object_rejects_non_exact_output(raw: str) -> None:
    with pytest.raises(ValueError, match="exactly one JSON object"):
        parse_json_object(raw)
