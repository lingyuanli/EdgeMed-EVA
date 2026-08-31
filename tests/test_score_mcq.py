import json
import sys
from pathlib import Path

from edgemed_bench.score_mcq import metric, wilson
from edgemed_bench.score_mcq import main as score_main


def test_metric_values() -> None:
    result = metric(3, 4)
    assert result["accuracy"] == 0.75
    assert result["accuracy_percent"] == 75.0
    assert result["correct"] == 3
    assert result["total"] == 4


def test_wilson_is_bounded() -> None:
    lower, upper = wilson(3, 4)
    assert 0 <= lower <= 75 <= upper <= 100


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_cli_reports_accuracy_tasks_and_invalid_parse(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    task = "Small-object Detection"
    write_jsonl(
        manifest,
        [
            {"sample_id": "mcq-1", "task": task},
            {"sample_id": "mcq-2", "task": task},
        ],
    )
    write_jsonl(
        references,
        [
            {"sample_id": "mcq-1", "answer": "A"},
            {"sample_id": "mcq-2", "answer": "B"},
        ],
    )
    write_jsonl(
        predictions,
        [
            {"sample_id": "mcq-1", "parsed_answer": "A", "parse_status": "exact_letter"},
            {"sample_id": "mcq-2", "parsed_answer": None, "parse_status": "invalid"},
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score_mcq",
            "--manifest",
            str(manifest),
            "--references",
            str(references),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )
    score_main()
    result = json.loads(output.read_text())
    assert result["complete"] is True
    assert result["overall"]["accuracy"] == 0.5
    assert result["by_task"]["SOD"]["accuracy"] == 0.5
    assert result["invalid_parse"]["rate"] == 0.5
