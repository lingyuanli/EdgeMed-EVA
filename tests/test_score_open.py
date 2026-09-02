import json
import sys
from pathlib import Path

from edgemed_bench.compare_open import compare_open
from edgemed_bench.score_open import main as score_main
from edgemed_bench.score_open import normalize_answer, token_f1


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_open_normalization_and_token_f1_are_explicit() -> None:
    assert normalize_answer("  Left-Lung! ") == "left lung"
    assert token_f1("left lower lung", "left lung") == 0.8


def test_open_score_reports_proxy_and_slices(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.jsonl"
    references = tmp_path / "references.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "metrics.json"
    _write_jsonl(
        manifest,
        [
            {"sample_id": "1", "evaluation_metadata": {"answer_type": "CLOSED"}},
            {"sample_id": "2", "evaluation_metadata": {"answer_type": "OPEN"}},
        ],
    )
    _write_jsonl(references, [{"sample_id": "1", "answer": "Yes"}, {"sample_id": "2", "answer": "left lung"}])
    _write_jsonl(
        predictions,
        [
            {"sample_id": "1", "parsed_answer": "yes.", "parse_status": "valid"},
            {"sample_id": "2", "parsed_answer": "lung", "parse_status": "valid"},
        ],
    )
    monkeypatch.setattr(sys, "argv", ["score_open", "--manifest", str(manifest), "--references", str(references), "--predictions", str(predictions), "--output", str(output)])
    score_main()
    result = json.loads(output.read_text())
    assert result["metric_status"] == "external_retention_proxy_not_medcmr_official"
    assert result["overall"]["normalized_exact_percent"] == 50.0
    assert result["overall"]["mean_token_f1_percent"] == 100 * (1 + 2 / 3) / 2
    assert result["by_slice"]["answer_type"]["CLOSED"]["normalized_exact"] == 1.0


def test_paired_open_comparison_requires_same_samples() -> None:
    references = [{"sample_id": "1", "answer": "yes"}, {"sample_id": "2", "answer": "left lung"}]
    predictions_a = [{"sample_id": "1", "parsed_answer": "no"}, {"sample_id": "2", "parsed_answer": "lung"}]
    predictions_b = [{"sample_id": "1", "parsed_answer": "yes"}, {"sample_id": "2", "parsed_answer": "left lung"}]
    result = compare_open(references, predictions_a, predictions_b, repetitions=100, seed=7)
    assert result["delta_normalized_exact_points"] == 100.0
    assert result["delta_mean_token_f1_points"] > 0
