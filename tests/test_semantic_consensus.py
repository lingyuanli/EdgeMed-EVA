from edgemed_bench.rotate_mcq import rotate_rows
from edgemed_bench.semantic_consensus import semantic_consensus


def _views():
    original = [
        {
            "sample_id": "1",
            "kind": "mcq",
            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        }
    ]
    refs = [{"sample_id": "1", "answer": "A"}]
    rotated = [rotate_rows(original, refs, shift=shift)[0] for shift in (1, 2, 3)]
    return original, rotated


def test_consensus_maps_rotated_letters_to_semantic_identity() -> None:
    original, rotated = _views()
    predictions = [[{"sample_id": "1", "parsed_answer": letter}] for letter in "ABCD"]
    rows, report = semantic_consensus(original, predictions[0], rotated, predictions[1:])
    assert rows[0]["parsed_answer"] == "A"
    assert rows[0]["parse_status"] == "semantic_consensus"
    assert rows[0]["agent_trace"]["canonical_vote_counts"] == {"A": 4}
    assert report["tie_count"] == 0


def test_consensus_is_invariant_to_rotated_view_argument_order() -> None:
    original, rotated = _views()
    original_prediction = [{"sample_id": "1", "parsed_answer": "A"}]
    rotated_predictions = [[{"sample_id": "1", "parsed_answer": letter}] for letter in "BCD"]
    rows_a, _ = semantic_consensus(original, original_prediction, rotated, rotated_predictions)
    rows_b, _ = semantic_consensus(
        original,
        original_prediction,
        list(reversed(rotated)),
        list(reversed(rotated_predictions)),
    )
    assert rows_a[0]["parsed_answer"] == rows_b[0]["parsed_answer"] == "A"


def test_consensus_tie_break_does_not_use_option_letter() -> None:
    original, rotated = _views()
    original_prediction = [{"sample_id": "1", "parsed_answer": "A"}]
    rotated_predictions = [
        [{"sample_id": "1", "parsed_answer": "C"}],
        [{"sample_id": "1", "parsed_answer": "C"}],
        [{"sample_id": "1", "parsed_answer": "A"}],
    ]
    rows, report = semantic_consensus(original, original_prediction, rotated, rotated_predictions)
    assert rows[0]["parse_status"] == "semantic_consensus_tiebreak"
    assert rows[0]["agent_trace"]["canonical_vote_counts"] == {"A": 2, "B": 2}
    assert rows[0]["agent_trace"]["tie_breaker"] == "sha256-normalized-option-content"
    assert report["tie_count"] == 1


def test_consensus_reports_all_invalid_without_guessing() -> None:
    original, rotated = _views()
    invalid = [{"sample_id": "1", "parsed_answer": None}]
    rows, report = semantic_consensus(original, invalid, rotated, [invalid, invalid, invalid])
    assert rows[0]["parsed_answer"] is None
    assert rows[0]["parse_status"] == "ensemble_all_invalid"
    assert report["all_invalid"] == 1


def test_consensus_does_not_break_duplicate_content_tie_by_letter() -> None:
    original, rotated = _views()
    original[0]["options"]["B"] = "alpha"
    rotated = [
        rotate_rows(original, [{"sample_id": "1", "answer": "A"}], shift=shift)[0]
        for shift in (1, 2, 3)
    ]
    original_prediction = [{"sample_id": "1", "parsed_answer": "A"}]
    rotated_predictions = [
        [{"sample_id": "1", "parsed_answer": "C"}],
        [{"sample_id": "1", "parsed_answer": "C"}],
        [{"sample_id": "1", "parsed_answer": "A"}],
    ]
    rows, report = semantic_consensus(original, original_prediction, rotated, rotated_predictions)
    assert rows[0]["parsed_answer"] is None
    assert rows[0]["parse_status"] == "ensemble_ambiguous_duplicate_content"
    assert report["ambiguous_duplicate_content"] == 1
