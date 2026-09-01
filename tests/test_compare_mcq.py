from edgemed_bench.compare_mcq import compare, mcnemar_exact_p_value, paired_bootstrap_ci


def test_exact_mcnemar_handles_no_discordant_pairs() -> None:
    assert mcnemar_exact_p_value(0, 0) == 1.0
    assert mcnemar_exact_p_value(0, 5) == 0.0625


def test_paired_bootstrap_is_deterministic_and_bounded() -> None:
    first = paired_bootstrap_ci([1, 0, -1, 1], repetitions=1000, seed=7)
    second = paired_bootstrap_ci([1, 0, -1, 1], repetitions=1000, seed=7)
    assert first == second
    assert -100 <= first[0] <= first[1] <= 100


def test_compare_reports_b_minus_a_and_contingency() -> None:
    references = [
        {"sample_id": "one", "answer": "A"},
        {"sample_id": "two", "answer": "B"},
        {"sample_id": "three", "answer": "C"},
        {"sample_id": "four", "answer": "D"},
    ]
    predictions_a = [
        {"sample_id": "one", "parsed_answer": "A"},
        {"sample_id": "two", "parsed_answer": "B"},
        {"sample_id": "three", "parsed_answer": "A"},
        {"sample_id": "four", "parsed_answer": "A"},
    ]
    predictions_b = [
        {"sample_id": "one", "parsed_answer": "A"},
        {"sample_id": "two", "parsed_answer": "A"},
        {"sample_id": "three", "parsed_answer": "C"},
        {"sample_id": "four", "parsed_answer": "A"},
    ]
    result = compare(references, predictions_a, predictions_b, repetitions=100, seed=3)
    assert result["delta_accuracy_points"] == 0.0
    assert result["contingency"] == {
        "both_correct": 1,
        "a_only_correct": 1,
        "b_only_correct": 1,
        "both_wrong": 1,
    }
    assert result["changed_samples"] == {"b_gains": ["three"], "b_losses": ["two"]}
