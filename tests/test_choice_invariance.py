from edgemed_bench.rotate_mcq import rotate_rows
from edgemed_bench.score_choice_invariance import score_invariance


def test_rotation_preserves_content_and_remaps_reference() -> None:
    manifest = [{"sample_id": "1", "kind": "mcq", "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}}]
    references = [{"sample_id": "1", "answer": "B"}]
    rotated, rotated_references = rotate_rows(manifest, references, shift=1)
    assert rotated[0]["options"] == {"A": "delta", "B": "alpha", "C": "beta", "D": "gamma"}
    assert rotated[0]["option_rotation"]["old_to_new"] == {"A": "B", "B": "C", "C": "D", "D": "A"}
    assert rotated_references == [{"sample_id": "1", "answer": "C"}]
    assert "answer" not in rotated[0]


def test_invariance_scores_content_equivalent_prediction() -> None:
    manifest = [{"sample_id": "1", "option_rotation": {"old_to_new": {"A": "B", "B": "C", "C": "D", "D": "A"}}}]
    original = [{"sample_id": "1", "parsed_answer": "B"}]
    rotated = [{"sample_id": "1", "parsed_answer": "C"}]
    result = score_invariance(manifest, original, rotated)
    assert result["overall"]["accuracy"] == 1.0
    assert result["overall"]["both_parseable"] == 1


def test_zero_shift_is_rejected() -> None:
    try:
        rotate_rows(
            [{"sample_id": "1", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}],
            [{"sample_id": "1", "answer": "A"}],
            shift=4,
        )
    except ValueError as error:
        assert "must change" in str(error)
    else:
        raise AssertionError("identity rotation was accepted")
