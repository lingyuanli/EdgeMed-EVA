from edgemed_bench.randomize_training_options import (
    deterministic_shift,
    randomize_training_rows,
    selection_audit,
)


def _fixture() -> tuple[list[dict], list[dict]]:
    manifest = [
        {
            "sample_id": str(index),
            "kind": "mcq",
            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        }
        for index in range(12)
    ]
    references = [
        {"sample_id": str(index), "answer": "ABCD"[index % 4]}
        for index in range(12)
    ]
    return manifest, references


def test_deterministic_training_order_preserves_answer_content() -> None:
    manifest, references = _fixture()
    moved, moved_references, audit = randomize_training_rows(manifest, references, seed=7)
    original_answers = {row["sample_id"]: row["answer"] for row in references}
    moved_answers = {row["sample_id"]: row["answer"] for row in moved_references}
    for original, randomized in zip(manifest, moved, strict=True):
        sample_id = original["sample_id"]
        assert randomized["options"][moved_answers[sample_id]] == original["options"][original_answers[sample_id]]
        assert "answer" not in randomized
    assert sum(audit["shift_counts"].values()) == 12


def test_shift_is_stable_and_seed_bound() -> None:
    assert deterministic_shift("sample-1", 9, 4) == deterministic_shift("sample-1", 9, 4)
    observed = {deterministic_shift(f"sample-{index}", 9, 4) for index in range(100)}
    assert observed == {0, 1, 2, 3}


def test_selection_audit_matches_training_shuffle() -> None:
    manifest, references = _fixture()
    moved, moved_references, _ = randomize_training_rows(manifest, references, seed=7)
    audit = selection_audit(moved, moved_references, seed=11, count=8)
    assert audit["count"] == 8
    assert sum(audit["answer_position_counts"].values()) == 8
    assert sum(audit["shift_counts"].values()) == 8
    assert len(audit["sample_ids_sha256"]) == 64
