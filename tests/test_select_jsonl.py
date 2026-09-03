from edgemed_bench.select_jsonl import select_rows


def test_select_rows_uses_explicit_selection_order() -> None:
    rows = [{"sample_id": "a", "v": 1}, {"sample_id": "b", "v": 2}]
    assert select_rows(rows, [{"sample_id": "b"}, {"sample_id": "a"}]) == [
        {"sample_id": "b", "v": 2},
        {"sample_id": "a", "v": 1},
    ]
