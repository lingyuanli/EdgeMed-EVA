import pytest

from edgemed_bench.select_smoke import EXPECTED_TASKS, select_sample_ids


def rows_for_each_task(count: int = 2) -> list[dict[str, str]]:
    return [
        {"sample_id": f"mcq-{task_index}-{row_index}", "task": task}
        for task_index, task in enumerate(EXPECTED_TASKS)
        for row_index in range(count)
    ]


def test_select_sample_ids_covers_every_task() -> None:
    selected = select_sample_ids(rows_for_each_task(), per_task=2)
    assert len(selected) == 14
    assert len(set(selected)) == 14


def test_select_sample_ids_rejects_missing_task() -> None:
    with pytest.raises(ValueError):
        select_sample_ids(rows_for_each_task()[:-1], per_task=2)
