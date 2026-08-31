import json
from pathlib import Path

import pytest

from edgemed_bench.run import completed_ids


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_completed_ids_accepts_exact_contract(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_rows(
        predictions,
        [{"sample_id": "mcq-1", "status": "completed", "contract_sha256": "abc"}],
    )
    assert completed_ids(predictions, {"mcq-1"}, "abc") == {"mcq-1"}


@pytest.mark.parametrize(
    "rows",
    [
        [{"sample_id": "mcq-2", "status": "completed", "contract_sha256": "abc"}],
        [{"sample_id": "mcq-1", "status": "failed", "contract_sha256": "abc"}],
        [{"sample_id": "mcq-1", "status": "completed", "contract_sha256": "wrong"}],
        [
            {"sample_id": "mcq-1", "status": "completed", "contract_sha256": "abc"},
            {"sample_id": "mcq-1", "status": "completed", "contract_sha256": "abc"},
        ],
    ],
)
def test_completed_ids_rejects_non_exact_resume(
    tmp_path: Path, rows: list[dict[str, str]]
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_rows(predictions, rows)
    with pytest.raises(RuntimeError):
        completed_ids(predictions, {"mcq-1"}, "abc")
