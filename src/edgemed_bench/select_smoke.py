"""Create a deterministic task-stratified Med-CMR smoke selection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, reject_reference_fields, sha256_file, write_json

EXPECTED_TASKS = (
    "Small-object Detection",
    "Fine-detail Discrimination",
    "Spatial Understanding",
    "Temporal Prediction",
    "Causal Reasoning",
    "Long-tail Generalization",
    "Multi-source Integration",
)


def select_sample_ids(rows: list[dict[str, Any]], per_task: int) -> list[str]:
    if per_task < 1:
        raise ValueError("per_task must be positive")
    selected: list[str] = []
    for task in EXPECTED_TASKS:
        task_ids = [str(row["sample_id"]) for row in rows if row.get("task") == task]
        if len(task_ids) < per_task:
            raise ValueError(f"Task {task!r} has only {len(task_ids)} rows")
        selected.extend(task_ids[:per_task])
    unknown = sorted({str(row.get("task")) for row in rows} - set(EXPECTED_TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=2)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    reject_reference_fields(rows)
    selected = select_sample_ids(rows, args.per_task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n")
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    write_json(
        metadata_path,
        {
            "schema_version": "medcmr-smoke-selection/v1",
            "source_manifest_sha256": sha256_file(args.manifest),
            "sample_id_file_sha256": sha256_file(args.output),
            "per_task": args.per_task,
            "selected_count": len(selected),
            "task_counts": dict(
                Counter(row["task"] for row in rows if row["sample_id"] in set(selected))
            ),
        },
    )
    print(metadata_path.read_text())


if __name__ == "__main__":
    main()
