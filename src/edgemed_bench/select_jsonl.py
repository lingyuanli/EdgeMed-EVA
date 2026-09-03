"""Materialize an ordered JSONL subset from an explicit sample-id source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import append_jsonl, read_jsonl, sha256_file


def select_rows(rows: list[dict], selection: list[dict]) -> list[dict]:
    by_id = {str(row["sample_id"]): row for row in rows}
    selected_ids = [str(row["sample_id"]) for row in selection]
    if len(by_id) != len(rows) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Input and selection require unique sample ids")
    missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    if missing:
        raise KeyError(f"Selection contains unknown sample ids: {missing[:10]}")
    return [by_id[sample_id] for sample_id in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("0600", "0644"), default="0644")
    args = parser.parse_args()
    selected = select_rows(read_jsonl(args.input), read_jsonl(args.selection))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            append_jsonl(handle, row)
    args.output.chmod(int(args.mode, 8))
    print(
        json.dumps(
            {
                "input_sha256": sha256_file(args.input),
                "selection_sha256": sha256_file(args.selection),
                "output_sha256": sha256_file(args.output),
                "written": len(selected),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
