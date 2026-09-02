"""Create answer-preserving cyclic option rotations for an MCQ manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, reject_reference_fields, sha256_file, write_json


def rotate_rows(
    manifest: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    shift: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    answers = {str(row["sample_id"]): str(row["answer"]).upper() for row in references}
    if len(answers) != len(references):
        raise ValueError("Duplicate reference sample ids")
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    if len(manifest_ids) != len(manifest):
        raise ValueError("Duplicate manifest sample ids")
    if manifest_ids != set(answers):
        raise ValueError("Manifest/reference sample ids differ")

    rotated_manifest: list[dict[str, Any]] = []
    rotated_references: list[dict[str, Any]] = []
    for row in manifest:
        sample_id = str(row["sample_id"])
        options = row.get("options")
        if not isinstance(options, dict):
            raise ValueError(f"Missing options: {sample_id}")
        letters = sorted(str(letter).upper() for letter in options)
        if "".join(letters) not in {"ABCD", "ABCDE"}:
            raise ValueError(f"Options must be contiguous A-D or A-E: {sample_id}")
        effective_shift = shift % len(letters)
        if effective_shift == 0:
            raise ValueError("Rotation shift must change option positions")
        old_to_new = {
            old: letters[(index + effective_shift) % len(letters)]
            for index, old in enumerate(letters)
        }
        rotated_options = {old_to_new[old]: options[old] for old in letters}
        rotated = dict(row)
        rotated["options"] = {letter: rotated_options[letter] for letter in letters}
        rotated["option_rotation"] = {
            "schema_version": "edgemed-option-rotation/v1",
            "shift": effective_shift,
            "old_to_new": old_to_new,
        }
        rotated_manifest.append(rotated)
        answer = answers[sample_id]
        if answer not in old_to_new:
            raise ValueError(f"Reference outside options: {sample_id}")
        rotated_references.append({"sample_id": sample_id, "answer": old_to_new[answer]})

    reject_reference_fields(rotated_manifest)
    return rotated_manifest, rotated_references


def _write_jsonl(path: Path, rows: list[dict[str, Any]], mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )
    if mode is not None:
        path.chmod(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-references", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shift", type=int, default=1)
    args = parser.parse_args()
    rotated_manifest, rotated_references = rotate_rows(
        read_jsonl(args.manifest), read_jsonl(args.references), shift=args.shift
    )
    _write_jsonl(args.output_manifest, rotated_manifest)
    _write_jsonl(args.output_references, rotated_references, mode=0o600)
    report = {
        "schema_version": "edgemed-option-rotation-report/v1",
        "count": len(rotated_manifest),
        "shift": args.shift,
        "answer_preserving": True,
        "leakage_boundary": {
            "inference_has_reference_fields": False,
            "references_mode": "0600",
        },
        "source_hashes": {
            "input_manifest_sha256": sha256_file(args.manifest),
            "input_references_sha256": sha256_file(args.references),
            "output_manifest_sha256": sha256_file(args.output_manifest),
            "output_references_sha256": sha256_file(args.output_references),
        },
    }
    write_json(args.report, report)
    print(args.report.read_text())


if __name__ == "__main__":
    main()
