"""Measure whether an MCQ prediction follows answer content through option rotation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json
from .score_mcq import metric


def score_invariance(
    rotated_manifest: list[dict[str, Any]],
    original_predictions: list[dict[str, Any]],
    rotated_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    rotations = {str(row["sample_id"]): row.get("option_rotation") for row in rotated_manifest}
    original = {str(row["sample_id"]): row for row in original_predictions}
    rotated = {str(row["sample_id"]): row for row in rotated_predictions}
    if len(rotations) != len(rotated_manifest) or len(original) != len(original_predictions) or len(rotated) != len(rotated_predictions):
        raise ValueError("Duplicate sample ids")
    if set(original) != set(rotations) or set(rotated) != set(rotations):
        raise ValueError("Prediction sample ids must exactly match the rotated manifest")

    consistent = 0
    both_parseable = 0
    transition_counts: Counter[str] = Counter()
    for sample_id in sorted(rotations):
        rotation = rotations[sample_id]
        if not isinstance(rotation, dict) or not isinstance(rotation.get("old_to_new"), dict):
            raise ValueError(f"Missing option rotation metadata: {sample_id}")
        answer_original = original[sample_id].get("parsed_answer")
        answer_rotated = rotated[sample_id].get("parsed_answer")
        if answer_original is not None and answer_rotated is not None:
            both_parseable += 1
        expected_rotated = rotation["old_to_new"].get(answer_original)
        is_consistent = expected_rotated is not None and answer_rotated == expected_rotated
        consistent += int(is_consistent)
        transition_counts[f"{answer_original}->{answer_rotated}"] += 1

    total = len(rotations)
    result = metric(consistent, total)
    result["name"] = "answer_content_consistency_under_rotation"
    result["both_parseable"] = both_parseable
    result["transition_counts"] = dict(sorted(transition_counts.items()))
    return {
        "schema_version": "edgemed-choice-invariance-metrics/v1",
        "complete": True,
        "overall": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotated-manifest", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--rotated-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_invariance(
        read_jsonl(args.rotated_manifest),
        read_jsonl(args.original_predictions),
        read_jsonl(args.rotated_predictions),
    )
    result["source_hashes"] = {
        "rotated_manifest_sha256": sha256_file(args.rotated_manifest),
        "original_predictions_sha256": sha256_file(args.original_predictions),
        "rotated_predictions_sha256": sha256_file(args.rotated_predictions),
    }
    write_json(args.output, result)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
