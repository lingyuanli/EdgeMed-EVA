"""Aggregate cyclic MCQ views by semantic option identity without references."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, reject_reference_fields, sha256_file, write_json


def _unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample ids in {label}")
    return result


def _content_tiebreak(option: Any) -> str:
    normalized = " ".join(str(option).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def semantic_consensus(
    original_manifest: list[dict[str, Any]],
    original_predictions: list[dict[str, Any]],
    rotated_manifests: list[list[dict[str, Any]]],
    rotated_predictions: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rotated_manifests) != 3 or len(rotated_predictions) != 3:
        raise ValueError("A1 requires exactly three rotated views plus the original")
    original = _unique(original_manifest, "original manifest")
    reject_reference_fields(original.values())
    prediction_views = [_unique(original_predictions, "original predictions")]
    inverse_views: list[dict[str, dict[str, str]] | None] = [None]
    shifts: set[int] = set()
    for index, (manifest_rows, prediction_rows) in enumerate(
        zip(rotated_manifests, rotated_predictions, strict=True), 1
    ):
        manifest = _unique(manifest_rows, f"rotated manifest {index}")
        reject_reference_fields(manifest.values())
        if set(manifest) != set(original):
            raise ValueError("Rotated manifest sample ids differ from original")
        inverse: dict[str, dict[str, str]] = {}
        current_shifts: set[int] = set()
        for sample_id, row in manifest.items():
            rotation = row.get("option_rotation")
            if not isinstance(rotation, dict):
                raise ValueError(f"Missing option rotation: {sample_id}")
            shift = int(rotation["shift"])
            current_shifts.add(shift)
            old_to_new = {str(old): str(new) for old, new in rotation["old_to_new"].items()}
            inverse[sample_id] = {new: old for old, new in old_to_new.items()}
            original_options = original[sample_id].get("options")
            rotated_options = row.get("options")
            if not isinstance(original_options, dict) or not isinstance(rotated_options, dict):
                raise ValueError(f"Missing options: {sample_id}")
            for old, new in old_to_new.items():
                if original_options.get(old) != rotated_options.get(new):
                    raise ValueError(f"Rotation changed option content: {sample_id}")
        if len(current_shifts) != 1:
            raise ValueError(f"Rotated view {index} mixes shifts: {sorted(current_shifts)}")
        shifts.update(current_shifts)
        prediction_views.append(_unique(prediction_rows, f"rotated predictions {index}"))
        inverse_views.append(inverse)
    if shifts != {1, 2, 3}:
        raise ValueError(f"Expected cyclic shifts 1,2,3; observed {sorted(shifts)}")
    if any(set(view) != set(original) for view in prediction_views):
        raise ValueError("Prediction sample ids must exactly match the original manifest")

    outputs: list[dict[str, Any]] = []
    ties = 0
    all_invalid = 0
    ambiguous_duplicate_content = 0
    vote_histogram: Counter[str] = Counter()
    for row in original_manifest:
        sample_id = str(row["sample_id"])
        options = row["options"]
        votes: list[str] = []
        trace: list[dict[str, Any]] = []
        for view_index, (predictions, inverse) in enumerate(zip(prediction_views, inverse_views, strict=True)):
            visible_answer = predictions[sample_id].get("parsed_answer")
            canonical_answer = visible_answer
            if inverse is not None and visible_answer is not None:
                canonical_answer = inverse[sample_id].get(str(visible_answer))
            if canonical_answer in options:
                votes.append(str(canonical_answer))
            trace.append(
                {
                    "view": view_index,
                    "visible_answer": visible_answer,
                    "canonical_answer": canonical_answer if canonical_answer in options else None,
                }
            )
        counts = Counter(votes)
        if not counts:
            winner = None
            parse_status = "ensemble_all_invalid"
            all_invalid += 1
            max_votes = 0
        else:
            max_votes = max(counts.values())
            candidates = [letter for letter, count in counts.items() if count == max_votes]
            tied = len(candidates) > 1
            ties += int(tied)
            ranked = {
                letter: (_content_tiebreak(options[letter]), " ".join(str(options[letter]).casefold().split()))
                for letter in candidates
            }
            best_key = min(ranked.values())
            winners = [letter for letter, key in ranked.items() if key == best_key]
            if len(winners) != 1:
                winner = None
                parse_status = "ensemble_ambiguous_duplicate_content"
                ambiguous_duplicate_content += 1
            else:
                winner = winners[0]
                parse_status = "semantic_consensus_tiebreak" if tied else "semantic_consensus"
        vote_histogram[str(max_votes)] += 1
        outputs.append(
            {
                "sample_id": sample_id,
                "parsed_answer": winner,
                "parse_status": parse_status,
                "agent_trace": {
                    "schema_version": "edgemed-semantic-consensus-trace/v1",
                    "views": trace,
                    "canonical_vote_counts": dict(sorted(counts.items())),
                    "parseable_views": len(votes),
                    "tie_breaker": "sha256-normalized-option-content",
                },
            }
        )
    report = {
        "schema_version": "edgemed-semantic-consensus-report/v1",
        "status": "completed",
        "count": len(outputs),
        "view_count": 4,
        "all_invalid": all_invalid,
        "ambiguous_duplicate_content": ambiguous_duplicate_content,
        "tie_count": ties,
        "max_vote_histogram": dict(sorted(vote_histogram.items())),
        "reference_fields_used": False,
        "tie_breaker": "sha256-normalized-option-content",
    }
    return outputs, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--original-predictions", type=Path, required=True)
    parser.add_argument("--rotated-manifest", type=Path, action="append", required=True)
    parser.add_argument("--rotated-predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("A1 output or report already exists")
    outputs, report = semantic_consensus(
        read_jsonl(args.original_manifest),
        read_jsonl(args.original_predictions),
        [read_jsonl(path) for path in args.rotated_manifest],
        [read_jsonl(path) for path in args.rotated_predictions],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in outputs:
            import json

            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report["source_hashes"] = {
        "original_manifest_sha256": sha256_file(args.original_manifest),
        "original_predictions_sha256": sha256_file(args.original_predictions),
        "rotated_manifest_sha256": [sha256_file(path) for path in args.rotated_manifest],
        "rotated_predictions_sha256": [sha256_file(path) for path in args.rotated_predictions],
        "output_sha256": sha256_file(args.output),
    }
    write_json(args.report, report)
    print(args.report.read_text())


if __name__ == "__main__":
    main()
