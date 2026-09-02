"""Create a deterministic, answer-preserving option-order training surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, reject_reference_fields, sha256_file, write_json


SCHEMA_VERSION = "edgemed-training-option-order/v1"


def deterministic_shift(sample_id: str, seed: int, option_count: int) -> int:
    if option_count <= 0:
        raise ValueError("option_count must be positive")
    payload = f"{SCHEMA_VERSION}\0{seed}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % option_count


def randomize_training_rows(
    manifest: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    answers = {str(row["sample_id"]): str(row["answer"]).upper() for row in references}
    if len(answers) != len(references):
        raise ValueError("Duplicate reference sample ids")
    manifest_ids = {str(row["sample_id"]) for row in manifest}
    if len(manifest_ids) != len(manifest):
        raise ValueError("Duplicate manifest sample ids")
    if manifest_ids != set(answers):
        raise ValueError("Manifest/reference sample ids differ")

    randomized_manifest: list[dict[str, Any]] = []
    randomized_references: list[dict[str, Any]] = []
    shift_counts: Counter[str] = Counter()
    answer_before: Counter[str] = Counter()
    answer_after: Counter[str] = Counter()
    for row in manifest:
        sample_id = str(row["sample_id"])
        options = row.get("options")
        if not isinstance(options, dict):
            raise ValueError(f"Missing options: {sample_id}")
        letters = sorted(str(letter).upper() for letter in options)
        if "".join(letters) not in {"ABCD", "ABCDE"}:
            raise ValueError(f"Options must be contiguous A-D or A-E: {sample_id}")
        answer = answers[sample_id]
        if answer not in letters:
            raise ValueError(f"Reference outside options: {sample_id}")
        shift = deterministic_shift(sample_id, seed, len(letters))
        old_to_new = {
            old: letters[(index + shift) % len(letters)]
            for index, old in enumerate(letters)
        }
        moved_options = {old_to_new[old]: options[old] for old in letters}
        randomized = dict(row)
        randomized["options"] = {letter: moved_options[letter] for letter in letters}
        randomized["training_option_order"] = {
            "schema_version": SCHEMA_VERSION,
            "seed": seed,
            "shift": shift,
            "old_to_new": old_to_new,
        }
        randomized_manifest.append(randomized)
        new_answer = old_to_new[answer]
        randomized_references.append({"sample_id": sample_id, "answer": new_answer})
        shift_counts[str(shift)] += 1
        answer_before[answer] += 1
        answer_after[new_answer] += 1

    reject_reference_fields(randomized_manifest)
    audit = {
        "shift_counts": dict(sorted(shift_counts.items())),
        "answer_position_counts_before": dict(sorted(answer_before.items())),
        "answer_position_counts_after": dict(sorted(answer_after.items())),
        "changed_rows": sum(int(value["training_option_order"]["shift"] != 0) for value in randomized_manifest),
    }
    return randomized_manifest, randomized_references, audit


def selection_audit(
    manifest: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    seed: int,
    count: int,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("selection count must be positive")
    answers = {str(row["sample_id"]): str(row["answer"]).upper() for row in references}
    order = list(range(len(manifest)))
    random.Random(seed).shuffle(order)
    if count > len(order):
        order = [order[index % len(order)] for index in range(count)]
    else:
        order = order[:count]
    chosen = [manifest[index] for index in order]
    positions = Counter(answers[str(row["sample_id"])] for row in chosen)
    shifts = Counter(str(row["training_option_order"]["shift"]) for row in chosen)
    return {
        "seed": seed,
        "count": count,
        "answer_position_counts": dict(sorted(positions.items())),
        "shift_counts": dict(sorted(shifts.items())),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(str(row["sample_id"]) for row in chosen).encode()
        ).hexdigest(),
    }


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
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--selected-examples", type=int, required=True)
    args = parser.parse_args()
    for output in (args.output_manifest, args.output_references, args.report):
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
    randomized_manifest, randomized_references, audit = randomize_training_rows(
        read_jsonl(args.manifest), read_jsonl(args.references), seed=args.order_seed
    )
    selected = selection_audit(
        randomized_manifest,
        randomized_references,
        seed=args.selection_seed,
        count=args.selected_examples,
    )
    _write_jsonl(args.output_manifest, randomized_manifest)
    _write_jsonl(args.output_references, randomized_references, mode=0o600)
    report = {
        "schema_version": "edgemed-training-option-order-report/v1",
        "count": len(randomized_manifest),
        "order_seed": args.order_seed,
        "answer_preserving": True,
        "augmentation": "deterministic-per-sample-cyclic-shift",
        "audit": audit,
        "selected_training_examples": selected,
        "leakage_boundary": {
            "inference_has_reference_fields": False,
            "references_mode": "0600",
            "evaluation_surfaces_changed": False,
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
