"""Paired B0-versus-candidate comparison of option-rotation consistency."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .compare_mcq import mcnemar_exact_p_value, paired_bootstrap_ci
from .io import read_jsonl, sha256_file, write_json


def _unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {label} sample ids")
    return result


def compare_invariance(
    rotated_manifest: list[dict[str, Any]],
    original_a: list[dict[str, Any]],
    rotated_a: list[dict[str, Any]],
    original_b: list[dict[str, Any]],
    rotated_b: list[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 20260902,
) -> dict[str, Any]:
    rotations = _unique(rotated_manifest, "manifest")
    inputs = {
        "original A": _unique(original_a, "original A"),
        "rotated A": _unique(rotated_a, "rotated A"),
        "original B": _unique(original_b, "original B"),
        "rotated B": _unique(rotated_b, "rotated B"),
    }
    if any(set(rows) != set(rotations) for rows in inputs.values()):
        raise ValueError("All predictions must exactly match the rotated manifest")

    a_only = b_only = both = neither = 0
    differences: list[int] = []
    for sample_id in sorted(rotations):
        mapping = rotations[sample_id].get("option_rotation", {}).get("old_to_new", {})
        expected_a = mapping.get(inputs["original A"][sample_id].get("parsed_answer"))
        expected_b = mapping.get(inputs["original B"][sample_id].get("parsed_answer"))
        consistent_a = (
            expected_a is not None
            and inputs["rotated A"][sample_id].get("parsed_answer") == expected_a
        )
        consistent_b = (
            expected_b is not None
            and inputs["rotated B"][sample_id].get("parsed_answer") == expected_b
        )
        differences.append(int(consistent_b) - int(consistent_a))
        if consistent_a and consistent_b:
            both += 1
        elif consistent_a:
            a_only += 1
        elif consistent_b:
            b_only += 1
        else:
            neither += 1
    total = len(rotations)
    return {
        "schema_version": "edgemed-paired-choice-invariance-comparison/v1",
        "complete": True,
        "total": total,
        "direction": "B_minus_A_higher_is_better",
        "consistency_a_percent": 100 * (both + a_only) / total,
        "consistency_b_percent": 100 * (both + b_only) / total,
        "delta_consistency_points": 100 * sum(differences) / total,
        "paired_bootstrap_95_percent": paired_bootstrap_ci(
            differences, repetitions=repetitions, seed=seed
        ),
        "mcnemar_exact_two_sided_p": mcnemar_exact_p_value(a_only, b_only),
        "contingency": {
            "both_consistent": both,
            "a_only_consistent": a_only,
            "b_only_consistent": b_only,
            "neither_consistent": neither,
        },
        "bootstrap": {"repetitions": repetitions, "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotated-manifest", type=Path, required=True)
    parser.add_argument("--original-a", type=Path, required=True)
    parser.add_argument("--rotated-a", type=Path, required=True)
    parser.add_argument("--original-b", type=Path, required=True)
    parser.add_argument("--rotated-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    paths = [args.rotated_manifest, args.original_a, args.rotated_a, args.original_b, args.rotated_b]
    result = compare_invariance(
        *(read_jsonl(path) for path in paths),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    labels = ("rotated_manifest", "original_a", "rotated_a", "original_b", "rotated_b")
    result["source_hashes"] = {
        f"{label}_sha256": sha256_file(path) for label, path in zip(labels, paths)
    }
    write_json(args.output, result)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
