"""Paired comparison for two complete MCQ prediction files."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    differences: list[int], *, repetitions: int = 10_000, seed: int = 20260901
) -> list[float]:
    """Return a percentile CI for mean paired accuracy difference, in percent."""
    if not differences:
        raise ValueError("At least one paired observation is required")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    size = len(differences)
    draws = [
        100 * sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(repetitions)
    ]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def mcnemar_exact_p_value(a_only_correct: int, b_only_correct: int) -> float:
    """Two-sided exact McNemar p-value using the discordant-pair binomial test."""
    discordant = a_only_correct + b_only_correct
    if discordant == 0:
        return 1.0
    smaller = min(a_only_correct, b_only_correct)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def _unique_by_sample(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate {label} sample id: {sample_id}")
        result[sample_id] = row
    return result


def compare(
    references: list[dict[str, Any]],
    predictions_a: list[dict[str, Any]],
    predictions_b: list[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 20260901,
) -> dict[str, Any]:
    answers = {
        sample_id: str(row["answer"]).upper()
        for sample_id, row in _unique_by_sample(references, "reference").items()
    }
    by_a = _unique_by_sample(predictions_a, "prediction A")
    by_b = _unique_by_sample(predictions_b, "prediction B")
    expected = set(answers)
    if set(by_a) != expected or set(by_b) != expected:
        raise ValueError("Both prediction files must exactly match the reference sample ids")

    both_correct = a_only_correct = b_only_correct = both_wrong = 0
    differences: list[int] = []
    gains: list[str] = []
    losses: list[str] = []
    for sample_id in sorted(expected):
        a_correct = by_a[sample_id].get("parsed_answer") == answers[sample_id]
        b_correct = by_b[sample_id].get("parsed_answer") == answers[sample_id]
        differences.append(int(b_correct) - int(a_correct))
        if a_correct and b_correct:
            both_correct += 1
        elif a_correct:
            a_only_correct += 1
            losses.append(sample_id)
        elif b_correct:
            b_only_correct += 1
            gains.append(sample_id)
        else:
            both_wrong += 1

    total = len(expected)
    delta = 100 * sum(differences) / total
    return {
        "schema_version": "edgemed-paired-mcq-comparison/v1",
        "complete": True,
        "total": total,
        "direction": "B_minus_A_higher_is_better",
        "accuracy_a_percent": 100 * (both_correct + a_only_correct) / total,
        "accuracy_b_percent": 100 * (both_correct + b_only_correct) / total,
        "delta_accuracy_points": delta,
        "paired_bootstrap_95_percent": paired_bootstrap_ci(
            differences, repetitions=repetitions, seed=seed
        ),
        "mcnemar_exact_two_sided_p": mcnemar_exact_p_value(
            a_only_correct, b_only_correct
        ),
        "contingency": {
            "both_correct": both_correct,
            "a_only_correct": a_only_correct,
            "b_only_correct": b_only_correct,
            "both_wrong": both_wrong,
        },
        "changed_samples": {"b_gains": gains, "b_losses": losses},
        "bootstrap": {"repetitions": repetitions, "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    result = compare(
        read_jsonl(args.references),
        read_jsonl(args.predictions_a),
        read_jsonl(args.predictions_b),
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    result["source_hashes"] = {
        "references_sha256": sha256_file(args.references),
        "predictions_a_sha256": sha256_file(args.predictions_a),
        "predictions_b_sha256": sha256_file(args.predictions_b),
    }
    write_json(args.output, result)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
