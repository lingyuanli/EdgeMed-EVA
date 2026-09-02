"""Paired comparison for complete open-ended retention predictions."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json
from .score_open import normalize_answer, token_f1


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(
    differences: list[float], *, repetitions: int, seed: int
) -> list[float]:
    if not differences or repetitions < 1:
        raise ValueError("Non-empty differences and positive repetitions are required")
    rng = random.Random(seed)
    size = len(differences)
    draws = [
        100 * sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(repetitions)
    ]
    return [_quantile(draws, 0.025), _quantile(draws, 0.975)]


def _unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate {label} sample id: {sample_id}")
        result[sample_id] = row
    return result


def compare_open(
    references: list[dict[str, Any]],
    predictions_a: list[dict[str, Any]],
    predictions_b: list[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 20260902,
) -> dict[str, Any]:
    answers = {key: row["answer"] for key, row in _unique(references, "reference").items()}
    by_a = _unique(predictions_a, "prediction A")
    by_b = _unique(predictions_b, "prediction B")
    expected = set(answers)
    if set(by_a) != expected or set(by_b) != expected:
        raise ValueError("Both prediction files must exactly match the reference sample ids")

    exact_differences: list[float] = []
    f1_differences: list[float] = []
    exact_a = exact_b = 0
    f1_a = f1_b = 0.0
    for sample_id in sorted(expected):
        reference = answers[sample_id]
        answer_a = by_a[sample_id].get("parsed_answer") or ""
        answer_b = by_b[sample_id].get("parsed_answer") or ""
        is_exact_a = normalize_answer(answer_a) == normalize_answer(reference)
        is_exact_b = normalize_answer(answer_b) == normalize_answer(reference)
        sample_f1_a = token_f1(answer_a, reference)
        sample_f1_b = token_f1(answer_b, reference)
        exact_a += int(is_exact_a)
        exact_b += int(is_exact_b)
        f1_a += sample_f1_a
        f1_b += sample_f1_b
        exact_differences.append(float(is_exact_b) - float(is_exact_a))
        f1_differences.append(sample_f1_b - sample_f1_a)

    total = len(expected)
    return {
        "schema_version": "edgemed-paired-open-proxy-comparison/v1",
        "metric_status": "external_retention_proxy_not_medcmr_official",
        "complete": True,
        "total": total,
        "direction": "B_minus_A_higher_is_better",
        "normalized_exact_a_percent": 100 * exact_a / total,
        "normalized_exact_b_percent": 100 * exact_b / total,
        "delta_normalized_exact_points": 100 * (exact_b - exact_a) / total,
        "normalized_exact_bootstrap_95_percent": _bootstrap_ci(
            exact_differences, repetitions=repetitions, seed=seed
        ),
        "mean_token_f1_a_percent": 100 * f1_a / total,
        "mean_token_f1_b_percent": 100 * f1_b / total,
        "delta_mean_token_f1_points": 100 * (f1_b - f1_a) / total,
        "token_f1_bootstrap_95_percent": _bootstrap_ci(
            f1_differences, repetitions=repetitions, seed=seed + 1
        ),
        "bootstrap": {"repetitions": repetitions, "seed": seed},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    result = compare_open(
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
