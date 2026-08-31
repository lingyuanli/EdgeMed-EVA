"""Score Med-CMR MCQ predictions without exposing references to the inference process."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json

TASK_CODES = {
    "Small-object Detection": "SOD",
    "Fine-detail Discrimination": "FDD",
    "Spatial Understanding": "SU",
    "Temporal Prediction": "TP",
    "Causal Reasoning": "CR",
    "Long-tail Generalization": "LTG",
    "Multi-source Integration": "MSI",
}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [100 * (center - margin), 100 * (center + margin)]


def metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "correct": successes,
        "total": total,
        "accuracy": successes / total if total else None,
        "accuracy_percent": 100 * successes / total if total else None,
        "wilson_95_percent": wilson(successes, total),
        "direction": "higher_is_better",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    references = read_jsonl(args.references)
    predictions = read_jsonl(args.predictions)
    tasks = {row["sample_id"]: row["task"] for row in manifest}
    answers = {row["sample_id"]: str(row["answer"]).upper() for row in references}
    if len(tasks) != len(manifest) or len(answers) != len(references):
        raise ValueError("Duplicate sample ids in manifest or references")
    if set(tasks) != set(answers):
        raise ValueError("Manifest/reference sample ids differ")

    by_sample: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        sample_id = prediction["sample_id"]
        if sample_id in by_sample:
            raise ValueError(f"Duplicate prediction: {sample_id}")
        if sample_id not in answers:
            raise KeyError(f"Prediction outside contract: {sample_id}")
        by_sample[sample_id] = prediction

    missing = sorted(set(answers) - set(by_sample))
    if missing and not args.allow_incomplete:
        raise ValueError(f"Missing {len(missing)} predictions; first={missing[:10]}")

    task_totals: Counter[str] = Counter()
    task_correct: Counter[str] = Counter()
    parse_statuses: Counter[str] = Counter()
    correct_total = 0
    for sample_id, prediction in by_sample.items():
        task = tasks[sample_id]
        task_totals[task] += 1
        parse_statuses[str(prediction.get("parse_status"))] += 1
        is_correct = prediction.get("parsed_answer") == answers[sample_id]
        if is_correct:
            correct_total += 1
            task_correct[task] += 1

    metrics = {
        "schema_version": "medcmr-mcq-metrics/v1",
        "complete": not missing and len(by_sample) == len(answers),
        "expected": len(answers),
        "observed": len(by_sample),
        "missing": len(missing),
        "overall": metric(correct_total, len(by_sample)),
        "invalid_parse": {
            "count": parse_statuses["invalid"],
            "total": len(by_sample),
            "rate": parse_statuses["invalid"] / len(by_sample) if by_sample else None,
            "rate_percent": 100 * parse_statuses["invalid"] / len(by_sample) if by_sample else None,
            "direction": "lower_is_better",
        },
        "by_task": {
            TASK_CODES.get(task, task): metric(task_correct[task], task_totals[task])
            for task in sorted(task_totals)
        },
        "parse_status_counts": dict(sorted(parse_statuses.items())),
        "source_hashes": {
            "manifest_sha256": sha256_file(args.manifest),
            "references_sha256": sha256_file(args.references),
            "predictions_sha256": sha256_file(args.predictions),
        },
    }
    write_json(args.output, metrics)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
