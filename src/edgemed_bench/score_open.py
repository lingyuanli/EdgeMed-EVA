"""Score open-ended predictions on an answer-isolated external retention set."""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json


def normalize_answer(value: object) -> str:
    """Apply a transparent, language-agnostic normalization for proxy scoring."""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def token_f1(prediction: object, reference: object) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def _aggregate(rows: list[tuple[bool, float]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(int(is_exact) for is_exact, _ in rows)
    f1_sum = sum(f1 for _, f1 in rows)
    return {
        "total": total,
        "normalized_exact": exact / total if total else None,
        "normalized_exact_percent": 100 * exact / total if total else None,
        "mean_token_f1": f1_sum / total if total else None,
        "mean_token_f1_percent": 100 * f1_sum / total if total else None,
        "direction": "higher_is_better",
    }


def score_open_rows(
    manifest: list[dict[str, Any]],
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    manifest_by_id = {str(row["sample_id"]): row for row in manifest}
    answers = {str(row["sample_id"]): row["answer"] for row in references}
    if len(manifest_by_id) != len(manifest) or len(answers) != len(references):
        raise ValueError("Duplicate sample ids in manifest or references")
    if set(manifest_by_id) != set(answers):
        raise ValueError("Manifest/reference sample ids differ")

    prediction_by_id: dict[str, dict[str, Any]] = {}
    for row in predictions:
        sample_id = str(row["sample_id"])
        if sample_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction: {sample_id}")
        if sample_id not in answers:
            raise KeyError(f"Prediction outside contract: {sample_id}")
        prediction_by_id[sample_id] = row
    missing = sorted(set(answers) - set(prediction_by_id))
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)} predictions; first={missing[:10]}")

    overall: list[tuple[bool, float]] = []
    slices: dict[str, dict[str, list[tuple[bool, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    parse_statuses: Counter[str] = Counter()
    for sample_id, prediction in prediction_by_id.items():
        predicted = prediction.get("parsed_answer") or ""
        reference = answers[sample_id]
        values = (
            normalize_answer(predicted) == normalize_answer(reference),
            token_f1(predicted, reference),
        )
        overall.append(values)
        parse_statuses[str(prediction.get("parse_status"))] += 1
        metadata = manifest_by_id[sample_id].get("evaluation_metadata") or {}
        if isinstance(metadata, dict):
            for key in ("answer_type", "base_type", "content_type", "modality"):
                value = metadata.get(key)
                if value is not None:
                    slices[key][str(value)].append(values)

    return {
        "schema_version": "edgemed-open-proxy-metrics/v1",
        "metric_status": "external_retention_proxy_not_medcmr_official",
        "complete": not missing and len(prediction_by_id) == len(answers),
        "expected": len(answers),
        "observed": len(prediction_by_id),
        "missing": len(missing),
        "overall": _aggregate(overall),
        "by_slice": {
            key: {value: _aggregate(rows) for value, rows in sorted(groups.items())}
            for key, groups in sorted(slices.items())
        },
        "parse_status_counts": dict(sorted(parse_statuses.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = score_open_rows(
        read_jsonl(args.manifest),
        read_jsonl(args.references),
        read_jsonl(args.predictions),
        allow_incomplete=args.allow_incomplete,
    )
    result["source_hashes"] = {
        "manifest_sha256": sha256_file(args.manifest),
        "references_sha256": sha256_file(args.references),
        "predictions_sha256": sha256_file(args.predictions),
    }
    write_json(args.output, result)
    print(args.output.read_text())


if __name__ == "__main__":
    main()
