"""Independent scorer for question-conditioned region localization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .io import read_jsonl, sha256_file, write_json


def box_iou(first: list[int], second: list[int]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def valid_box(value: Any) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) for item in value)
    ):
        return False
    x1, y1, x2, y2 = value
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def score_locator_predictions(
    predictions: list[dict[str, Any]], targets: list[dict[str, Any]]
) -> dict[str, Any]:
    prediction_map = {str(row["sample_id"]): row for row in predictions}
    target_map = {str(row["sample_id"]): row for row in targets}
    if len(prediction_map) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("Duplicate locator sample ids")
    if set(prediction_map) != set(target_map):
        raise ValueError("Locator prediction and target sample ids differ")
    records = []
    for sample_id, target in target_map.items():
        prediction = prediction_map[sample_id]
        tool_call = prediction.get("tool_call")
        arguments = tool_call.get("arguments") if isinstance(tool_call, dict) else None
        predicted_box = arguments.get("region_xyxy_1000") if isinstance(arguments, dict) else None
        output_valid = (
            prediction.get("status") == "completed"
            and isinstance(tool_call, dict)
            and tool_call.get("name") == "region_inspect"
            and isinstance(arguments, dict)
            and arguments.get("media_id") == "image-0"
            and valid_box(predicted_box)
        )
        target_box = target.get("region_xyxy_1000")
        if not valid_box(target_box):
            raise ValueError(f"Invalid locator target box: {sample_id}")
        area = (
            (predicted_box[2] - predicted_box[0])
            * (predicted_box[3] - predicted_box[1])
            / 1_000_000
            if output_valid
            else None
        )
        iou = box_iou(predicted_box, target_box) if output_valid else 0.0
        records.append(
            {
                "sample_id": sample_id,
                "target_label": target.get("target_label"),
                "output_valid": output_valid,
                "targeted_area": area is not None and 0.01 <= area <= 0.64,
                "predicted_box": predicted_box if output_valid else None,
                "target_box": target_box,
                "iou": iou,
            }
        )
    count = len(records)
    ious = [row["iou"] for row in records]
    return {
        "sample_count": count,
        "valid_output_rate": sum(row["output_valid"] for row in records) / count,
        "targeted_area_rate": sum(row["targeted_area"] for row in records) / count,
        "mean_iou": sum(ious) / count,
        "iou_at_0_3": sum(value >= 0.3 for value in ious) / count,
        "iou_at_0_5": sum(value >= 0.5 for value in ious) / count,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    targets = read_jsonl(args.targets)
    metrics = score_locator_predictions(predictions, targets)
    metrics.update(
        {
            "schema_version": "edgemed-locator-metrics/v1",
            "source_hashes": {
                "predictions_sha256": sha256_file(args.predictions),
                "targets_sha256": sha256_file(args.targets),
            },
        }
    )
    if args.baseline_predictions is not None:
        baseline = score_locator_predictions(
            read_jsonl(args.baseline_predictions), targets
        )
        current_by_id = {row["sample_id"]: row["iou"] for row in metrics["records"]}
        baseline_by_id = {row["sample_id"]: row["iou"] for row in baseline["records"]}
        metrics["paired_vs_baseline"] = {
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions),
            "mean_iou_delta": metrics["mean_iou"] - baseline["mean_iou"],
            "iou_at_0_3_delta": metrics["iou_at_0_3"] - baseline["iou_at_0_3"],
            "improved": sum(
                current_by_id[key] > baseline_by_id[key] for key in current_by_id
            ),
            "tied": sum(
                current_by_id[key] == baseline_by_id[key] for key in current_by_id
            ),
            "worsened": sum(
                current_by_id[key] < baseline_by_id[key] for key in current_by_id
            ),
        }
    write_json(args.output, metrics)
    summary = {key: value for key, value in metrics.items() if key != "records"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
