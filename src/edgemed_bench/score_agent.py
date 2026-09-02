"""Reference-isolated E0-E3 scoring for medical Agent predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import read_jsonl, reject_reference_fields, sha256_file, write_json


def _unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["sample_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample ids in {label}")
    return result


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) and 0 <= value <= 1000 for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _schema_valid(
    sample: dict[str, Any], prediction: dict[str, Any], traces: dict[str, dict[str, Any]]
) -> tuple[bool, bool, bool]:
    output = prediction.get("agent_output")
    if not isinstance(output, dict):
        return False, False, False
    required = {
        "sample_id", "hypotheses", "evidence", "answer", "answer_evidence_ids",
        "confidence", "insufficient_evidence", "tool_trace_ids",
    }
    schema_ok = required <= output.keys() and output.get("sample_id") == sample["sample_id"]
    schema_ok = schema_ok and isinstance(output.get("confidence"), (int, float))
    schema_ok = schema_ok and 0 <= float(output.get("confidence", -1)) <= 1
    schema_ok = schema_ok and isinstance(output.get("hypotheses"), list)
    schema_ok = schema_ok and isinstance(output.get("answer_evidence_ids"), list)
    schema_ok = schema_ok and isinstance(output.get("tool_trace_ids"), list)
    schema_ok = schema_ok and isinstance(output.get("insufficient_evidence"), bool)
    schema_ok = schema_ok and prediction.get("status", "completed") == "completed"
    if sample.get("question_type") == "mcq":
        schema_ok = schema_ok and output.get("answer") in sample.get("options", {})
    evidence = output.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False, False, False
    evidence_ids = [item.get("evidence_id") for item in evidence if isinstance(item, dict)]
    citation_ok = len(evidence_ids) == len(evidence) == len(set(evidence_ids))
    citation_ok = citation_ok and all(isinstance(item, str) and item for item in evidence_ids)
    citation_ok = citation_ok and bool(output.get("answer_evidence_ids"))
    citation_ok = citation_ok and set(output.get("answer_evidence_ids", [])) <= set(evidence_ids)
    media_ids = {str(item["media_id"]) for item in sample["media"]}
    trace_ids = output.get("tool_trace_ids")
    trace_ids = trace_ids if isinstance(trace_ids, list) else []
    prediction_trace_ids = prediction.get("tool_trace_ids")
    prediction_trace_ids = prediction_trace_ids if isinstance(prediction_trace_ids, list) else []
    trace_ok = (
        bool(trace_ids)
        and set(trace_ids) <= set(prediction_trace_ids)
        and all(trace_id in traces and traces[trace_id].get("status") == "completed" for trace_id in trace_ids)
        and all(traces[trace_id].get("sample_id") == sample["sample_id"] for trace_id in trace_ids)
        and all(
            trace_id in traces and traces[trace_id].get("sample_id") == sample["sample_id"]
            for trace_id in prediction_trace_ids
        )
    )
    bound_traces = [traces[trace_id] for trace_id in trace_ids if trace_id in traces]
    hypotheses = output.get("hypotheses") if isinstance(output.get("hypotheses"), list) else []
    hypothesis_ids = {
        item.get("id") for item in hypotheses if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    hypotheses_ok = len(hypothesis_ids) == len(hypotheses) and all(
        isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and bool(item.get("id"))
        and isinstance(item.get("label"), str)
        and bool(item.get("label").strip())
        and item.get("status") in {"supported", "refuted", "uncertain"}
        for item in hypotheses
    )

    def evidence_item_valid(item: Any) -> bool:
        if not isinstance(item, dict) or item.get("media_id") not in media_ids:
            return False
        if not str(item.get("observation", "")).strip():
            return False
        if not isinstance(item.get("view_or_time"), str) or not item["view_or_time"].strip():
            return False
        if item.get("acquisition") not in {"inspect_overview", "temporal_skim", "region_inspect"}:
            return False
        if not isinstance(item.get("confidence"), (int, float)) or isinstance(item.get("confidence"), bool):
            return False
        if not 0 <= float(item["confidence"]) <= 1:
            return False
        for relation in ("supports", "contradicts"):
            if not isinstance(item.get(relation), list) or not set(item[relation]) <= hypothesis_ids:
                return False
        box = item.get("region_xyxy_1000")
        if box is not None and not _valid_box(box):
            return False
        compatible = [
            trace for trace in bound_traces
            if trace.get("tool_name") == item.get("acquisition")
            and any(
                frame.get("media_id") == item.get("media_id")
                for frame in trace.get("selected_frames", [])
                if isinstance(frame, dict)
            )
        ]
        if box is None:
            return item.get("acquisition") != "region_inspect" and bool(compatible)
        return any(
            trace.get("request", {}).get("media_id") == item.get("media_id")
            and trace.get("request", {}).get("region_xyxy_1000") == box
            for trace in compatible
        )

    evidence_ok = all(evidence_item_valid(item) for item in evidence)
    return bool(schema_ok and hypotheses_ok and citation_ok and evidence_ok and trace_ok), bool(citation_ok), bool(trace_ok)


def score_agent_rows(
    inference_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    tool_trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reject_reference_fields(inference_rows)
    samples = _unique(inference_rows, "inference manifest")
    references = _unique(reference_rows, "references")
    predictions = _unique(prediction_rows, "predictions")
    traces = {str(row["trace_id"]): row for row in tool_trace_rows}
    if len(traces) != len(tool_trace_rows):
        raise ValueError("Duplicate tool trace ids")
    if set(samples) != set(references) or set(samples) != set(predictions):
        raise ValueError("Inference, reference, and prediction sample ids must match exactly")

    schema_valid = citation_valid = trace_bound = correct = 0
    grounded_samples = 0
    localization_ious: list[float] = []
    for sample_id, sample in samples.items():
        prediction = predictions[sample_id]
        reference = references[sample_id]
        valid, cited, bound = _schema_valid(sample, prediction, traces)
        schema_valid += int(valid)
        citation_valid += int(cited)
        trace_bound += int(bound)
        correct += int(prediction.get("parsed_answer") == reference.get("answer"))
        output = prediction.get("agent_output")
        predicted_evidence = output.get("evidence", []) if isinstance(output, dict) else []
        reference_evidence = reference.get("evidence", [])
        sample_ious = []
        for predicted in predicted_evidence:
            pbox = predicted.get("region_xyxy_1000") if isinstance(predicted, dict) else None
            if not _valid_box(pbox):
                continue
            compatible = [
                item for item in reference_evidence
                if item.get("media_id") == predicted.get("media_id") and _valid_box(item.get("region_xyxy_1000"))
            ]
            if compatible:
                sample_ious.append(max(box_iou(pbox, item["region_xyxy_1000"]) for item in compatible))
        if sample_ious:
            grounded_samples += 1
            localization_ious.append(max(sample_ious))
    count = len(samples)
    return {
        "schema_version": "edgemed-medical-agent-metrics/v1",
        "status": "completed",
        "sample_count": count,
        "e0_structure": {
            "schema_valid_rate": schema_valid / count,
            "citation_valid_rate": citation_valid / count,
            "tool_trace_bound_rate": trace_bound / count,
        },
        "e1_answer": {"accuracy": correct / count, "correct": correct},
        "e2_evidence": {
            "localized_sample_count": grounded_samples,
            "mean_best_region_iou": (
                sum(localization_ious) / len(localization_ious) if localization_ious else None
            ),
            "scope": "reference-box localization proxy; not semantic/clinical correctness",
        },
        "e3_causal": {
            "status": "DEFER",
            "reason": "No matched no-tool, forced-tool, or evidence-intervention arm supplied.",
        },
        "reference_access": "scorer_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tool-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = score_agent_rows(
        read_jsonl(args.inference_manifest), read_jsonl(args.references),
        read_jsonl(args.predictions), read_jsonl(args.tool_traces),
    )
    metrics["source_hashes"] = {
        "inference_manifest_sha256": sha256_file(args.inference_manifest),
        "references_sha256": sha256_file(args.references),
        "predictions_sha256": sha256_file(args.predictions),
        "tool_traces_sha256": sha256_file(args.tool_traces),
    }
    write_json(args.output, metrics)


if __name__ == "__main__":
    main()
