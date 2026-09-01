"""Fail-closed provenance and benchmark-overlap gate for external development data."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from PIL import Image

from .io import read_jsonl, sha256_file, write_json

REQUIRED_FIELDS = {
    "record_id",
    "source_dataset",
    "source_version",
    "license",
    "patient_group_hash",
    "image_path",
    "image_sha256",
    "question",
    "answer",
    "annotation_type",
    "quality_status",
    "benchmark_overlap",
}
ANNOTATION_TYPES = {"human", "report-derived", "synthetic"}
QUALITY_STATUSES = {"accepted", "quarantined"}
OVERLAP_STATUSES = {"none", "suspected", "blocked"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalized_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.casefold()))


def token_set(text: str) -> set[str]:
    return set(normalized_text(text).split())


def image_dhash(path: Path) -> int:
    with Image.open(path) as source:
        image = source.convert("L").resize((9, 8))
    pixels = list(image.get_flattened_data())
    value = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    return value


@dataclass
class _DHashNode:
    value: int
    sample_ids: list[str] = field(default_factory=list)
    children: dict[int, "_DHashNode"] = field(default_factory=dict)


class _DHashBKTree:
    """Small metric index preserving exact Hamming-threshold semantics."""

    def __init__(self) -> None:
        self.root: _DHashNode | None = None

    def add(self, sample_id: str, value: int) -> None:
        if self.root is None:
            self.root = _DHashNode(value, [sample_id])
            return
        node = self.root
        while True:
            distance = (value ^ node.value).bit_count()
            if distance == 0:
                node.sample_ids.append(sample_id)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _DHashNode(value, [sample_id])
                return
            node = child

    def query(self, value: int, threshold: int) -> list[tuple[str, int]]:
        if self.root is None:
            return []
        matches: list[tuple[str, int]] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node.value).bit_count()
            if distance <= threshold:
                matches.extend((sample_id, distance) for sample_id in node.sample_ids)
            lower, upper = distance - threshold, distance + threshold
            pending.extend(
                child for edge, child in node.children.items() if lower <= edge <= upper
            )
        return matches


def _validate_schema(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        record_id = row.get("record_id", f"row-{index}")
        missing = sorted(REQUIRED_FIELDS.difference(row))
        if missing:
            problems.append({"record_id": record_id, "kind": "missing_fields", "fields": missing})
            continue
        if not isinstance(record_id, str) or not record_id.strip():
            problems.append({"record_id": record_id, "kind": "invalid_record_id"})
        elif record_id in seen:
            problems.append({"record_id": record_id, "kind": "duplicate_record_id"})
        seen.add(record_id)
        for field in ("source_dataset", "source_version", "license", "patient_group_hash"):
            if not isinstance(row[field], str) or not row[field].strip():
                problems.append({"record_id": record_id, "kind": "empty_provenance", "field": field})
        if row["annotation_type"] not in ANNOTATION_TYPES:
            problems.append({"record_id": record_id, "kind": "invalid_annotation_type"})
        if row["quality_status"] not in QUALITY_STATUSES:
            problems.append({"record_id": record_id, "kind": "invalid_quality_status"})
        if row["benchmark_overlap"] not in OVERLAP_STATUSES:
            problems.append({"record_id": record_id, "kind": "invalid_overlap_status"})
        if not isinstance(row["question"], str) or not normalized_text(row["question"]):
            problems.append({"record_id": record_id, "kind": "empty_question"})
        if not isinstance(row["answer"], str) or not row["answer"].strip():
            problems.append({"record_id": record_id, "kind": "empty_answer"})
    return problems


def validate_external_data(
    rows: list[dict[str, Any]],
    external_data_root: Path,
    benchmark_rows: list[dict[str, Any]],
    benchmark_data_root: Path,
    text_similarity_threshold: float = 0.92,
    image_hamming_threshold: int = 4,
) -> dict[str, Any]:
    schema_problems = _validate_schema(rows)
    if schema_problems:
        return {
            "schema_version": "edgemed-external-data-gate/v1",
            "status": "failed",
            "row_count": len(rows),
            "schema_problems": schema_problems,
            "overlap_checks_run": False,
        }

    benchmark_image_hashes = {row["image_sha256"]: row["sample_id"] for row in benchmark_rows}
    benchmark_texts = [(row["sample_id"], normalized_text(row["question"])) for row in benchmark_rows]
    benchmark_exact_text = {text: sample_id for sample_id, text in benchmark_texts}

    token_frequency = Counter(token for _, text in benchmark_texts for token in set(text.split()))
    max_frequency = max(1, int(len(benchmark_rows) * 0.05))
    distinctive_index: dict[str, set[int]] = defaultdict(set)
    for index, (_, text) in enumerate(benchmark_texts):
        for token in set(text.split()):
            if len(token) >= 4 and token_frequency[token] <= max_frequency:
                distinctive_index[token].add(index)

    benchmark_dhashes = _DHashBKTree()
    file_problems: list[dict[str, Any]] = []
    for row in benchmark_rows:
        path = (benchmark_data_root / row["image_path"]).resolve()
        if not path.is_file():
            file_problems.append(
                {"record_id": row["sample_id"], "kind": "missing_benchmark_image", "path": str(path)}
            )
            continue
        benchmark_dhashes.add(row["sample_id"], image_dhash(path))

    overlaps: list[dict[str, Any]] = []
    checked = 0
    for row in rows:
        if row["quality_status"] != "accepted":
            continue
        checked += 1
        record_id = row["record_id"]
        path = (external_data_root / row["image_path"]).resolve()
        if not path.is_file():
            file_problems.append({"record_id": record_id, "kind": "missing_external_image", "path": str(path)})
            continue
        actual_sha = sha256_file(path)
        if actual_sha != row["image_sha256"]:
            file_problems.append(
                {
                    "record_id": record_id,
                    "kind": "external_image_hash_mismatch",
                    "declared": row["image_sha256"],
                    "actual": actual_sha,
                }
            )
            continue
        if row["benchmark_overlap"] != "none":
            overlaps.append(
                {"record_id": record_id, "kind": "declared_overlap", "value": row["benchmark_overlap"]}
            )
        if actual_sha in benchmark_image_hashes:
            overlaps.append(
                {
                    "record_id": record_id,
                    "kind": "exact_image",
                    "benchmark_sample_id": benchmark_image_hashes[actual_sha],
                }
            )

        text = normalized_text(row["question"])
        if text in benchmark_exact_text:
            overlaps.append(
                {
                    "record_id": record_id,
                    "kind": "exact_text",
                    "benchmark_sample_id": benchmark_exact_text[text],
                }
            )
        else:
            candidates: set[int] = set()
            for token in token_set(text):
                candidates.update(distinctive_index.get(token, ()))
            for index in candidates:
                sample_id, benchmark_text = benchmark_texts[index]
                similarity = SequenceMatcher(None, text, benchmark_text).ratio()
                if similarity >= text_similarity_threshold:
                    overlaps.append(
                        {
                            "record_id": record_id,
                            "kind": "near_text",
                            "benchmark_sample_id": sample_id,
                            "similarity": similarity,
                        }
                    )

        dhash = image_dhash(path)
        for sample_id, distance in benchmark_dhashes.query(dhash, image_hamming_threshold):
            overlaps.append(
                {
                    "record_id": record_id,
                    "kind": "near_image",
                    "benchmark_sample_id": sample_id,
                    "hamming_distance": distance,
                }
            )

    status = "passed" if checked > 0 and not file_problems and not overlaps else "failed"
    return {
        "schema_version": "edgemed-external-data-gate/v1",
        "status": status,
        "row_count": len(rows),
        "accepted_checked": checked,
        "quarantined_skipped": sum(row["quality_status"] == "quarantined" for row in rows),
        "schema_problems": [],
        "file_problems": file_problems,
        "overlaps": overlaps,
        "overlap_checks_run": True,
        "thresholds": {
            "near_text_sequence_ratio": text_similarity_threshold,
            "near_image_dhash_hamming": image_hamming_threshold,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--benchmark-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-similarity-threshold", type=float, default=0.92)
    parser.add_argument("--image-hamming-threshold", type=int, default=4)
    args = parser.parse_args()

    report = validate_external_data(
        read_jsonl(args.manifest),
        args.data_root,
        read_jsonl(args.benchmark_manifest),
        args.benchmark_data_root,
        args.text_similarity_threshold,
        args.image_hamming_threshold,
    )
    report["source_hashes"] = {
        "external_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_sha256": sha256_file(args.benchmark_manifest),
    }
    write_json(args.output, report)
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
