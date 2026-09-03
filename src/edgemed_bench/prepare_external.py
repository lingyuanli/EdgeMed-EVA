"""Build source-pinned external medical VQA manifests without benchmark labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .io import append_jsonl, read_jsonl, reject_reference_fields, sha256_file, write_json

PMC_REVISION = "b56ae594f794867893143b337b4118a835794647"
SLAKE_REVISION = "a9083ce6c34ac3ffb17671a605962924d8a8f9e9"
PMC_ALLOWED_LICENSES = {"CC0", "CC BY", "CC BY-SA"}
PMC_ID_RE = re.compile(r"^(PMC\d+)", re.IGNORECASE)
CHOICE_PREFIX_RE = re.compile(r"^[A-D]\s*:\s*", re.IGNORECASE)
QUESTION_TOKEN_RE = re.compile(r"[a-z0-9]+")
YES_NO_QUESTION_RE = re.compile(
    r"^(is|are|does|do|did|can|could|has|have|was|were|will|would|should)\b",
    re.IGNORECASE,
)


def _group_hash(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()


def _selection_key(seed: str, source_id: str) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()


def _safe_relative_path(value: str) -> Path | None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        return None
    return path


def _clean_choice(value: str) -> str:
    return CHOICE_PREFIX_RE.sub("", value.strip(), count=1).strip()


def _normalized_question(value: str) -> str:
    return " ".join(QUESTION_TOKEN_RE.findall(value.casefold()))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row)
    os.chmod(path, mode)


def _image_record(path: Path) -> tuple[str, str] | None:
    if not path.is_file():
        return None
    return path.as_posix(), sha256_file(path)


def extract_zip_safe(archive_path: Path, output_root: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Archive SHA-256 mismatch: {actual_sha256}")
    output_root.mkdir(parents=True, exist_ok=True)
    files = 0
    total_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if stat.S_ISLNK(mode):
                raise ValueError(f"Archive symlink is not allowed: {info.filename}")
            if info.is_dir():
                continue
            target = output_root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(output_root.resolve()):
                raise ValueError(f"Archive target escaped output root: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            files += 1
            total_bytes += info.file_size
    return {
        "schema_version": "edgemed-external-extraction/v1",
        "archive_sha256": actual_sha256,
        "files": files,
        "uncompressed_bytes": total_bytes,
    }


def load_pmc_licenses(path: Path) -> dict[str, str]:
    licenses: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pmcid = (row.get("Accession ID") or "").strip().upper()
            license_name = (row.get("License") or "").strip().upper()
            if pmcid:
                licenses[pmcid] = license_name
    return licenses


def build_pmc_vqa(
    csv_path: Path,
    license_path: Path,
    image_root: Path,
    output: Path,
    report_path: Path,
    limit: int,
    seed: str,
    max_per_image: int = 1,
    required_split: str = "train",
    cohort: str = "train-seed",
    exclude_manifest: Path | None = None,
) -> dict[str, Any]:
    if limit <= 0 or max_per_image <= 0:
        raise ValueError("limit and max_per_image must be positive")
    licenses = load_pmc_licenses(license_path)
    excluded_questions = (
        {_normalized_question(row["question"]) for row in read_jsonl(exclude_manifest)}
        if exclude_manifest is not None
        else set()
    )
    candidates: list[dict[str, Any]] = []
    rejected = Counter()
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            source_id = (row.get("index") or str(row_number - 2)).strip()
            figure_path = (row.get("Figure_path") or "").strip()
            pmc_match = PMC_ID_RE.match(figure_path)
            pmcid = pmc_match.group(1).upper() if pmc_match else ""
            article_license = licenses.get(pmcid, "")
            answer = (row.get("Answer") or "").strip().upper()
            choices = {letter: _clean_choice(row.get(f"Choice {letter}") or "") for letter in "ABCD"}
            if (row.get("split") or "").strip().casefold() != required_split.casefold():
                rejected["wrong_split"] += 1
                continue
            if not pmcid or article_license not in PMC_ALLOWED_LICENSES:
                rejected["license_or_pmcid"] += 1
                continue
            if answer not in choices or not all(choices.values()):
                rejected["invalid_answer_or_choices"] += 1
                continue
            question = (row.get("Question") or "").strip()
            if not question:
                rejected["empty_question"] += 1
                continue
            if _normalized_question(question) in excluded_questions:
                rejected["excluded_question_overlap"] += 1
                continue
            relative_image_path = _safe_relative_path(figure_path)
            if relative_image_path is None:
                rejected["unsafe_image_path"] += 1
                continue
            candidates.append(
                {
                    "source_id": source_id,
                    "figure_path": figure_path,
                    "pmcid": pmcid,
                    "article_license": article_license,
                    "question": question,
                    "answer": answer,
                    "answer_text": choices[answer],
                    "options": choices,
                    "source_caption": (row.get("Caption") or "").strip(),
                    "rank": _selection_key(seed, source_id),
                }
            )

    candidates.sort(key=lambda row: (row["rank"], row["source_id"]))
    selected: list[dict[str, Any]] = []
    per_image = Counter()
    for row in candidates:
        if per_image[row["figure_path"]] >= max_per_image:
            rejected["per_image_cap"] += 1
            continue
        image_info = _image_record(image_root / row["figure_path"])
        if image_info is None:
            rejected["missing_image"] += 1
            continue
        row["image_sha256"] = image_info[1]
        per_image[row["figure_path"]] += 1
        selected.append(row)
        if len(selected) == limit:
            break

    manifest = []
    for index, row in enumerate(selected):
        manifest.append(
            {
                "record_id": f"pmc-vqa-v2-{cohort}-{row['source_id']}",
                "source_dataset": "RadGenome/PMC-VQA-v2",
                "source_version": PMC_REVISION,
                "source_record_id": row["source_id"],
                "source_split": required_split,
                "cohort": cohort,
                "source_article_id": row["pmcid"],
                "license": f"CC-BY-SA; source-article={row['article_license']}",
                "patient_group_hash": _group_hash("pmc-article", row["pmcid"]),
                "image_path": row["figure_path"],
                "image_sha256": row["image_sha256"],
                "question": row["question"],
                "options": row["options"],
                "answer": row["answer"],
                "answer_text": row["answer_text"],
                "source_caption": row["source_caption"],
                "evidence_source": "caption-derived",
                "evidence_target_eligible": False,
                "annotation_type": "synthetic",
                "quality_status": "accepted",
                "benchmark_overlap": "none",
                "selection_index": index,
            }
        )
    _write_jsonl(output, manifest)
    report = {
        "schema_version": "edgemed-external-build/v1",
        "source_dataset": "RadGenome/PMC-VQA-v2",
        "source_version": PMC_REVISION,
        "source_hashes": {
            "csv_sha256": sha256_file(csv_path),
            "license_csv_sha256": sha256_file(license_path),
        },
        "selection": {
            "seed": seed,
            "limit": limit,
            "max_per_image": max_per_image,
            "required_split": required_split,
            "cohort": cohort,
        },
        "eligible_before_selection": len(candidates),
        "written": len(manifest),
        "unique_images": len(per_image),
        "rejected": dict(sorted(rejected.items())),
    }
    if exclude_manifest is not None:
        report["source_hashes"]["exclusion_manifest_sha256"] = sha256_file(exclude_manifest)
    write_json(report_path, report)
    return report


def build_slake(
    json_path: Path,
    image_root: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("SLAKE validation source must be a JSON array")
    manifest: list[dict[str, Any]] = []
    rejected = Counter()
    image_hashes: dict[str, str] = {}
    for source_index, row in enumerate(raw):
        if not isinstance(row, dict):
            rejected["not_object"] += 1
            continue
        if row.get("q_lang") != "en":
            rejected["not_english"] += 1
            continue
        image_name = str(row.get("img_name") or "").strip()
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not image_name or not question or not answer:
            rejected["missing_required_value"] += 1
            continue
        relative_image_path = _safe_relative_path(image_name)
        if relative_image_path is None:
            rejected["unsafe_image_path"] += 1
            continue
        image_path = image_root / relative_image_path
        if image_name not in image_hashes:
            image_info = _image_record(image_path)
            if image_info is None:
                rejected["missing_image"] += 1
                continue
            image_hashes[image_name] = image_info[1]
        qid = str(row.get("qid", source_index))
        manifest.append(
            {
                "record_id": f"slake-validation-{qid}",
                "source_dataset": "BoKelvin/SLAKE",
                "source_version": SLAKE_REVISION,
                "source_record_id": qid,
                "source_split": "validation",
                "license": "CC-BY-4.0",
                "patient_group_hash": _group_hash("slake-image", image_name),
                "image_path": image_name,
                "image_sha256": image_hashes[image_name],
                "question": question,
                "answer": answer,
                "annotation_type": "human",
                "quality_status": "accepted",
                "benchmark_overlap": "none",
                "slake_metadata": {
                    key: row.get(key)
                    for key in ("location", "modality", "answer_type", "base_type", "content_type")
                },
            }
        )
    _write_jsonl(output, manifest)
    report = {
        "schema_version": "edgemed-external-build/v1",
        "source_dataset": "BoKelvin/SLAKE",
        "source_version": SLAKE_REVISION,
        "source_hashes": {"validation_json_sha256": sha256_file(json_path)},
        "written": len(manifest),
        "unique_images": len(image_hashes),
        "rejected": dict(sorted(rejected.items())),
    }
    write_json(report_path, report)
    return report


def split_surfaces(
    manifest_path: Path,
    kind: str,
    inference_output: Path,
    references_output: Path,
    report_path: Path,
) -> dict[str, Any]:
    rows = [row for row in read_jsonl(manifest_path) if row["quality_status"] == "accepted"]
    inference_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = row["record_id"]
        inference = {
            "sample_id": sample_id,
            "kind": kind,
            "question": row["question"],
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "task": row.get("cohort", row["source_dataset"]),
            "source_dataset": row["source_dataset"],
            "source_version": row["source_version"],
            "source_record_id": row.get("source_record_id"),
        }
        evaluation_metadata = row.get("slake_metadata")
        if isinstance(evaluation_metadata, dict):
            inference["evaluation_metadata"] = evaluation_metadata
        if kind == "mcq":
            options = row.get("options")
            if not isinstance(options, dict) or set(options) != set("ABCD"):
                raise ValueError(f"MCQ record has invalid options: {sample_id}")
            if row["answer"] not in options:
                raise ValueError(f"MCQ record has invalid answer: {sample_id}")
            inference["options"] = options
        inference_rows.append(inference)
        reference_rows.append({"sample_id": sample_id, "answer": row["answer"]})

    if not inference_rows:
        raise ValueError("No accepted records to split")
    reject_reference_fields(inference_rows)
    _write_jsonl(inference_output, inference_rows)
    _write_jsonl(references_output, reference_rows, mode=0o600)
    report = {
        "schema_version": "edgemed-external-surfaces/v1",
        "kind": kind,
        "count": len(inference_rows),
        "source_hashes": {
            "admitted_manifest_sha256": sha256_file(manifest_path),
            "inference_manifest_sha256": sha256_file(inference_output),
            "references_sha256": sha256_file(references_output),
        },
        "leakage_boundary": {
            "inference_has_reference_fields": False,
            "references_mode": "0600",
        },
    }
    write_json(report_path, report)
    return report


def build_slake_binary_surface(
    inference_path: Path,
    references_path: Path,
    inference_output: Path,
    references_output: Path,
    report_path: Path,
    *,
    limit: int = 96,
    max_per_image: int = 1,
    seed: str = "edgemed-slake-binary-v1",
) -> dict[str, Any]:
    """Derive a reference-blind yes/no MCQ cohort, then bind labels separately."""
    if limit <= 0 or max_per_image <= 0:
        raise ValueError("limit and max_per_image must be positive")
    inference_rows = read_jsonl(inference_path)
    reject_reference_fields(inference_rows)
    candidates = []
    rejected = Counter()
    for row in inference_rows:
        metadata = row.get("evaluation_metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        question = str(row.get("question", "")).strip()
        if metadata.get("answer_type") != "CLOSED":
            rejected["not_closed"] += 1
            continue
        if not YES_NO_QUESTION_RE.match(question) or " or " in question.casefold():
            rejected["not_unambiguous_yes_no_form"] += 1
            continue
        candidates.append({**row, "_rank": _selection_key(seed, str(row["sample_id"]))})
    candidates.sort(key=lambda row: (row["_rank"], str(row["sample_id"])))
    selected = []
    per_image = Counter()
    for row in candidates:
        image_key = str(row.get("image_sha256", row.get("image_path", "")))
        if per_image[image_key] >= max_per_image:
            rejected["per_image_cap"] += 1
            continue
        per_image[image_key] += 1
        selected.append(row)
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} rows but selected {len(selected)}")

    source_reference_rows = read_jsonl(references_path)
    source_references = {str(row["sample_id"]): row for row in source_reference_rows}
    if len(source_references) != len(source_reference_rows):
        raise ValueError("Duplicate source reference sample ids")
    derived_inference = []
    derived_references = []
    for row in selected:
        sample_id = str(row["sample_id"])
        if sample_id not in source_references:
            raise ValueError(f"Missing SLAKE reference: {sample_id}")
        answer = str(source_references[sample_id].get("answer", "")).strip().casefold()
        if answer not in {"yes", "no"}:
            raise ValueError(f"Frozen question-only selector admitted non-binary answer: {sample_id}")
        metadata = row.get("evaluation_metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        output_row = {key: value for key, value in row.items() if key != "_rank"}
        output_row.update(
            {
                "kind": "mcq",
                "options": {"A": "Yes", "B": "No"},
                "task": "slake-closed-binary",
                "modality": metadata.get("modality", "unknown"),
                "surface_derivation": "closed_interrogative_without_or/v1",
            }
        )
        derived_inference.append(output_row)
        derived_references.append(
            {"sample_id": sample_id, "answer": "A" if answer == "yes" else "B"}
        )
    reject_reference_fields(derived_inference)
    _write_jsonl(inference_output, derived_inference)
    _write_jsonl(references_output, derived_references, mode=0o600)
    report = {
        "schema_version": "edgemed-slake-binary-surface/v1",
        "selection": {
            "answer_blind": True,
            "seed": seed,
            "limit": limit,
            "max_per_image": max_per_image,
            "rule": "CLOSED metadata; interrogative prefix; exclude questions containing ' or '",
        },
        "eligible_before_image_cap": len(candidates),
        "written": len(derived_inference),
        "unique_images": len(per_image),
        "rejected": dict(sorted(rejected.items())),
        "source_hashes": {
            "source_inference_sha256": sha256_file(inference_path),
            "source_references_sha256": sha256_file(references_path),
            "inference_manifest_sha256": sha256_file(inference_output),
            "references_sha256": sha256_file(references_output),
        },
        "leakage_boundary": {
            "selection_fields": ["evaluation_metadata.answer_type", "question", "image_sha256"],
            "references_used_after_selection_only": True,
            "inference_has_reference_fields": False,
            "references_mode": "0600",
        },
    }
    write_json(report_path, report)
    return report


def _slake_detection_groups(
    detection_rows: Any, question: str
) -> dict[str, list[tuple[str, list[float]]]]:
    groups: dict[str, list[tuple[str, list[float]]]] = {}
    normalized_question = _normalized_question(question)
    if not isinstance(detection_rows, list):
        return groups
    for item in detection_rows:
        if not isinstance(item, dict):
            continue
        for label, box in item.items():
            normalized_label = _normalized_question(str(label))
            if not normalized_label or normalized_label not in normalized_question:
                continue
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(not isinstance(value, (int, float)) for value in box)
            ):
                continue
            numeric_box = [float(value) for value in box]
            if not all(math.isfinite(value) for value in numeric_box):
                continue
            x, y, width, height = numeric_box
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                continue
            groups.setdefault(normalized_label, []).append((str(label), numeric_box))
    return groups


def _normalize_detection_union(
    detections: list[tuple[str, list[float]]], image_width: int, image_height: int
) -> list[int]:
    if image_width <= 0 or image_height <= 0 or not detections:
        raise ValueError("Cannot normalize an empty detection group or empty image")
    x1 = min(box[0] for _, box in detections)
    y1 = min(box[1] for _, box in detections)
    x2 = max(box[0] + box[2] for _, box in detections)
    y2 = max(box[1] + box[3] for _, box in detections)
    normalized = [
        max(0, min(1000, math.floor(1000 * x1 / image_width))),
        max(0, min(1000, math.floor(1000 * y1 / image_height))),
        max(0, min(1000, math.ceil(1000 * x2 / image_width))),
        max(0, min(1000, math.ceil(1000 * y2 / image_height))),
    ]
    if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
        raise ValueError("Normalized detection union is empty")
    return normalized


def build_slake_localization_surface(
    json_path: Path,
    image_root: Path,
    inference_output: Path,
    targets_output: Path,
    report_path: Path,
    *,
    source_split: str,
    limit: int = 0,
    max_per_image: int = 2,
    max_per_label: int = 0,
    seed: str = "edgemed-slake-localization-v1",
    area_min: float = 0.01,
    area_max: float = 0.64,
) -> dict[str, Any]:
    """Build question-to-box supervision without reading SLAKE VQA answers."""
    if limit < 0 or max_per_image <= 0 or max_per_label < 0:
        raise ValueError(
            "limit/max_per_label must be non-negative and max_per_image must be positive"
        )
    if not (0 < area_min <= area_max < 1):
        raise ValueError("Expected 0 < area_min <= area_max < 1")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("SLAKE localization source must be a JSON array")

    candidates: list[dict[str, Any]] = []
    rejected = Counter()
    for source_index, row in enumerate(raw):
        if not isinstance(row, dict):
            rejected["not_object"] += 1
            continue
        if row.get("q_lang") != "en":
            rejected["not_english"] += 1
            continue
        image_name = str(row.get("img_name") or "").strip()
        question = str(row.get("question") or "").strip()
        relative_image_path = _safe_relative_path(image_name)
        if relative_image_path is None or not question:
            rejected["missing_or_unsafe_input"] += 1
            continue
        image_path = image_root / relative_image_path
        detection_path = image_path.parent / "detection.json"
        if not image_path.is_file() or not detection_path.is_file():
            rejected["missing_image_or_detection"] += 1
            continue
        groups = _slake_detection_groups(
            json.loads(detection_path.read_text(encoding="utf-8")), question
        )
        if len(groups) != 1:
            rejected["matched_label_count_not_one"] += 1
            continue
        normalized_label, detections = next(iter(groups.items()))
        with Image.open(image_path) as image:
            region = _normalize_detection_union(detections, image.width, image.height)
        area = (region[2] - region[0]) * (region[3] - region[1]) / 1_000_000
        if area < area_min or area > area_max:
            rejected["outside_area_interval"] += 1
            continue
        source_id = str(row.get("qid", source_index))
        label = detections[0][0]
        candidates.append(
            {
                "sample_id": f"slake-{source_split}-locator-{source_id}",
                "source_record_id": source_id,
                "question": question,
                "image_path": image_name,
                "image_sha256": sha256_file(image_path),
                "target_label": label,
                "normalized_label": normalized_label,
                "region_xyxy_1000": region,
                "region_area": area,
                "rank": _selection_key(seed, source_id),
            }
        )

    candidates.sort(key=lambda row: (row["rank"], row["sample_id"]))
    selected: list[dict[str, Any]] = []
    per_image = Counter()
    per_label = Counter()
    for row in candidates:
        if per_image[row["image_sha256"]] >= max_per_image:
            rejected["per_image_cap"] += 1
            continue
        if max_per_label and per_label[row["normalized_label"]] >= max_per_label:
            rejected["per_label_cap"] += 1
            continue
        per_image[row["image_sha256"]] += 1
        per_label[row["normalized_label"]] += 1
        selected.append(row)
        if limit and len(selected) == limit:
            break
    if not selected:
        raise ValueError("No SLAKE localization records selected")
    if limit and len(selected) != limit:
        raise ValueError(f"Requested {limit} rows but selected {len(selected)}")

    inference_rows = []
    target_rows = []
    for row in selected:
        inference_rows.append(
            {
                "sample_id": row["sample_id"],
                "kind": "localization",
                "question": row["question"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
                "task": "slake-question-conditioned-localization",
                "source_dataset": "BoKelvin/SLAKE",
                "source_version": SLAKE_REVISION,
                "source_split": source_split,
                "source_record_id": row["source_record_id"],
            }
        )
        target_rows.append(
            {
                "sample_id": row["sample_id"],
                "target_label": row["target_label"],
                "region_xyxy_1000": row["region_xyxy_1000"],
                "tool_call": {
                    "name": "region_inspect",
                    "arguments": {
                        "media_id": "image-0",
                        "region_xyxy_1000": row["region_xyxy_1000"],
                        "target": row["target_label"],
                    },
                },
            }
        )
    reject_reference_fields(inference_rows)
    _write_jsonl(inference_output, inference_rows)
    _write_jsonl(targets_output, target_rows, mode=0o600)
    report = {
        "schema_version": "edgemed-slake-localization-surface/v1",
        "source_dataset": "BoKelvin/SLAKE",
        "source_version": SLAKE_REVISION,
        "source_split": source_split,
        "selection": {
            "seed": seed,
            "limit": limit,
            "max_per_image": max_per_image,
            "max_per_label": max_per_label,
            "question_language": "en",
            "matched_normalized_detection_labels": 1,
            "target_area_interval": [area_min, area_max],
            "vqa_answer_read": False,
        },
        "eligible_before_cap": len(candidates),
        "written": len(inference_rows),
        "unique_images": len(per_image),
        "target_labels": dict(sorted(Counter(row["target_label"] for row in selected).items())),
        "target_area": {
            "minimum": min(row["region_area"] for row in selected),
            "maximum": max(row["region_area"] for row in selected),
            "mean": sum(row["region_area"] for row in selected) / len(selected),
        },
        "rejected": dict(sorted(rejected.items())),
        "source_hashes": {
            "source_json_sha256": sha256_file(json_path),
            "inference_manifest_sha256": sha256_file(inference_output),
            "targets_sha256": sha256_file(targets_output),
        },
        "leakage_boundary": {
            "selection_fields": ["q_lang", "img_name", "question", "detection.json"],
            "inference_has_target_fields": False,
            "targets_mode": "0600",
        },
    }
    write_json(report_path, report)
    return report


def build_slake_oracle_crop_answer_surface(
    locator_inference_path: Path,
    locator_targets_path: Path,
    source_json_path: Path,
    image_root: Path,
    output_root: Path,
    full_output: Path,
    crop_output: Path,
    black_output: Path,
    references_output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Materialize full/oracle/black answer arms from a frozen locator selection."""
    inference_rows = read_jsonl(locator_inference_path)
    reject_reference_fields(inference_rows)
    target_rows = read_jsonl(locator_targets_path)
    targets = {str(row["sample_id"]): row for row in target_rows}
    if len(targets) != len(target_rows):
        raise ValueError("Duplicate locator target sample ids")
    if {str(row["sample_id"]) for row in inference_rows} != set(targets):
        raise ValueError("Locator inference and target sample ids differ")
    raw_source = json.loads(source_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_source, list):
        raise TypeError("SLAKE answer source must be a JSON array")
    source_by_id = {str(row.get("qid")): row for row in raw_source if isinstance(row, dict)}
    if len(source_by_id) != len(raw_source):
        raise ValueError("SLAKE source contains missing or duplicate qid")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "full").mkdir(exist_ok=True)
    (output_root / "crops").mkdir(exist_ok=True)
    (output_root / "black").mkdir(exist_ok=True)
    full_rows = []
    crop_rows = []
    black_rows = []
    references = []
    artifact_hashes = {}
    for row in inference_rows:
        sample_id = str(row["sample_id"])
        source_id = str(row.get("source_record_id"))
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"Missing SLAKE source record: {source_id}")
        if source.get("question") != row.get("question") or source.get("img_name") != row.get(
            "image_path"
        ):
            raise ValueError(f"Frozen locator/source mismatch: {sample_id}")
        answer = str(source.get("answer") or "").strip()
        if not answer:
            raise ValueError(f"Missing SLAKE answer: {sample_id}")
        target = targets[sample_id]
        box = target.get("region_xyxy_1000")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, int) for value in box)
        ):
            raise ValueError(f"Invalid oracle box: {sample_id}")
        source_image_path = image_root / str(row["image_path"])
        if not source_image_path.is_file() or sha256_file(source_image_path) != row.get(
            "image_sha256"
        ):
            raise ValueError(f"Missing or changed oracle source image: {sample_id}")
        with Image.open(source_image_path) as source_image:
            image = source_image.convert("RGB")
        x1 = max(0, min(image.width - 1, math.floor(box[0] * image.width / 1000)))
        y1 = max(0, min(image.height - 1, math.floor(box[1] * image.height / 1000)))
        x2 = max(x1 + 1, min(image.width, math.ceil(box[2] * image.width / 1000)))
        y2 = max(y1 + 1, min(image.height, math.ceil(box[3] * image.height / 1000)))
        crop = image.crop((x1, y1, x2, y2))
        artifact_id = hashlib.sha256(sample_id.encode()).hexdigest()
        full_suffix = source_image_path.suffix.lower() or ".img"
        full_path = output_root / "full" / f"{artifact_id}{full_suffix}"
        crop_path = output_root / "crops" / f"{artifact_id}.png"
        black_path = output_root / "black" / f"{artifact_id}.png"
        shutil.copyfile(source_image_path, full_path)
        crop.save(crop_path, format="PNG", optimize=False)
        Image.new("RGB", crop.size, "black").save(black_path, format="PNG", optimize=False)
        crop_sha = sha256_file(crop_path)
        black_sha = sha256_file(black_path)
        full_sha = sha256_file(full_path)
        if full_sha != row["image_sha256"]:
            raise ValueError(f"Materialized full image hash mismatch: {sample_id}")
        artifact_hashes[f"full/{artifact_id}{full_suffix}"] = full_sha
        artifact_hashes[f"crops/{artifact_id}.png"] = crop_sha
        artifact_hashes[f"black/{artifact_id}.png"] = black_sha
        common = {
            "sample_id": sample_id,
            "kind": "open",
            "question": row["question"],
            "task": "slake-oracle-crop-answer",
            "source_dataset": row.get("source_dataset"),
            "source_version": row.get("source_version"),
            "source_record_id": source_id,
        }
        full_rows.append(
            {
                **common,
                "image_path": f"full/{artifact_id}{full_suffix}",
                "image_sha256": full_sha,
                "visual_arm": "full-image",
            }
        )
        crop_rows.append(
            {
                **common,
                "image_path": f"crops/{artifact_id}.png",
                "image_sha256": crop_sha,
                "visual_arm": "oracle-crop",
            }
        )
        black_rows.append(
            {
                **common,
                "image_path": f"black/{artifact_id}.png",
                "image_sha256": black_sha,
                "visual_arm": "black-crop",
            }
        )
        references.append({"sample_id": sample_id, "answer": answer})
    for rows in (full_rows, crop_rows, black_rows):
        reject_reference_fields(rows)
    _write_jsonl(full_output, full_rows)
    _write_jsonl(crop_output, crop_rows)
    _write_jsonl(black_output, black_rows)
    _write_jsonl(references_output, references, mode=0o600)
    report = {
        "schema_version": "edgemed-slake-oracle-crop-answer-surface/v1",
        "written_per_arm": len(inference_rows),
        "selection": {
            "frozen_locator_inference_order": True,
            "vqa_answer_used_for_selection": False,
            "oracle_box_used_only_for_crop_materialization": True,
        },
        "arms": ["full-image", "oracle-crop", "black-crop"],
        "source_hashes": {
            "locator_inference_sha256": sha256_file(locator_inference_path),
            "locator_targets_sha256": sha256_file(locator_targets_path),
            "source_json_sha256": sha256_file(source_json_path),
            "full_inference_sha256": sha256_file(full_output),
            "crop_inference_sha256": sha256_file(crop_output),
            "black_inference_sha256": sha256_file(black_output),
            "references_sha256": sha256_file(references_output),
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "leakage_boundary": {
            "inference_has_answer_fields": False,
            "references_mode": "0600",
            "references_materialized_after_frozen_selection": True,
        },
    }
    write_json(report_path, report)
    return report


def build_slake_multiview_answer_surface(
    full_path: Path,
    crop_path: Path,
    black_path: Path,
    oracle_multiview_output: Path,
    black_multiview_output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Combine answer-isolated full/crop arms without reading references."""
    full_rows = read_jsonl(full_path)
    crop_rows = read_jsonl(crop_path)
    black_rows = read_jsonl(black_path)
    for rows in (full_rows, crop_rows, black_rows):
        reject_reference_fields(rows)
    full_by_id = {str(row["sample_id"]): row for row in full_rows}
    crop_by_id = {str(row["sample_id"]): row for row in crop_rows}
    black_by_id = {str(row["sample_id"]): row for row in black_rows}
    expected_ids = [str(row["sample_id"]) for row in full_rows]
    if (
        len(full_by_id) != len(full_rows)
        or len(crop_by_id) != len(crop_rows)
        or len(black_by_id) != len(black_rows)
        or set(expected_ids) != set(crop_by_id)
        or set(expected_ids) != set(black_by_id)
    ):
        raise ValueError("Full/crop/black manifests require identical unique sample ids")

    oracle_rows = []
    black_control_rows = []
    for sample_id in expected_ids:
        full = full_by_id[sample_id]
        crop = crop_by_id[sample_id]
        black = black_by_id[sample_id]
        stable_fields = ("question", "kind", "source_record_id")
        if any(full.get(field) != crop.get(field) or full.get(field) != black.get(field) for field in stable_fields):
            raise ValueError(f"Arm metadata mismatch: {sample_id}")
        common = {
            key: value
            for key, value in full.items()
            if key not in {"image_path", "image_sha256", "visual_arm"}
        }
        full_spec = {
            "role": "full_context",
            "image_path": full["image_path"],
            "image_sha256": full["image_sha256"],
        }
        oracle_rows.append(
            {
                **common,
                "images": [
                    full_spec,
                    {
                        "role": "local_detail",
                        "image_path": crop["image_path"],
                        "image_sha256": crop["image_sha256"],
                    },
                ],
                "visual_arm": "full-plus-oracle-crop",
            }
        )
        black_control_rows.append(
            {
                **common,
                "images": [
                    full_spec,
                    {
                        "role": "local_detail",
                        "image_path": black["image_path"],
                        "image_sha256": black["image_sha256"],
                    },
                ],
                "visual_arm": "full-plus-black-crop",
            }
        )
    reject_reference_fields(oracle_rows)
    reject_reference_fields(black_control_rows)
    _write_jsonl(oracle_multiview_output, oracle_rows)
    _write_jsonl(black_multiview_output, black_control_rows)
    report = {
        "schema_version": "edgemed-slake-multiview-answer-surface/v1",
        "written_per_arm": len(expected_ids),
        "arms": ["full-plus-oracle-crop", "full-plus-black-crop"],
        "input_layout": "labeled-multi-image-v1",
        "source_hashes": {
            "full_inference_sha256": sha256_file(full_path),
            "crop_inference_sha256": sha256_file(crop_path),
            "black_inference_sha256": sha256_file(black_path),
            "oracle_multiview_sha256": sha256_file(oracle_multiview_output),
            "black_multiview_sha256": sha256_file(black_multiview_output),
        },
        "leakage_boundary": {
            "references_read": False,
            "inference_has_answer_fields": False,
        },
    }
    write_json(report_path, report)
    return report


def build_slake_learned_crop_multiview_surface(
    full_path: Path,
    locator_predictions_path: Path,
    locator_run_manifest_path: Path,
    surface_root: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Materialize learned crops from a completed, hash-bound locator run."""
    full_rows = read_jsonl(full_path)
    reject_reference_fields(full_rows)
    predictions = read_jsonl(locator_predictions_path)
    reject_reference_fields(predictions)
    run_manifest = json.loads(locator_run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("status") != "completed":
        raise ValueError("Locator run is not completed")
    predictions_sha = sha256_file(locator_predictions_path)
    if run_manifest.get("predictions_sha256") != predictions_sha:
        raise ValueError("Locator predictions do not match the completed run manifest")
    prediction_by_id = {str(row["sample_id"]): row for row in predictions}
    expected_ids = [str(row["sample_id"]) for row in full_rows]
    if (
        len(set(expected_ids)) != len(expected_ids)
        or len(prediction_by_id) != len(predictions)
        or set(expected_ids) != set(prediction_by_id)
    ):
        raise ValueError("Full manifest and locator predictions require identical unique sample ids")

    learned_root = surface_root / "learned-crops"
    learned_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    artifact_hashes = {}
    for full in full_rows:
        sample_id = str(full["sample_id"])
        prediction = prediction_by_id[sample_id]
        if prediction.get("status") != "completed":
            raise ValueError(f"Incomplete locator prediction: {sample_id}")
        tool_call = prediction.get("tool_call")
        arguments = tool_call.get("arguments") if isinstance(tool_call, dict) else None
        box = arguments.get("region_xyxy_1000") if isinstance(arguments, dict) else None
        if (
            not isinstance(tool_call, dict)
            or tool_call.get("name") != "region_inspect"
            or not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, int) or value < 0 or value > 1000 for value in box)
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise ValueError(f"Invalid learned locator box: {sample_id}")
        source_image_path = surface_root / str(full["image_path"])
        if not source_image_path.is_file() or sha256_file(source_image_path) != full.get(
            "image_sha256"
        ):
            raise ValueError(f"Missing or changed full image: {sample_id}")
        with Image.open(source_image_path) as source:
            image = source.convert("RGB")
        x1 = max(0, min(image.width - 1, math.floor(box[0] * image.width / 1000)))
        y1 = max(0, min(image.height - 1, math.floor(box[1] * image.height / 1000)))
        x2 = max(x1 + 1, min(image.width, math.ceil(box[2] * image.width / 1000)))
        y2 = max(y1 + 1, min(image.height, math.ceil(box[3] * image.height / 1000)))
        crop = image.crop((x1, y1, x2, y2))
        artifact_id = hashlib.sha256(sample_id.encode()).hexdigest()
        learned_path = learned_root / f"{artifact_id}.png"
        crop.save(learned_path, format="PNG", optimize=False)
        learned_sha = sha256_file(learned_path)
        relative_learned_path = f"learned-crops/{artifact_id}.png"
        artifact_hashes[relative_learned_path] = learned_sha
        common = {
            key: value
            for key, value in full.items()
            if key not in {"image_path", "image_sha256", "visual_arm"}
        }
        output_rows.append(
            {
                **common,
                "images": [
                    {
                        "role": "full_context",
                        "image_path": full["image_path"],
                        "image_sha256": full["image_sha256"],
                    },
                    {
                        "role": "local_detail",
                        "image_path": relative_learned_path,
                        "image_sha256": learned_sha,
                    },
                ],
                "visual_arm": "full-plus-learned-crop",
            }
        )
    reject_reference_fields(output_rows)
    _write_jsonl(output_path, output_rows)
    report = {
        "schema_version": "edgemed-slake-learned-crop-multiview-surface/v1",
        "written": len(output_rows),
        "input_layout": "labeled-multi-image-v1",
        "source_hashes": {
            "full_inference_sha256": sha256_file(full_path),
            "locator_predictions_sha256": predictions_sha,
            "locator_run_manifest_sha256": sha256_file(locator_run_manifest_path),
            "output_sha256": sha256_file(output_path),
        },
        "locator_binding": {
            "run_id": run_manifest.get("run_id"),
            "contract_sha256": run_manifest.get("contract_sha256"),
            "code_commit": run_manifest.get("code_commit"),
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "leakage_boundary": {
            "references_read": False,
            "ground_truth_boxes_read": False,
            "inference_has_answer_fields": False,
        },
    }
    write_json(report_path, report)
    return report


def build_slake_oracle_pointer_surface(
    full_path: Path,
    locator_targets_path: Path,
    surface_root: Path,
    pointer_output: Path,
    sham_output: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Draw correct or deterministically permuted boxes on otherwise identical full images."""
    full_rows = read_jsonl(full_path)
    reject_reference_fields(full_rows)
    targets = read_jsonl(locator_targets_path)
    target_by_id = {str(row["sample_id"]): row for row in targets}
    sample_ids = [str(row["sample_id"]) for row in full_rows]
    if (
        len(sample_ids) < 2
        or len(set(sample_ids)) != len(sample_ids)
        or len(target_by_id) != len(targets)
        or set(sample_ids) != set(target_by_id)
    ):
        raise ValueError("Pointer surface requires identical unique full/target ids")
    sorted_ids = sorted(sample_ids)
    sham_source = {
        sample_id: sorted_ids[(index + 1) % len(sorted_ids)]
        for index, sample_id in enumerate(sorted_ids)
    }
    pointer_root = surface_root / "oracle-pointer"
    sham_root = surface_root / "sham-pointer"
    pointer_root.mkdir(parents=True, exist_ok=True)
    sham_root.mkdir(parents=True, exist_ok=True)
    pointer_rows = []
    sham_rows = []
    artifact_hashes = {}

    def draw_pointer(image: Image.Image, box: list[int]) -> Image.Image:
        pointed = image.copy()
        xyxy = (
            math.floor(box[0] * image.width / 1000),
            math.floor(box[1] * image.height / 1000),
            max(0, math.ceil(box[2] * image.width / 1000) - 1),
            max(0, math.ceil(box[3] * image.height / 1000) - 1),
        )
        width = max(2, round(min(image.size) * 0.008))
        ImageDraw.Draw(pointed).rectangle(xyxy, outline=(255, 0, 0), width=width)
        return pointed

    for full in full_rows:
        sample_id = str(full["sample_id"])
        true_box = target_by_id[sample_id].get("region_xyxy_1000")
        sham_box = target_by_id[sham_source[sample_id]].get("region_xyxy_1000")
        for box, label in ((true_box, "oracle"), (sham_box, "sham")):
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(not isinstance(value, int) or value < 0 or value > 1000 for value in box)
                or box[0] >= box[2]
                or box[1] >= box[3]
            ):
                raise ValueError(f"Invalid {label} pointer box: {sample_id}")
        source_path = surface_root / str(full["image_path"])
        if not source_path.is_file() or sha256_file(source_path) != full.get("image_sha256"):
            raise ValueError(f"Missing or changed pointer source image: {sample_id}")
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        artifact_id = hashlib.sha256(sample_id.encode()).hexdigest()
        pointer_path = pointer_root / f"{artifact_id}.png"
        sham_path = sham_root / f"{artifact_id}.png"
        draw_pointer(image, true_box).save(pointer_path, format="PNG", optimize=False)
        draw_pointer(image, sham_box).save(sham_path, format="PNG", optimize=False)
        pointer_sha = sha256_file(pointer_path)
        sham_sha = sha256_file(sham_path)
        pointer_relative = f"oracle-pointer/{artifact_id}.png"
        sham_relative = f"sham-pointer/{artifact_id}.png"
        artifact_hashes[pointer_relative] = pointer_sha
        artifact_hashes[sham_relative] = sham_sha
        common = {
            key: value
            for key, value in full.items()
            if key not in {"image_path", "image_sha256", "visual_arm"}
        }
        pointer_rows.append(
            {
                **common,
                "image_path": pointer_relative,
                "image_sha256": pointer_sha,
                "visual_arm": "full-with-oracle-pointer",
            }
        )
        sham_rows.append(
            {
                **common,
                "image_path": sham_relative,
                "image_sha256": sham_sha,
                "visual_arm": "full-with-permuted-pointer",
            }
        )
    reject_reference_fields(pointer_rows)
    reject_reference_fields(sham_rows)
    _write_jsonl(pointer_output, pointer_rows)
    _write_jsonl(sham_output, sham_rows)
    report = {
        "schema_version": "edgemed-slake-oracle-pointer-surface/v1",
        "written_per_arm": len(sample_ids),
        "arms": ["full-with-oracle-pointer", "full-with-permuted-pointer"],
        "pointer_style": {"color_rgb": [255, 0, 0], "width_fraction": 0.008, "min_width": 2},
        "sham_assignment": "sorted-sample-id-rotate-one",
        "source_hashes": {
            "full_inference_sha256": sha256_file(full_path),
            "locator_targets_sha256": sha256_file(locator_targets_path),
            "pointer_inference_sha256": sha256_file(pointer_output),
            "sham_inference_sha256": sha256_file(sham_output),
        },
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "leakage_boundary": {
            "references_read": False,
            "answers_read": False,
            "oracle_boxes_used_only_for_pointer_materialization": True,
        },
    }
    write_json(report_path, report)
    return report


def quarantine_gate_candidates(
    manifest_path: Path,
    gate_report_path: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(manifest_path)
    gate = json.loads(gate_report_path.read_text())
    candidate_ids = {
        item["record_id"]
        for item in gate.get("near_image_candidates_rejected_by_confirmation", [])
    }
    known_ids = {row["record_id"] for row in rows}
    if not candidate_ids.issubset(known_ids):
        raise ValueError("Gate report contains records outside the manifest")
    quarantined = 0
    for row in rows:
        if row["record_id"] in candidate_ids:
            row["quality_status"] = "quarantined"
            row["benchmark_overlap"] = "suspected"
            row["quarantine_reason"] = "unreviewed_near_image_candidate"
            quarantined += 1
    _write_jsonl(output, rows)
    report = {
        "schema_version": "edgemed-external-quarantine/v1",
        "rows": len(rows),
        "accepted": len(rows) - quarantined,
        "quarantined": quarantined,
        "source_hashes": {
            "input_manifest_sha256": sha256_file(manifest_path),
            "input_gate_report_sha256": sha256_file(gate_report_path),
            "output_manifest_sha256": sha256_file(output),
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="source", required=True)
    pmc = subparsers.add_parser("pmc-vqa")
    pmc.add_argument("--csv", type=Path, required=True)
    pmc.add_argument("--license-csv", type=Path, required=True)
    pmc.add_argument("--image-root", type=Path, required=True)
    pmc.add_argument("--output", type=Path, required=True)
    pmc.add_argument("--report", type=Path, required=True)
    pmc.add_argument("--limit", type=int, default=2000)
    pmc.add_argument("--seed", default="edgemed-pmc-vqa-v2-seed-20260901")
    pmc.add_argument("--max-per-image", type=int, default=1)
    pmc.add_argument("--required-split", choices=("train", "test"), default="train")
    pmc.add_argument("--cohort", default="train-seed")
    pmc.add_argument("--exclude-manifest", type=Path)
    slake = subparsers.add_parser("slake")
    slake.add_argument("--json", type=Path, required=True)
    slake.add_argument("--image-root", type=Path, required=True)
    slake.add_argument("--output", type=Path, required=True)
    slake.add_argument("--report", type=Path, required=True)
    extract = subparsers.add_parser("extract-zip")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output-root", type=Path, required=True)
    extract.add_argument("--expected-sha256", required=True)
    extract.add_argument("--report", type=Path, required=True)
    surfaces = subparsers.add_parser("split-surfaces")
    surfaces.add_argument("--manifest", type=Path, required=True)
    surfaces.add_argument("--kind", choices=("mcq", "open"), required=True)
    surfaces.add_argument("--inference-output", type=Path, required=True)
    surfaces.add_argument("--references-output", type=Path, required=True)
    surfaces.add_argument("--report", type=Path, required=True)
    quarantine = subparsers.add_parser("quarantine-candidates")
    quarantine.add_argument("--manifest", type=Path, required=True)
    quarantine.add_argument("--gate-report", type=Path, required=True)
    quarantine.add_argument("--output", type=Path, required=True)
    quarantine.add_argument("--report", type=Path, required=True)
    slake_binary = subparsers.add_parser("slake-binary-surface")
    slake_binary.add_argument("--inference", type=Path, required=True)
    slake_binary.add_argument("--references", type=Path, required=True)
    slake_binary.add_argument("--inference-output", type=Path, required=True)
    slake_binary.add_argument("--references-output", type=Path, required=True)
    slake_binary.add_argument("--report", type=Path, required=True)
    slake_binary.add_argument("--limit", type=int, default=96)
    slake_binary.add_argument("--max-per-image", type=int, default=1)
    slake_binary.add_argument("--seed", default="edgemed-slake-binary-v1")
    slake_localization = subparsers.add_parser("slake-localization-surface")
    slake_localization.add_argument("--json", type=Path, required=True)
    slake_localization.add_argument("--image-root", type=Path, required=True)
    slake_localization.add_argument("--inference-output", type=Path, required=True)
    slake_localization.add_argument("--targets-output", type=Path, required=True)
    slake_localization.add_argument("--report", type=Path, required=True)
    slake_localization.add_argument(
        "--source-split", choices=("train", "validation", "test"), required=True
    )
    slake_localization.add_argument("--limit", type=int, default=0)
    slake_localization.add_argument("--max-per-image", type=int, default=2)
    slake_localization.add_argument("--max-per-label", type=int, default=0)
    slake_localization.add_argument("--seed", default="edgemed-slake-localization-v1")
    slake_localization.add_argument("--area-min", type=float, default=0.01)
    slake_localization.add_argument("--area-max", type=float, default=0.64)
    oracle_crop = subparsers.add_parser("slake-oracle-crop-answer-surface")
    oracle_crop.add_argument("--locator-inference", type=Path, required=True)
    oracle_crop.add_argument("--locator-targets", type=Path, required=True)
    oracle_crop.add_argument("--source-json", type=Path, required=True)
    oracle_crop.add_argument("--image-root", type=Path, required=True)
    oracle_crop.add_argument("--output-root", type=Path, required=True)
    oracle_crop.add_argument("--full-output", type=Path, required=True)
    oracle_crop.add_argument("--crop-output", type=Path, required=True)
    oracle_crop.add_argument("--black-output", type=Path, required=True)
    oracle_crop.add_argument("--references-output", type=Path, required=True)
    oracle_crop.add_argument("--report", type=Path, required=True)
    multiview = subparsers.add_parser("slake-multiview-answer-surface")
    multiview.add_argument("--full", type=Path, required=True)
    multiview.add_argument("--crop", type=Path, required=True)
    multiview.add_argument("--black", type=Path, required=True)
    multiview.add_argument("--oracle-multiview-output", type=Path, required=True)
    multiview.add_argument("--black-multiview-output", type=Path, required=True)
    multiview.add_argument("--report", type=Path, required=True)
    learned_multiview = subparsers.add_parser("slake-learned-crop-multiview-surface")
    learned_multiview.add_argument("--full", type=Path, required=True)
    learned_multiview.add_argument("--locator-predictions", type=Path, required=True)
    learned_multiview.add_argument("--locator-run-manifest", type=Path, required=True)
    learned_multiview.add_argument("--surface-root", type=Path, required=True)
    learned_multiview.add_argument("--output", type=Path, required=True)
    learned_multiview.add_argument("--report", type=Path, required=True)
    oracle_pointer = subparsers.add_parser("slake-oracle-pointer-surface")
    oracle_pointer.add_argument("--full", type=Path, required=True)
    oracle_pointer.add_argument("--locator-targets", type=Path, required=True)
    oracle_pointer.add_argument("--surface-root", type=Path, required=True)
    oracle_pointer.add_argument("--pointer-output", type=Path, required=True)
    oracle_pointer.add_argument("--sham-output", type=Path, required=True)
    oracle_pointer.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.source == "pmc-vqa":
        report = build_pmc_vqa(
            args.csv,
            args.license_csv,
            args.image_root,
            args.output,
            args.report,
            args.limit,
            args.seed,
            args.max_per_image,
            args.required_split,
            args.cohort,
            args.exclude_manifest,
        )
    elif args.source == "slake":
        report = build_slake(args.json, args.image_root, args.output, args.report)
    elif args.source == "extract-zip":
        report = extract_zip_safe(args.archive, args.output_root, args.expected_sha256)
        write_json(args.report, report)
    elif args.source == "split-surfaces":
        report = split_surfaces(
            args.manifest,
            args.kind,
            args.inference_output,
            args.references_output,
            args.report,
        )
    elif args.source == "slake-binary-surface":
        report = build_slake_binary_surface(
            args.inference,
            args.references,
            args.inference_output,
            args.references_output,
            args.report,
            limit=args.limit,
            max_per_image=args.max_per_image,
            seed=args.seed,
        )
    elif args.source == "slake-localization-surface":
        report = build_slake_localization_surface(
            args.json,
            args.image_root,
            args.inference_output,
            args.targets_output,
            args.report,
            source_split=args.source_split,
            limit=args.limit,
            max_per_image=args.max_per_image,
            max_per_label=args.max_per_label,
            seed=args.seed,
            area_min=args.area_min,
            area_max=args.area_max,
        )
    elif args.source == "slake-oracle-crop-answer-surface":
        report = build_slake_oracle_crop_answer_surface(
            args.locator_inference,
            args.locator_targets,
            args.source_json,
            args.image_root,
            args.output_root,
            args.full_output,
            args.crop_output,
            args.black_output,
            args.references_output,
            args.report,
        )
    elif args.source == "slake-multiview-answer-surface":
        report = build_slake_multiview_answer_surface(
            args.full,
            args.crop,
            args.black,
            args.oracle_multiview_output,
            args.black_multiview_output,
            args.report,
        )
    elif args.source == "slake-learned-crop-multiview-surface":
        report = build_slake_learned_crop_multiview_surface(
            args.full,
            args.locator_predictions,
            args.locator_run_manifest,
            args.surface_root,
            args.output,
            args.report,
        )
    elif args.source == "slake-oracle-pointer-surface":
        report = build_slake_oracle_pointer_surface(
            args.full,
            args.locator_targets,
            args.surface_root,
            args.pointer_output,
            args.sham_output,
            args.report,
        )
    else:
        report = quarantine_gate_candidates(
            args.manifest,
            args.gate_report,
            args.output,
            args.report,
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
