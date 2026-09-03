"""Build source-pinned external medical VQA manifests without benchmark labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

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
