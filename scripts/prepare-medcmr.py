#!/usr/bin/env python3
"""Verify, extract, and split the official Med-CMR release into inference/reference surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

DATASET_REVISION = "a9b2d6e610c6c5dcf4f3e5aa89c7ec9fd7a05b73"
ARCHIVE_SHA256 = "8495f3b6b2901a095918ab27c600e42da7287a87f6aea490912ba60d27de3775"
MCQ_JSON = "dataset/cmed.copy.mcq_cate.cleaned.json"
OPEN_JSON = "dataset/cmed.copy.open_taged.cleaned.json"
EXPECTED_MCQS = 16_655
RELEASED_OPEN = 3_999
PAPER_OPEN = 3_998


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_sha256(path: Path) -> str:
    return sha256_file(path)


def safe_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in archive.infolist():
        posix = PurePosixPath(info.filename)
        if posix.is_absolute() or ".." in posix.parts:
            raise ValueError(f"Unsafe archive member: {info.filename}")
        if not posix.parts or posix.parts[0] != "dataset":
            continue
        if info.is_dir() or info.filename.endswith(".bak"):
            continue
        if info.filename in {MCQ_JSON, OPEN_JSON} or (
            len(posix.parts) == 3
            and posix.parts[:2] == ("dataset", "imgs")
            and posix.suffix.lower() == ".png"
        ):
            yield info


def extract_release(archive_path: Path, raw_root: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in safe_members(archive):
            target = (raw_root / info.filename).resolve()
            if not target.is_relative_to(raw_root.resolve()):
                raise ValueError(f"Archive target escaped raw root: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            extracted.append(info.filename)
    return extracted


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: int = 0o644) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    path.chmod(mode)
    return count


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise TypeError(f"Expected a list of objects in {path}")
    return data


def build_surfaces(
    records: list[dict[str, Any]],
    kind: str,
    raw_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected = EXPECTED_MCQS if kind == "mcq" else RELEASED_OPEN
    if len(records) != expected:
        raise ValueError(f"Unexpected {kind} count: {len(records)} != {expected}")

    required = {
        "index",
        "question",
        "image_path",
        "answer",
        "visual_description",
        "task",
        "subtype",
        "question type",
        "modality",
        "organ system",
    }
    if kind == "mcq":
        required.update("ABCDE")

    inference_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    missing_images: list[str] = []
    seen_ids: set[str] = set()
    image_hashes: set[str] = set()

    for record in records:
        missing_fields = sorted(required - record.keys())
        if missing_fields:
            raise ValueError(f"{kind} index {record.get('index')} missing {missing_fields}")
        index = int(record["index"])
        sample_id = f"{kind}-{index}"
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)

        relative_image = PurePosixPath(str(record["image_path"]))
        image_path = raw_root.joinpath(*relative_image.parts)
        if not image_path.is_file():
            missing_images.append(str(relative_image))
            image_sha256 = None
        else:
            image_sha256 = sha256_file(image_path)
            image_hashes.add(image_sha256)

        inference: dict[str, Any] = {
            "sample_id": sample_id,
            "index": index,
            "kind": kind,
            "question": record["question"],
            "image_path": str(relative_image),
            "image_sha256": image_sha256,
            "task": record["task"],
            "subtype": record["subtype"],
            "question_type": record["question type"],
            "modality": record["modality"],
            "organ_system": record["organ system"],
        }
        if kind == "mcq":
            inference["options"] = {letter: record[letter] for letter in "ABCDE"}
        inference_rows.append(inference)

        reference: dict[str, Any] = {
            "sample_id": sample_id,
            "answer": record["answer"],
        }
        if kind == "open":
            reference["visual_description"] = record["visual_description"]
        reference_rows.append(reference)

    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images; first={missing_images[:3]}")

    stats = {
        "count": len(records),
        "index_min": min(int(row["index"]) for row in records),
        "index_max": max(int(row["index"]) for row in records),
        "unique_sample_ids": len(seen_ids),
        "unique_image_hashes": len(image_hashes),
        "task_counts": dict(sorted(Counter(str(row["task"]) for row in records).items())),
    }
    return inference_rows, reference_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    snapshot_dir = args.snapshot_dir.resolve()
    output_root = args.output_root.resolve()
    archive_path = snapshot_dir / "dataset.zip"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    actual_archive_sha = sha256_file(archive_path)
    if actual_archive_sha != ARCHIVE_SHA256:
        raise ValueError(f"Archive SHA-256 mismatch: {actual_archive_sha}")

    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    extracted = extract_release(archive_path, raw_root)

    mcq_records = load_json(raw_root / MCQ_JSON)
    open_records = load_json(raw_root / OPEN_JSON)
    mcq_inputs, mcq_refs, mcq_stats = build_surfaces(mcq_records, "mcq", raw_root)
    open_inputs, open_refs, open_stats = build_surfaces(open_records, "open", raw_root)

    manifests_dir = output_root / "manifests"
    references_dir = output_root / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    references_dir.chmod(0o700)

    # Keep the original answer-bearing JSON outside the runner's data root.
    source_dir = references_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_dir.chmod(0o700)
    protected_sources: dict[str, Path] = {}
    for name, relative_path in (("mcq_source", MCQ_JSON), ("open_source", OPEN_JSON)):
        source_path = raw_root / relative_path
        protected_path = source_dir / Path(relative_path).name
        shutil.move(source_path, protected_path)
        protected_path.chmod(0o600)
        protected_sources[name] = protected_path

    paths = {
        "mcq_manifest": manifests_dir / "mcq.jsonl",
        "open_manifest": manifests_dir / "open.jsonl",
        "mcq_references": references_dir / "mcq.jsonl",
        "open_references": references_dir / "open.jsonl",
    }
    write_jsonl(paths["mcq_manifest"], mcq_inputs)
    write_jsonl(paths["open_manifest"], open_inputs)
    write_jsonl(paths["mcq_references"], mcq_refs, mode=0o600)
    write_jsonl(paths["open_references"], open_refs, mode=0o600)

    manifest = {
        "schema_version": "medcmr-dataset-manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo_id": "aaassddaadf/Med-CMR",
            "revision": DATASET_REVISION,
            "archive_filename": "dataset.zip",
            "archive_size": archive_path.stat().st_size,
            "archive_sha256": actual_archive_sha,
            "license_declared": "apache-2.0",
        },
        "release_contract": {
            "mcq_released": EXPECTED_MCQS,
            "open_released": RELEASED_OPEN,
            "open_paper_reported": PAPER_OPEN,
            "open_count_mismatch": RELEASED_OPEN - PAPER_OPEN,
            "split_field_present": False,
            "usage": "zero-shot public benchmark test only",
        },
        "extraction": {
            "extracted_files": len(extracted),
            "png_files": sum(name.endswith(".png") for name in extracted),
            "excluded_macosx_and_backups": True,
        },
        "stats": {"mcq": mcq_stats, "open": open_stats},
        "surface_hashes": {
            **{name: jsonl_sha256(path) for name, path in paths.items()},
            **{name: sha256_file(path) for name, path in protected_sources.items()},
        },
        "leakage_boundary": {
            "runner_allowed": ["manifests/mcq.jsonl", "manifests/open.jsonl", "raw/dataset/imgs"],
            "runner_forbidden": ["references", "answer", "visual_description"],
            "answer_bearing_json_removed_from_raw_root": True,
        },
    }
    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
