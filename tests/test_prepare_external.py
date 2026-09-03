import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

from edgemed_bench.io import read_jsonl
from edgemed_bench.prepare_external import (
    build_slake_localization_surface,
    build_pmc_vqa,
    build_slake,
    build_slake_binary_surface,
    extract_zip_safe,
    quarantine_gate_candidates,
    split_surfaces,
)


def make_image(path: Path, color: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_pmc_vqa_is_license_filtered_and_deterministic(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    sha1 = make_image(image_root / "PMC1_F1.jpg", "red")
    make_image(image_root / "PMC2_F1.jpg", "blue")
    csv_path = tmp_path / "train.csv"
    fields = [
        "index", "Figure_path", "Caption", "Question", "Choice A", "Choice B",
        "Choice C", "Choice D", "Answer", "split",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"index": "10", "Figure_path": "PMC1_F1.jpg", "Caption": "cap", "Question": "Q1?", "Choice A": "A: yes", "Choice B": "B: no", "Choice C": "C: x", "Choice D": "D: y", "Answer": "A", "split": "train"},
                {"index": "11", "Figure_path": "PMC1_F1.jpg", "Caption": "cap", "Question": "Q2?", "Choice A": "yes", "Choice B": "no", "Choice C": "x", "Choice D": "y", "Answer": "B", "split": "train"},
                {"index": "12", "Figure_path": "PMC2_F1.jpg", "Caption": "cap", "Question": "Q3?", "Choice A": "yes", "Choice B": "no", "Choice C": "x", "Choice D": "y", "Answer": "A", "split": "train"},
            ]
        )
    license_path = tmp_path / "licenses.csv"
    license_path.write_text("Accession ID,License\nPMC1,CC BY\nPMC2,CC BY-ND\n")
    output = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "report.json"
    report = build_pmc_vqa(csv_path, license_path, image_root, output, report_path, 5, "seed")
    rows = read_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["image_sha256"] == sha1
    assert rows[0]["answer_text"] == "yes"
    assert rows[0]["options"]["A"] == "yes"
    assert rows[0]["annotation_type"] == "synthetic"
    assert rows[0]["evidence_target_eligible"] is False
    assert report["rejected"] == {"license_or_pmcid": 1, "per_image_cap": 1}
    assert report["selection"]["required_split"] == "train"


def test_build_pmc_vqa_can_exclude_questions_from_another_manifest(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    make_image(image_root / "PMC1_F1.jpg", "red")
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "index,Figure_path,Caption,Question,Choice A,Choice B,Choice C,Choice D,Answer,split\n"
        "1,PMC1_F1.jpg,cap,What finding is visible?,one,two,three,four,A,test\n"
    )
    license_path = tmp_path / "licenses.csv"
    license_path.write_text("Accession ID,License\nPMC1,CC BY\n")
    exclusion = tmp_path / "train.jsonl"
    exclusion.write_text(json.dumps({"question": "What finding is visible"}) + "\n")
    report = build_pmc_vqa(
        csv_path,
        license_path,
        image_root,
        tmp_path / "manifest.jsonl",
        tmp_path / "report.json",
        1,
        "seed",
        required_split="test",
        cohort="dev",
        exclude_manifest=exclusion,
    )
    assert report["written"] == 0
    assert report["rejected"]["excluded_question_overlap"] == 1


def test_build_slake_keeps_english_validation_and_image_groups(tmp_path: Path) -> None:
    image_root = tmp_path / "imgs"
    image_sha = make_image(image_root / "xmlab0" / "source.jpg", "green")
    source = [
        {"img_id": 0, "img_name": "xmlab0/source.jpg", "question": "What modality?", "answer": "MRI", "q_lang": "en", "qid": 1, "content_type": "Modality"},
        {"img_id": 0, "img_name": "xmlab0/source.jpg", "question": "什么模态？", "answer": "MRI", "q_lang": "zh", "qid": 2},
    ]
    json_path = tmp_path / "validation.json"
    json_path.write_text(json.dumps(source))
    output = tmp_path / "manifest.jsonl"
    report_path = tmp_path / "report.json"
    report = build_slake(json_path, image_root, output, report_path)
    rows = read_jsonl(output)
    assert len(rows) == 1
    assert rows[0]["image_sha256"] == image_sha
    assert rows[0]["source_split"] == "validation"
    assert rows[0]["annotation_type"] == "human"
    assert report["rejected"] == {"not_english": 1}


def test_extract_zip_safe_checks_hash_and_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("imgs/example.txt", "ok")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = extract_zip_safe(archive, tmp_path / "out", digest)
    assert (tmp_path / "out" / "imgs" / "example.txt").read_text() == "ok"
    assert report["files"] == 1

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as handle:
        handle.writestr("../escape.txt", "bad")
    unsafe_digest = hashlib.sha256(unsafe.read_bytes()).hexdigest()
    try:
        extract_zip_safe(unsafe, tmp_path / "unsafe-out", unsafe_digest)
    except ValueError as error:
        assert "Unsafe archive member" in str(error)
    else:
        raise AssertionError("path traversal archive was accepted")


def test_split_surfaces_keeps_answers_out_of_inference(tmp_path: Path) -> None:
    manifest = tmp_path / "admitted.jsonl"
    row = {
        "record_id": "external-1",
        "source_dataset": "fixture",
        "source_version": "v1",
        "source_record_id": "1",
        "quality_status": "accepted",
        "question": "Question?",
        "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        "answer": "B",
        "answer_text": "two",
        "source_caption": "answer-bearing context",
        "slake_metadata": {"answer_type": "OPEN", "content_type": "Abnormality"},
        "image_path": "image.png",
        "image_sha256": "abc",
    }
    manifest.write_text(json.dumps(row) + "\n")
    inference = tmp_path / "inference.jsonl"
    references = tmp_path / "references.jsonl"
    report_path = tmp_path / "surfaces.json"
    report = split_surfaces(manifest, "mcq", inference, references, report_path)
    inference_row = read_jsonl(inference)[0]
    assert "answer" not in inference_row
    assert "source_caption" not in inference_row
    assert inference_row["evaluation_metadata"] == {
        "answer_type": "OPEN",
        "content_type": "Abnormality",
    }
    assert set(inference_row["options"]) == set("ABCD")
    assert read_jsonl(references) == [{"answer": "B", "sample_id": "external-1"}]
    assert oct(references.stat().st_mode & 0o777) == "0o600"
    assert report["leakage_boundary"]["inference_has_reference_fields"] is False


def test_quarantine_gate_candidates_preserves_but_excludes_records(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_id": record_id,
                    "quality_status": "accepted",
                    "benchmark_overlap": "none",
                }
            )
            for record_id in ("keep", "flag")
        )
        + "\n"
    )
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "near_image_candidates_rejected_by_confirmation": [
                    {"record_id": "flag", "benchmark_sample_id": "mcq-1"}
                ]
            }
        )
    )
    output = tmp_path / "admitted.jsonl"
    report = quarantine_gate_candidates(manifest, gate, output, tmp_path / "report.json")
    rows = {row["record_id"]: row for row in read_jsonl(output)}
    assert rows["keep"]["quality_status"] == "accepted"
    assert rows["flag"]["quality_status"] == "quarantined"
    assert rows["flag"]["benchmark_overlap"] == "suspected"
    assert report["accepted"] == 1
    assert report["quarantined"] == 1


def test_slake_binary_surface_selection_is_question_only_and_reference_isolated(
    tmp_path: Path,
) -> None:
    inference = tmp_path / "source-inference.jsonl"
    references = tmp_path / "source-references.jsonl"
    rows = [
        {
            "sample_id": "s1", "kind": "open", "question": "Is a lesion visible?",
            "image_path": "one.png", "image_sha256": "sha1",
            "evaluation_metadata": {"answer_type": "CLOSED", "modality": "CT"},
        },
        {
            "sample_id": "s2", "kind": "open", "question": "Is this T1 or T2?",
            "image_path": "two.png", "image_sha256": "sha2",
            "evaluation_metadata": {"answer_type": "CLOSED", "modality": "MRI"},
        },
        {
            "sample_id": "s3", "kind": "open", "question": "What is visible?",
            "image_path": "three.png", "image_sha256": "sha3",
            "evaluation_metadata": {"answer_type": "OPEN", "modality": "XR"},
        },
    ]
    inference.write_text("".join(json.dumps(row) + "\n" for row in rows))
    references.write_text(
        json.dumps({"sample_id": "s1", "answer": "yes"}) + "\n"
        + json.dumps({"sample_id": "s2", "answer": "t1"}) + "\n"
        + json.dumps({"sample_id": "s3", "answer": "finding"}) + "\n"
    )
    output = tmp_path / "inference.jsonl"
    output_references = tmp_path / "references.jsonl"
    report = build_slake_binary_surface(
        inference, references, output, output_references, tmp_path / "report.json",
        limit=1, max_per_image=1,
    )
    selected = read_jsonl(output)
    assert selected[0]["sample_id"] == "s1"
    assert selected[0]["options"] == {"A": "Yes", "B": "No"}
    assert selected[0]["modality"] == "CT"
    assert "answer" not in selected[0]
    assert read_jsonl(output_references) == [{"sample_id": "s1", "answer": "A"}]
    assert oct(output_references.stat().st_mode & 0o777) == "0o600"
    assert report["selection"]["answer_blind"] is True
    assert report["eligible_before_image_cap"] == 1


def test_slake_localization_surface_uses_detection_without_vqa_answer(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "imgs"
    image_path = image_root / "xmlab1" / "source.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 200), "gray").save(image_path)
    (image_path.parent / "detection.json").write_text(
        json.dumps([{"Liver": [10, 20, 40, 60]}])
    )
    source = tmp_path / "train.json"
    source.write_text(
        json.dumps(
            [
                {
                    "qid": 7,
                    "img_name": "xmlab1/source.jpg",
                    "question": "Does the liver look normal?",
                    "answer": "SECRET",
                    "q_lang": "en",
                },
                {
                    "qid": 8,
                    "img_name": "xmlab1/source.jpg",
                    "question": "What modality is shown?",
                    "answer": "ALSO_SECRET",
                    "q_lang": "en",
                },
            ]
        )
    )
    inference = tmp_path / "inference.jsonl"
    targets = tmp_path / "targets.jsonl"
    report = build_slake_localization_surface(
        source,
        image_root,
        inference,
        targets,
        tmp_path / "report.json",
        source_split="train",
        max_per_image=1,
    )
    inference_rows = read_jsonl(inference)
    target_rows = read_jsonl(targets)
    assert len(inference_rows) == len(target_rows) == 1
    assert inference_rows[0]["sample_id"] == "slake-train-locator-7"
    assert "answer" not in inference_rows[0]
    assert "region_xyxy_1000" not in inference_rows[0]
    assert target_rows[0]["target_label"] == "Liver"
    assert target_rows[0]["region_xyxy_1000"] == [100, 100, 500, 400]
    assert target_rows[0]["tool_call"]["arguments"]["media_id"] == "image-0"
    assert oct(targets.stat().st_mode & 0o777) == "0o600"
    assert report["selection"]["vqa_answer_read"] is False
    assert report["rejected"]["matched_label_count_not_one"] == 1
