import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image

from edgemed_bench.io import read_jsonl
from edgemed_bench.prepare_external import build_pmc_vqa, build_slake, extract_zip_safe


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
