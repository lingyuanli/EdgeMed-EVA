import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from edgemed_bench.validate_external_data import image_dhash, main, validate_external_data


def make_image(path: Path, pattern: str) -> str:
    image = Image.new("RGB", (32, 32), "white")
    draw = ImageDraw.Draw(image)
    if pattern == "left":
        draw.rectangle((0, 0, 12, 31), fill="black")
    elif pattern == "diagonal":
        draw.line((0, 0, 31, 31), fill="black", width=4)
    image.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_row(image_sha: str, question: str = "Which uncommon finding is visible?") -> dict:
    return {
        "record_id": "external-1",
        "source_dataset": "fixture",
        "source_version": "v1",
        "license": "CC-BY-4.0",
        "patient_group_hash": "patient-hash-1",
        "image_path": "external.png",
        "image_sha256": image_sha,
        "question": question,
        "answer": "A",
        "annotation_type": "human",
        "quality_status": "accepted",
        "benchmark_overlap": "none",
    }


def benchmark_row(image_sha: str, question: str = "What diagnosis is shown?") -> dict:
    return {
        "sample_id": "mcq-0",
        "image_path": "benchmark.png",
        "image_sha256": image_sha,
        "question": question,
    }


def test_clean_external_record_passes(tmp_path: Path) -> None:
    external_sha = make_image(tmp_path / "external.png", "diagonal")
    benchmark_sha = make_image(tmp_path / "benchmark.png", "left")
    report = validate_external_data(
        [external_row(external_sha)],
        tmp_path,
        [benchmark_row(benchmark_sha)],
        tmp_path,
    )
    assert report["status"] == "passed"
    assert report["accepted_checked"] == 1
    assert report["overlaps"] == []


def test_exact_image_overlap_fails(tmp_path: Path) -> None:
    image_sha = make_image(tmp_path / "external.png", "left")
    (tmp_path / "benchmark.png").write_bytes((tmp_path / "external.png").read_bytes())
    report = validate_external_data(
        [external_row(image_sha)],
        tmp_path,
        [benchmark_row(image_sha)],
        tmp_path,
    )
    assert report["status"] == "failed"
    assert {item["kind"] for item in report["overlaps"]} >= {"exact_image", "near_image"}


def test_dhash_candidate_needs_pixel_confirmation(tmp_path: Path) -> None:
    first = Image.new("L", (32, 32))
    second = Image.new("L", (32, 32))
    for y in range(32):
        for x in range(32):
            first.putpixel((x, y), x * 8)
            second.putpixel((x, y), int(255 * (x / 31) ** 4))
    first.save(tmp_path / "external.png")
    second.save(tmp_path / "benchmark.png")
    assert image_dhash(tmp_path / "external.png") == image_dhash(tmp_path / "benchmark.png")
    external_sha = hashlib.sha256((tmp_path / "external.png").read_bytes()).hexdigest()
    benchmark_sha = hashlib.sha256((tmp_path / "benchmark.png").read_bytes()).hexdigest()
    report = validate_external_data(
        [external_row(external_sha)],
        tmp_path,
        [benchmark_row(benchmark_sha)],
        tmp_path,
    )
    assert report["status"] == "passed"
    assert report["overlaps"] == []
    assert len(report["near_image_candidates_rejected_by_confirmation"]) == 1


def test_near_text_overlap_fails(tmp_path: Path) -> None:
    external_sha = make_image(tmp_path / "external.png", "diagonal")
    benchmark_sha = make_image(tmp_path / "benchmark.png", "left")
    report = validate_external_data(
        [external_row(external_sha, "What diagnosis is shown in this image")],
        tmp_path,
        [benchmark_row(benchmark_sha, "What diagnosis is shown in the image")],
        tmp_path,
        text_similarity_threshold=0.85,
    )
    assert report["status"] == "failed"
    assert "near_text" in {item["kind"] for item in report["overlaps"]}


def test_missing_provenance_fails_before_overlap(tmp_path: Path) -> None:
    row = external_row("not-checked")
    del row["license"]
    report = validate_external_data([row], tmp_path, [], tmp_path)
    assert report["status"] == "failed"
    assert report["overlap_checks_run"] is False
    assert report["schema_problems"][0]["kind"] == "missing_fields"


def test_cli_writes_hash_bound_pass_report(tmp_path: Path, monkeypatch) -> None:
    external_sha = make_image(tmp_path / "external.png", "diagonal")
    benchmark_sha = make_image(tmp_path / "benchmark.png", "left")
    external_manifest = tmp_path / "external.jsonl"
    benchmark_manifest = tmp_path / "benchmark.jsonl"
    output = tmp_path / "report.json"
    external_manifest.write_text(json.dumps(external_row(external_sha)) + "\n")
    benchmark_manifest.write_text(json.dumps(benchmark_row(benchmark_sha)) + "\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_external_data",
            "--manifest",
            str(external_manifest),
            "--data-root",
            str(tmp_path),
            "--benchmark-manifest",
            str(benchmark_manifest),
            "--benchmark-data-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert set(report["source_hashes"]) == {
        "external_manifest_sha256",
        "benchmark_manifest_sha256",
    }
