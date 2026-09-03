import hashlib
from pathlib import Path

from PIL import Image

from edgemed_bench.run import build_message_content, load_visual_inputs


def _make_image(path: Path, color: str) -> str:
    Image.new("RGB", (20, 10), color).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multiview_loading_hash_checks_and_labels_roles(tmp_path: Path) -> None:
    full_sha = _make_image(tmp_path / "full.png", "white")
    crop_sha = _make_image(tmp_path / "crop.png", "red")
    row = {
        "sample_id": "s1",
        "images": [
            {"role": "full_context", "image_path": "full.png", "image_sha256": full_sha},
            {"role": "local_detail", "image_path": "crop.png", "image_sha256": crop_sha},
        ],
    }
    visuals = load_visual_inputs(row, tmp_path, max_pixels=100)
    assert [item["role"] for item in visuals] == ["full_context", "local_detail"]
    assert all(item["image"].width * item["image"].height <= 100 for item in visuals)
    content = build_message_content(visuals, "Question?")
    assert [item["type"] for item in content] == ["text", "image", "text", "image", "text"]
    assert content[-1] == {"type": "text", "text": "Question?"}


def test_legacy_single_image_content_is_unchanged(tmp_path: Path) -> None:
    image_sha = _make_image(tmp_path / "single.png", "gray")
    visuals = load_visual_inputs(
        {
            "sample_id": "s1",
            "image_path": "single.png",
            "image_sha256": image_sha,
        },
        tmp_path,
        max_pixels=None,
    )
    content = build_message_content(visuals, "Question?")
    assert [item["type"] for item in content] == ["image", "text"]
