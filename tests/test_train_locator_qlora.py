import json
from pathlib import Path

import pytest

from edgemed_bench.train_locator_qlora import (
    load_training_rows,
    localization_prompt,
    localization_target,
)


def _target(box=None):
    box = box or [100, 200, 500, 600]
    return {
        "sample_id": "s1",
        "target_label": "Liver",
        "region_xyxy_1000": box,
        "tool_call": {
            "name": "region_inspect",
            "arguments": {
                "media_id": "image-0",
                "region_xyxy_1000": box,
                "target": "Liver",
            },
        },
    }


def test_locator_target_is_compact_region_only_json() -> None:
    output = localization_target(_target())
    parsed = json.loads(output)
    assert parsed["content"] == "inspect Liver"
    assert parsed["arguments"]["region_xyxy_1000"] == [100, 200, 500, 600]
    assert "tool_call" not in parsed
    assert "answer" not in output


def test_locator_target_rejects_full_frame() -> None:
    with pytest.raises(ValueError, match="area"):
        localization_target(_target([0, 0, 1000, 1000]))


def test_locator_prompt_matches_runtime_contract() -> None:
    prompt = localization_prompt(
        {
            "sample_id": "s1",
            "question": "Where is the liver?",
            "image_path": "image.jpg",
            "image_sha256": "abc",
        }
    )
    assert "REGION_TOOL=" in prompt
    assert '"question": "Where is the liver?"' in prompt
    assert "Do not return null" in prompt


def test_locator_training_surface_keeps_targets_separate(tmp_path: Path) -> None:
    manifest = tmp_path / "inference.jsonl"
    targets = tmp_path / "targets.jsonl"
    row = {
        "sample_id": "s1",
        "kind": "localization",
        "question": "Where is the liver?",
        "image_path": "image.jpg",
        "image_sha256": "abc",
    }
    manifest.write_text(json.dumps(row) + "\n")
    targets.write_text(json.dumps(_target()) + "\n")
    paired = load_training_rows(manifest, targets)
    assert paired[0][0] == row
    assert paired[0][1]["target_label"] == "Liver"
