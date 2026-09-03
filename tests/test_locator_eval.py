from edgemed_bench.score_locator import box_iou, score_locator_predictions, valid_box


def test_box_iou_and_validation() -> None:
    assert box_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert box_iou([0, 0, 100, 100], [100, 100, 200, 200]) == 0.0
    assert valid_box([10, 20, 30, 40]) is True
    assert valid_box([0, 0, 1000, 1000]) is True
    assert valid_box([10, 20, 10, 40]) is False


def test_locator_scorer_counts_invalid_as_zero() -> None:
    predictions = [
        {
            "sample_id": "s1",
            "status": "completed",
            "tool_call": {
                "name": "region_inspect",
                "arguments": {
                    "media_id": "image-0",
                    "region_xyxy_1000": [100, 100, 500, 500],
                },
            },
        },
        {"sample_id": "s2", "status": "failed"},
    ]
    targets = [
        {"sample_id": "s1", "target_label": "Liver", "region_xyxy_1000": [100, 100, 500, 500]},
        {"sample_id": "s2", "target_label": "Spleen", "region_xyxy_1000": [200, 200, 400, 400]},
    ]
    metrics = score_locator_predictions(predictions, targets)
    assert metrics["valid_output_rate"] == 0.5
    assert metrics["targeted_area_rate"] == 0.5
    assert metrics["mean_iou"] == 0.5
    assert metrics["iou_at_0_3"] == 0.5
