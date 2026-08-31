import pytest

from edgemed_bench.io import reject_reference_fields


def test_clean_manifest_is_allowed() -> None:
    reject_reference_fields([{"sample_id": "mcq-0", "question": "q"}])


@pytest.mark.parametrize("field", ["answer", "visual_description", "ground_truth", "reference"])
def test_reference_fields_are_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        reject_reference_fields([{"sample_id": "mcq-0", field: "secret"}])
