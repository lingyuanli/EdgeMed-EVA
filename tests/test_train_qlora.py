import pytest

from edgemed_bench.train_qlora import assistant_loss_labels


def test_assistant_loss_labels_masks_exact_prompt_prefix() -> None:
    assert assistant_loss_labels([1, 2, 3], [1, 2, 3, 4, 5]) == [-100, -100, -100, 4, 5]


def test_assistant_loss_labels_rejects_non_continuation() -> None:
    with pytest.raises(ValueError, match="exact continuation"):
        assistant_loss_labels([1, 2], [1, 9, 3])

