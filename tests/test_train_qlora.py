import pytest

from edgemed_bench.train_qlora import assistant_loss_labels, require_finite_gradient_norm


def test_assistant_loss_labels_masks_exact_prompt_prefix() -> None:
    assert assistant_loss_labels([1, 2, 3], [1, 2, 3, 4, 5]) == [-100, -100, -100, 4, 5]


def test_assistant_loss_labels_rejects_non_continuation() -> None:
    with pytest.raises(ValueError, match="exact continuation"):
        assistant_loss_labels([1, 2], [1, 9, 3])


def test_gradient_norm_gate_accepts_finite_value() -> None:
    assert require_finite_gradient_norm(1.25, 2) == 1.25


def test_gradient_norm_gate_rejects_nan() -> None:
    with pytest.raises(FloatingPointError, match="optimizer step 1"):
        require_finite_gradient_norm(float("nan"), 1)
