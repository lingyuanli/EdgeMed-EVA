import torch

from edgemed_bench.run_mcq_likelihood import mean_target_logprob


def test_mean_target_logprob_aligns_completion_tokens() -> None:
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits = torch.zeros((1, 4, 5))
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 4.0
    score, count = mean_target_logprob(logits, input_ids, prefix_length=2)
    expected = float(torch.log_softmax(torch.tensor([0.0, 0.0, 4.0, 0.0, 0.0]), 0)[2])
    assert count == 2
    assert abs(score - expected) < 1e-6


def test_mean_target_logprob_rejects_empty_completion() -> None:
    try:
        mean_target_logprob(torch.zeros((1, 2, 3)), torch.tensor([[0, 1]]), 2)
    except ValueError as error:
        assert "at least one token" in str(error)
    else:
        raise AssertionError("empty completion was accepted")
