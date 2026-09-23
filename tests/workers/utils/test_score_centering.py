import pytest
import torch

from verl.workers.utils.score_centering import correction_for_response_logits, topk_score_correction


def test_centered_constant_reward_has_zero_expected_gradient():
    logits = torch.tensor([[1.2, 0.3, -0.4, -1.1]], dtype=torch.float64, requires_grad=True)
    topk_ids = torch.tensor([[0, 1]])
    q_head = torch.tensor([[0.65, 0.20]], dtype=torch.float64)
    correction, q_mass, p_mass = topk_score_correction(logits, topk_ids, q_head.log())
    p = logits.detach().softmax(-1).squeeze(0)
    rho = (1 - q_mass.item()) / (1 - p_mass.item())
    q = torch.tensor([0.65, 0.20, rho * p[2].item(), rho * p[3].item()])
    assert q.sum().item() == pytest.approx(1.0, abs=1e-6)

    expected_gradient = torch.zeros_like(logits)
    for token_id in range(4):
        sampled_logp = logits.log_softmax(-1)[0, token_id]
        gradient = torch.autograd.grad(sampled_logp - correction[0], logits, retain_graph=True)[0]
        expected_gradient += q[token_id] * gradient
    torch.testing.assert_close(expected_gradient, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_missing_topk_slots_contribute_no_mass():
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    ids = torch.tensor([[1, 0, 0]])
    q_logp = torch.tensor([[0.8, 0.2, 0.0]]).log()
    q_logp[0, 2] = float("-inf")
    correction, mass, _ = topk_score_correction(logits, ids, q_logp)
    assert torch.isfinite(correction).all()
    assert mass.item() == pytest.approx(1.0, abs=1e-6)
    correction.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_response_alignment_with_padded_responses():
    # Two nested sequences, but (as in OpenPangu) responses and attention masks
    # remain padded after left_right_2_no_padding.
    input_ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7])], layout=torch.jagged)
    logits = torch.randn(7, 5, requires_grad=True)
    data = {
        "input_ids": input_ids,
        "prompts": torch.tensor([[0, 1, 2], [0, 0, 5]]),
        "responses": torch.tensor([[3, 4], [6, 7]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]]),
        "score_centering_ids": torch.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]]),
        "score_centering_logprobs": torch.log(torch.tensor([[[0.5, 0.2], [0.5, 0.2]], [[0.5, 0.2], [0.5, 0.2]]])),
    }
    correction, _, _ = correction_for_response_logits(logits, data)
    assert correction.shape == (7,)
    assert correction[[0, 3, 6]].tolist() == [0.0, 0.0, 0.0]
    correction.sum().backward()
    assert logits.grad.abs().sum() > 0
