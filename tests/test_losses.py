"""Numerical/gradient oracles for the migrated objectives, independent of trainers."""

import pytest
import torch

from baseline_common.losses import causal_distillation_loss, distillm2_loss, token_divergence


def _reference_ab(student, teacher, alpha, beta):
    log_q, log_p = student.log_softmax(-1), teacher.log_softmax(-1)
    q, p = log_q.exp(), log_p.exp()
    if alpha == beta == 0:
        return 0.5 * (log_q - log_p).square().sum(-1)
    if alpha == 0:
        return (q.pow(beta) * (beta * (log_q - log_p) - 1) + p.pow(beta)).sum(-1) / beta**2
    if beta == 0:
        return (p.pow(alpha) * (alpha * (log_p - log_q) - 1) + q.pow(alpha)).sum(-1) / alpha**2
    if alpha + beta == 0:
        return (alpha * (log_q - log_p) + (p / q).pow(alpha) - 1).sum(-1) / alpha**2
    return (
        alpha / (alpha + beta) * p.pow(alpha + beta)
        + beta / (alpha + beta) * q.pow(alpha + beta)
        - p.pow(alpha) * q.pow(beta)
    ).sum(-1) / (alpha * beta)


@pytest.mark.parametrize("alpha,beta", [(0.1, 0.8), (0, 0), (0, 0.8), (0.1, 0), (0.4, -0.4), (1, 0)])
def test_ab_values_and_gradients_match_independent_double_oracle(alpha, beta):
    student = torch.tensor([[0.2, 1.1, -0.4], [2.2, -1.2, 0.4]], requires_grad=True)
    teacher = torch.tensor([[-0.5, 0.3, 0.8], [-1.1, 1.3, 0.1]])
    reference_student = student.detach().double().requires_grad_()
    expected = _reference_ab(reference_student, teacher.double(), alpha, beta)
    actual = token_divergence(student, teacher, "abkd", alpha=alpha, beta=beta)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-4, atol=2e-5)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(student.grad.double(), reference_student.grad, rtol=3e-4, atol=2e-5)


def test_kd_is_forward_kl_over_full_head_and_never_differentiates_teacher():
    student = torch.tensor([[1.0, -1.0, 0.5, 2.0]], requires_grad=True)
    teacher = torch.tensor([[0.2, 0.9, -0.6, 1.3]], requires_grad=True)
    actual = token_divergence(student, teacher)
    p, q = teacher.detach().softmax(-1), student.softmax(-1)
    expected = (p * (p.log() - q.log())).sum(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(student.grad, q.detach() - p)
    assert teacher.grad is None
    # A padded output-head row is still part of the distribution.
    changed = teacher.detach().clone()
    changed[:, -1] += 10
    assert not torch.allclose(actual, token_divergence(student, changed))


def test_fp32_distribution_math_for_bfloat16_inputs():
    s = torch.tensor([[30, -30, 0, 2]], dtype=torch.bfloat16, requires_grad=True)
    t = torch.tensor([[-30, 30, 1, 2]], dtype=torch.bfloat16)
    loss = token_divergence(s, t)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(s.grad).all()


@pytest.mark.parametrize("method", ["skew_forward", "skew_reverse"])
def test_skew_value_and_detached_reverse_mixture_gradient(method):
    student = torch.tensor([[1.3, -0.1, 0.7]], requires_grad=True)
    teacher = torch.tensor([[0.1, 0.5, -0.7]], requires_grad=True)
    reference_student = student.detach().double().requires_grad_()
    p, q = teacher.detach().double().softmax(-1), reference_student.softmax(-1)
    alpha = 0.17
    if method == "skew_forward":
        mixture = alpha * p + (1 - alpha) * q
        expected = (p * (p.log() - mixture.log())).sum(-1)
    else:
        mixture = (1 - alpha) * p + alpha * q.detach()
        expected = (q * (q.log() - mixture.log())).sum(-1)
    actual = token_divergence(student, teacher, method, skew_alpha=alpha)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(student.grad.double(), reference_student.grad, rtol=1e-5, atol=1e-6)
    assert teacher.grad is None
    if method == "skew_reverse":
        wrong_student = student.detach().double().requires_grad_()
        wrong_q = wrong_student.softmax(-1)
        wrong = (wrong_q * (wrong_q.log() - ((1 - alpha) * p + alpha * wrong_q).log())).sum()
        wrong.backward()
        assert not torch.allclose(student.grad.double(), wrong_student.grad, atol=1e-4)


def test_causal_mask_keeps_first_response_and_eos_ignores_unused_nan_logits():
    student = torch.full((1, 6, 3), torch.nan)
    teacher = torch.full_like(student, torch.nan)
    student[0, 2:4] = torch.tensor([[0.2, -0.3, 0.7], [1.0, -0.2, 0.3]])
    teacher[0, 2:4] = torch.tensor([[-0.2, 0.5, 0.1], [0.2, 0.9, -0.5]])
    student.requires_grad_()
    labels = torch.tensor([[-100, -100, -100, 2, 0, -100]])  # EOS=0, also the pad ID.
    actual = causal_distillation_loss(student, teacher, labels, chunk_size=1)
    expected = token_divergence(student[0, 2:4], teacher[0, 2:4]).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(student.grad[0, :2]) == 0
    assert torch.count_nonzero(student.grad[0, 4:]) == 0
    assert torch.count_nonzero(student.grad[0, 2]) > 0
    assert torch.count_nonzero(student.grad[0, 3]) > 0


@pytest.mark.parametrize("method", ["kd", "abkd", "skew_forward", "skew_reverse"])
def test_checkpoint_chunking_preserves_values_and_gradients(method):
    torch.manual_seed(4)
    original = torch.randn(2, 5, 9)
    teacher = torch.randn_like(original)
    labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 1, 2, -100, -100]])
    full = original.clone().requires_grad_()
    chunked = original.clone().requires_grad_()
    a = causal_distillation_loss(full, teacher, labels, method, chunk_size=100, checkpoint_chunks=False)
    b = causal_distillation_loss(chunked, teacher, labels, method, chunk_size=2, checkpoint_chunks=True)
    a.backward()
    b.backward()
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(full.grad, chunked.grad)


def test_distillm2_means_each_response_then_sums_and_averages_pairs():
    torch.manual_seed(18)
    chosen = torch.randn(2, 5, 4, requires_grad=True)
    chosen_teacher = torch.randn_like(chosen)
    rejected = torch.randn(2, 4, 4, requires_grad=True)
    rejected_teacher = torch.randn_like(rejected)
    chosen_labels = torch.tensor([[-100, 1, 2, 3, 1], [-100, -100, -100, 1, -100]])
    rejected_labels = torch.tensor([[-100, 1, -100, -100], [-100, 1, 2, 3]])
    expected_pairs = []
    for i in range(2):
        cmask, rmask = chosen_labels[i, 1:] != -100, rejected_labels[i, 1:] != -100
        expected_pairs.append(
            token_divergence(chosen[i, :-1][cmask], chosen_teacher[i, :-1][cmask], "skew_forward").mean()
            + token_divergence(rejected[i, :-1][rmask], rejected_teacher[i, :-1][rmask], "skew_reverse").mean()
        )
    expected = torch.stack(expected_pairs)
    args = (chosen, chosen_teacher, chosen_labels, rejected, rejected_teacher, rejected_labels)
    torch.testing.assert_close(distillm2_loss(*args), expected.mean())
    torch.testing.assert_close(distillm2_loss(*args, reduction="sum"), expected.sum())
    torch.testing.assert_close(distillm2_loss(*args, reduction="none"), expected)
    distillm2_loss(*args).backward()
    assert chosen.grad.abs().sum() > 0 and rejected.grad.abs().sum() > 0


def test_sum_normalization_across_unequal_microbatches_matches_global_token_mean():
    torch.manual_seed(8)
    s = torch.randn(2, 5, 4, requires_grad=True)
    t = torch.randn_like(s)
    labels = torch.tensor([[-100, 1, 2, 3, 1], [-100, 1, -100, -100, -100]])
    global_loss = causal_distillation_loss(s, t, labels)
    accumulated = sum(causal_distillation_loss(s[i:i+1], t[i:i+1], labels[i:i+1], reduction="sum") for i in range(2)) / 5
    torch.testing.assert_close(global_loss, accumulated)


def test_empty_masks_are_explicit_and_sum_remains_differentiable():
    s = torch.full((1, 3, 4), torch.nan, requires_grad=True)
    labels = torch.full((1, 3), -100)
    result = causal_distillation_loss(s, s.detach(), labels, reduction="sum")
    assert result.item() == 0
    result.backward()
    assert torch.count_nonzero(s.grad) == 0
    with pytest.raises(ValueError, match="no valid"):
        causal_distillation_loss(s, s.detach(), labels)
    with pytest.raises(ValueError, match="Every response"):
        causal_distillation_loss(s, s.detach(), labels, reduction="none")


@pytest.mark.parametrize("kwargs", [{"method": "gkd"}, {"chunk_size": 0}, {"reduction": "bad"}])
def test_rejects_invalid_loss_options(kwargs):
    logits = torch.randn(1, 3, 4)
    with pytest.raises(ValueError):
        causal_distillation_loss(logits, logits, torch.tensor([[-100, 1, 2]]), **kwargs)
