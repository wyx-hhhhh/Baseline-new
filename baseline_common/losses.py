"""Full-vocabulary distillation losses with response-only causal alignment.

AB formulas follow ``abkd/distillation_llm/distillm/losses.py::ab_div``.
The paired skew objectives and detached reverse mixture follow
``distillm-2/src/distillm_trainer.py::get_batch_logps`` and ``dpo_loss``.
Teacher logits are always detached; all distribution arithmetic uses FP32.
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def token_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    method: str = "kd",
    *,
    alpha: float = 0.1,
    beta: float = 0.8,
    skew_alpha: float = 0.1,
) -> Tensor:
    """Return one divergence per token, retaining the complete output head.

    ``kd``/``skd`` is KL(teacher || student). ``abkd`` implements the
    alpha-beta divergence including its four analytic limiting cases.
    ``skew_forward`` and ``skew_reverse`` are the two DistiLLM-2 branches.
    No temperature scaling is implicit: the primary protocol uses T=1.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim < 2:
        raise ValueError("Teacher/student logits must have equal [..., vocab] shapes")
    log_q = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
    log_p = F.log_softmax(teacher_logits.detach(), dim=-1, dtype=torch.float32)
    if method in {"kd", "skd"}:
        return (log_p.exp() * (log_p - log_q)).sum(-1)
    if method == "abkd":
        if not math.isfinite(alpha) or not math.isfinite(beta):
            raise ValueError("AB alpha and beta must be finite")
        eps = 1e-8
        if abs(alpha) < eps and abs(beta) < eps:
            return 0.5 * (log_q - log_p).square().sum(-1)
        if abs(alpha) < eps:
            return (
                (beta * log_q).exp() * (beta * (log_q - log_p) - 1)
                + (beta * log_p).exp()
            ).sum(-1) / beta**2
        if abs(beta) < eps:
            return (
                (alpha * log_p).exp() * (alpha * (log_p - log_q) - 1)
                + (alpha * log_q).exp()
            ).sum(-1) / alpha**2
        if abs(alpha + beta) < eps:
            ratio = log_q - log_p
            return (alpha * ratio + (-alpha * ratio).expm1()).sum(-1) / alpha**2
        total = alpha + beta
        cross = torch.logsumexp(alpha * log_p + beta * log_q, dim=-1).exp()
        p_moment = torch.logsumexp(total * log_p, dim=-1).exp()
        q_moment = torch.logsumexp(total * log_q, dim=-1).exp()
        return ((alpha / total) * p_moment + (beta / total) * q_moment - cross) / (alpha * beta)
    if method in {"skew_forward", "skew_reverse"}:
        if not 0 < skew_alpha < 1:
            raise ValueError("DistiLLM-2 skew_alpha must be strictly between 0 and 1")
        if method == "skew_forward":
            mixture = torch.logaddexp(
                math.log(skew_alpha) + log_p, math.log1p(-skew_alpha) + log_q
            )
            return (log_p.exp() * (log_p - mixture)).sum(-1)
        # This detach is part of the released DistiLLM-2 student-side gradient.
        mixture = torch.logaddexp(
            math.log1p(-skew_alpha) + log_p, math.log(skew_alpha) + log_q.detach()
        )
        return (log_q.exp() * (log_q - mixture)).sum(-1)
    raise ValueError(f"Unsupported distillation method: {method!r}")


def causal_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    method: str = "kd",
    *,
    chunk_size: int = 128,
    reduction: str = "token_mean",
    alpha: float = 0.1,
    beta: float = 0.8,
    skew_alpha: float = 0.1,
    checkpoint_chunks: bool = True,
) -> Tensor:
    """Distill logits at t against a valid response label at t+1.

    Labels have shape [batch, sequence], with prompt/padding labels set to
    -100 and real EOS labels retained. Invalid positions are excluded *before*
    softmax. Only selected token chunks acquire FP32 vocabulary intermediates.
    Activation checkpointing recomputes these intermediates during backward;
    it does not eliminate the models' original full logits or activations.

    ``sum`` supports external normalization across accumulation/ranks.
    ``none`` returns each response's token mean; ``sequence_mean`` averages
    those means. ``token_mean`` divides by all valid tokens in this batch.
    """
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Expected equal [batch, sequence, vocab] teacher/student logits")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("Labels must match the logits' batch and sequence dimensions")
    if labels.device != student_logits.device or teacher_logits.device != student_logits.device:
        raise ValueError("Loss logits and labels must be on the same device")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if reduction not in {"sum", "token_mean", "sequence_mean", "none"}:
        raise ValueError(f"Unknown reduction: {reduction!r}")
    positions = labels[:, 1:].ne(-100).nonzero(as_tuple=False)
    counts = labels[:, 1:].ne(-100).sum(-1)
    if reduction in {"none", "sequence_mean"} and bool((counts == 0).any()):
        raise ValueError("Every response must contain at least one valid causal target")
    if reduction == "token_mean" and positions.shape[0] == 0:
        raise ValueError("The batch has no valid causal response targets")
    # Empty slice gives a differentiable zero even if unused logits are NaN.
    zero = student_logits.reshape(-1)[:0].sum(dtype=torch.float32)
    sums = torch.zeros(student_logits.shape[0], dtype=torch.float32, device=student_logits.device) + zero
    divergence = partial(token_divergence, method=method, alpha=alpha, beta=beta, skew_alpha=skew_alpha)
    for start in range(0, positions.shape[0], chunk_size):
        indices = positions[start : start + chunk_size]
        batch_indices, prediction_indices = indices.unbind(-1)
        # prediction_indices already index the prefix immediately before labels[:, 1:].
        student_chunk = student_logits[batch_indices, prediction_indices]
        teacher_chunk = teacher_logits.detach()[batch_indices, prediction_indices]
        if checkpoint_chunks and student_chunk.requires_grad and torch.is_grad_enabled():
            values = checkpoint(divergence, student_chunk, teacher_chunk, use_reentrant=False)
        else:
            values = divergence(student_chunk, teacher_chunk)
        sums = sums.index_add(0, batch_indices, values)
    if reduction == "sum":
        return sums.sum()
    if reduction == "token_mean":
        return sums.sum() / counts.sum()
    means = sums / counts
    return means if reduction == "none" else means.mean()


def distillm2_loss(
    chosen_student: Tensor,
    chosen_teacher: Tensor,
    chosen_labels: Tensor,
    rejected_student: Tensor,
    rejected_teacher: Tensor,
    rejected_labels: Tensor,
    *,
    alpha_1: float = 0.1,
    alpha_2: float = 0.1,
    chunk_size: int = 128,
    reduction: str = "mean",
    checkpoint_chunks: bool = True,
) -> Tensor:
    """Teacher chosen + initial-student rejected response means, averaged by pair.

    This is released ``distillm_v2`` with adaptive alpha and gradual beta off
    (effective beta=1). ``sum`` sums pair losses for global normalization.
    """
    if chosen_student.shape[0] != rejected_student.shape[0]:
        raise ValueError("Chosen and rejected batches must contain the same number of pairs")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown pair reduction: {reduction!r}")
    chosen = causal_distillation_loss(
        chosen_student, chosen_teacher, chosen_labels, "skew_forward",
        skew_alpha=alpha_1, chunk_size=chunk_size, reduction="none", checkpoint_chunks=checkpoint_chunks,
    )
    rejected = causal_distillation_loss(
        rejected_student, rejected_teacher, rejected_labels, "skew_reverse",
        skew_alpha=alpha_2, chunk_size=chunk_size, reduction="none", checkpoint_chunks=checkpoint_chunks,
    )
    pairs = chosen + rejected
    if reduction == "none":
        return pairs
    return pairs.sum() if reduction == "sum" else pairs.mean()
