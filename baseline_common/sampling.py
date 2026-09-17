"""Portable speculative KD, using native causal-LM forwards without patches.

The block/intervention policy follows the supplied SKD implementation in
``speculative_kd/speculative_kd/transformers/utils.py``. Acceptance tests the
actual proposed IDs against teacher top-k, correcting the legacy re-sampling
bug. Teacher acceptance ranks the full teacher distribution; generation
top-k/top-p only control token sampling, avoiding arbitrary zero-mass ties.

This reference implementation has no KV cache: student prefixes are recomputed
for each proposal, and one teacher pass verifies each block. Its cost counters
include those repeated input tokens. There is no teacher bonus token when the
entire block is accepted, matching the supplied SKD policy.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from .models import mask_generation_logits, check_supported


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("Sampling top_p must be in (0, 1]")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("Sampling top_k must be a nonnegative integer")

    def as_generate_kwargs(self) -> dict:
        return {"do_sample": True, "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}


def sampling_logits(logits: Tensor, config: SamplingConfig) -> Tensor:
    """FP32 temperature, top-k, then nucleus filtering (retain boundary token)."""
    scores = logits.float() / config.temperature
    if config.top_k:
        threshold = scores.topk(min(config.top_k, scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if config.top_p < 1:
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        remove = sorted_scores.softmax(-1).cumsum(-1) > config.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scores = scores.masked_fill(remove.scatter(-1, sorted_indices, remove), -torch.inf)
    return scores


def sample_token(logits: Tensor, config: SamplingConfig, generator: torch.Generator | None = None) -> Tensor:
    """Sample one ID per row. The optional RNG must match the logits' device."""
    return torch.multinomial(sampling_logits(logits, config).softmax(-1), 1, generator=generator)


def _model_device(model) -> torch.device:
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().weight.device
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def _forward_logits(model, ids: Tensor, keep: int) -> Tensor:
    kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
    # Modern Qwen/Llama can project only the needed suffix. Other models safely
    # use their native full output; no undocumented kwargs or module patches.
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = keep
    elif "num_logits_to_keep" in parameters:
        kwargs["num_logits_to_keep"] = keep
    result = model(**kwargs)
    logits = result["logits"] if isinstance(result, dict) else result.logits
    return mask_generation_logits(logits, model)


@torch.no_grad()
def skd_generate(
    student,
    teacher,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] | int | None,
    proposal_config: SamplingConfig | None = None,
    teacher_config: SamplingConfig | None = None,
    acceptance_k: int = 25,
    block_size: int = 5,
    return_stats: bool = False,
    student_generator: torch.Generator | None = None,
    teacher_generator: torch.Generator | None = None,
):
    """Return one unpadded full sequence on the prompt's original device.

    A single prompt is intentional: callers loop over microbatch prompts, so
    EOS/pad collisions and variable proposal acceptance cannot corrupt masks.
    Separate RNGs allow teacher/student models to reside on different devices.
    Sampling is no-grad and restores both models' prior train/eval modes.
    """
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("SKD requires one nonempty, unpadded prompt")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(acceptance_k, int) or acceptance_k < 1:
        raise ValueError("acceptance_k must be a positive integer")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    proposal_config = proposal_config or SamplingConfig()
    teacher_config = teacher_config or SamplingConfig()
    eos_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or ())
    original_device = prompt_ids.device
    check_supported(prompt_ids, student, "SKD prompt")
    check_supported(prompt_ids, teacher, "SKD prompt")
    if getattr(student, "_baseline_vocab_alignment", None) != getattr(teacher, "_baseline_vocab_alignment", None):
        raise ValueError("SKD teacher/student vocabulary policies differ")
    student_device, teacher_device = _model_device(student), _model_device(teacher)
    sequence = prompt_ids.to(device=student_device, dtype=torch.long)
    prompt_length = sequence.shape[1]
    stats = {
        "proposed_tokens": 0, "accepted_tokens": 0, "teacher_interventions": 0,
        "student_forward_tokens": 0, "teacher_forward_tokens": 0,
        "student_forward_calls": 0, "teacher_forward_calls": 0,
    }
    student_training, teacher_training = student.training, teacher.training
    student.eval()
    teacher.eval()
    try:
        while sequence.shape[1] - prompt_length < max_new_tokens:
            remaining = max_new_tokens - (sequence.shape[1] - prompt_length)
            candidate = sequence
            proposals = []
            vocab_size = None
            for _ in range(min(block_size, remaining)):
                logits = _forward_logits(student, candidate, keep=1)[:, -1, :]
                stats["student_forward_tokens"] += candidate.numel()
                stats["student_forward_calls"] += 1
                vocab_size = logits.shape[-1]
                token = sample_token(logits, proposal_config, student_generator)
                proposals.append(int(token.item()))
                candidate = torch.cat((candidate, token), dim=1)
                if proposals[-1] in eos_ids:
                    break
            n_proposals = len(proposals)
            stats["proposed_tokens"] += n_proposals
            teacher_ids = candidate.to(teacher_device)
            teacher_logits = _forward_logits(teacher, teacher_ids, keep=n_proposals + 1)
            stats["teacher_forward_tokens"] += teacher_ids.numel()
            stats["teacher_forward_calls"] += 1
            if teacher_logits.shape[-1] != vocab_size:
                raise ValueError("SKD requires matching student/teacher output vocabularies")
            # The row before each emitted proposal predicts that proposal.
            verify_logits = teacher_logits[:, -n_proposals - 1 : -1, :]
            if verify_logits.shape[1] != n_proposals:
                raise ValueError("Teacher did not return all proposal verification positions")
            accepted = []
            for position, token in enumerate(proposals):
                teacher_top = verify_logits[0, position].topk(min(acceptance_k, vocab_size)).indices
                if bool((teacher_top == token).any()):
                    accepted.append(token)
                    stats["accepted_tokens"] += 1
                else:
                    replacement = sample_token(verify_logits[:, position, :], teacher_config, teacher_generator)
                    accepted.append(int(replacement.item()))
                    stats["teacher_interventions"] += 1
                    break
            sequence = torch.cat((sequence, sequence.new_tensor([accepted])), dim=1)
            if accepted[-1] in eos_ids:
                break
    finally:
        student.train(student_training)
        teacher.train(teacher_training)
    stats["generated_tokens"] = sequence.shape[1] - prompt_length
    output = sequence.to(original_device)
    return (output, stats) if return_stats else output
