"""Audited vocabulary policies for same-index conditional distillation.

The opt-in Llama 3 policy conditions each model on the 128,000 identical
lexical tokens and five identical active chat tokens. Excluded output rows
retain their original checkpoint identities; they are never renamed.
"""

from __future__ import annotations

import json
import re


_BASE_SIZE = 128000
_HEAD_SIZE = 128256
_ACTIVE_SPECIALS = {
    128000: "<|begin_of_text|>",
    128001: "<|end_of_text|>",
    128006: "<|start_header_id|>",
    128007: "<|end_header_id|>",
    128009: "<|eot_id|>",
}
_OPTIONAL_SPECIALS = {
    128004: "<|finetune_right_pad_id|>",
    128008: "<|eom_id|>",
    128010: "<|python_tag|>",
}
_FLAGS = {"single_word": False, "lstrip": False, "rstrip": False,
          "normalized": False, "special": True}
_MACHINERY = ("model", "normalizer", "pre_tokenizer", "post_processor", "decoder")


def validate_vocabulary(teacher_tokenizer, student_tokenizer, policy="full",
                        teacher_config=None, student_config=None):
    """Validate a policy and return its compact, JSON-serializable support.

    ``full`` retains exact tokenizer equality. ``llama3_shared`` requires
    Llama model metadata, identical complete lexical tokenization machinery,
    and audited same-index special tokens. Different templates are permitted
    because the training pipeline renders one shared prompt for both models.
    """
    if policy not in {"full", "llama3_shared"}:
        raise ValueError(f"Unknown vocabulary policy: {policy!r}")
    tv, sv = teacher_tokenizer.get_vocab(), student_tokenizer.get_vocab()
    if policy == "full" and tv != sv:
        raise ValueError("Teacher/student full token-to-ID maps differ; token-level KD is invalid")
    tj, sj = [json.loads(t.backend_tokenizer.to_str())
              for t in (teacher_tokenizer, student_tokenizer)]
    for key in _MACHINERY + (("added_tokens",) if policy == "full" else ()):
        if tj.get(key) != sj.get(key):
            raise ValueError(f"Teacher/student tokenizer {key} differs")
    if (teacher_tokenizer.eos_token_id != student_tokenizer.eos_token_id
            or teacher_tokenizer.bos_token_id != student_tokenizer.bos_token_id):
        raise ValueError("Teacher/student BOS/EOS semantics differ")
    if policy == "full":
        width = max(tv.values(), default=-1) + 1
        if teacher_config is not None or student_config is not None:
            if not isinstance(teacher_config, dict) or not isinstance(student_config, dict):
                raise ValueError("Both model configurations are required to audit full output heads")
            teacher_width, student_width = (config.get("vocab_size")
                                            for config in (teacher_config, student_config))
            if (type(teacher_width) is not int or type(student_width) is not int
                    or teacher_width != student_width or teacher_width < width):
                raise ValueError("Full output heads must have equal sizes covering all tokenizer IDs")
            width = teacher_width
        return {"policy": policy, "vocabulary_size": width,
                "base_vocabulary_size": width, "shared_special_ids": [],
                "excluded_ids": [], "support_size": width}

    for role, tokenizer, vocab, backend, config in (
        ("teacher", teacher_tokenizer, tv, tj, teacher_config),
        ("student", student_tokenizer, sv, sj, student_config),
    ):
        if not isinstance(config, dict) or config.get("model_type") != "llama":
            raise ValueError(f"llama3_shared requires {role} model_type llama metadata")
        if config.get("vocab_size") != _HEAD_SIZE:
            raise ValueError(f"llama3_shared requires {role} model vocab_size {_HEAD_SIZE}")
        model = backend.get("model")
        if not isinstance(model, dict) or model.get("type") != "BPE":
            raise ValueError(f"llama3_shared requires a BPE {role} tokenizer")
        base_vocab = model.get("vocab", {})
        if (len(base_vocab) != _BASE_SIZE
                or set(base_vocab.values()) != set(range(_BASE_SIZE))):
            raise ValueError(f"llama3_shared requires {_BASE_SIZE} contiguous {role} lexical IDs")
        if len(vocab) != _HEAD_SIZE or set(vocab.values()) != set(range(_HEAD_SIZE)):
            raise ValueError(f"llama3_shared requires {_HEAD_SIZE} contiguous {role} full IDs")
        added = backend.get("added_tokens", [])
        by_id = {token["id"]: token for token in added}
        if len(added) != 256 or set(by_id) != set(range(_BASE_SIZE, _HEAD_SIZE)):
            raise ValueError(f"llama3_shared requires exactly 256 {role} added special tokens")
        for token_id, token in by_id.items():
            if {k: v for k, v in token.items() if k not in {"id", "content"}} != _FLAGS:
                raise ValueError(f"llama3_shared {role} added token flags differ at ID {token_id}")
            content = token.get("content", "")
            if token_id in _ACTIVE_SPECIALS:
                if content != _ACTIVE_SPECIALS[token_id]:
                    raise ValueError(f"llama3_shared {role} active special token differs at ID {token_id}")
            elif (content != _OPTIONAL_SPECIALS.get(token_id)
                  and re.fullmatch(r"<\|reserved_special_token_\d+\|>", content) is None):
                raise ValueError(f"llama3_shared unknown {role} added token at ID {token_id}")
            if vocab.get(content) != token_id:
                raise ValueError(f"llama3_shared {role} added token map differs at ID {token_id}")
        if any(vocab.get(token) != token_id for token, token_id in base_vocab.items()):
            raise ValueError(f"llama3_shared {role} lexical token map differs")
        if tokenizer.bos_token_id != 128000 or tokenizer.eos_token_id != 128009:
            raise ValueError(f"llama3_shared {role} BOS/EOS semantics differ")
        if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in _ACTIVE_SPECIALS:
            raise ValueError(f"llama3_shared {role} padding token is outside the shared support")

    excluded = [i for i in range(_BASE_SIZE, _HEAD_SIZE) if i not in _ACTIVE_SPECIALS]
    return {"policy": policy, "vocabulary_size": _HEAD_SIZE,
            "base_vocabulary_size": _BASE_SIZE,
            "shared_special_ids": list(_ACTIVE_SPECIALS),
            "excluded_ids": excluded, "support_size": _BASE_SIZE + len(_ACTIVE_SPECIALS)}


def support_indices(alignment, device=None):
    """Return original vocabulary IDs in the projected distribution's order."""
    import torch
    base = torch.arange(alignment["base_vocabulary_size"], device=device)
    specials = torch.tensor(alignment["shared_special_ids"], dtype=torch.long, device=device)
    return torch.cat((base, specials)) if specials.numel() else base


def allowed_token_mask(alignment, device=None):
    """Return a boolean mask in the original output-head coordinates."""
    import torch
    mask = torch.ones(alignment["vocabulary_size"], dtype=torch.bool, device=device)
    if alignment["excluded_ids"]:
        mask[torch.tensor(alignment["excluded_ids"], device=device)] = False
    return mask


def project_logits(logits, alignment):
    """Condition distributions by removing excluded columns before softmax.

    Use this for losses. For original-ID generation use ``mask_logits``.
    Passing -inf masked logits into KL/AB formulas can otherwise yield NaNs.
    """
    if logits.shape[-1] != alignment["vocabulary_size"]:
        raise ValueError("Model output vocabulary size differs from the audited alignment")
    if not alignment["excluded_ids"]:
        return logits
    return logits.index_select(-1, support_indices(alignment, logits.device))


def mask_logits(logits, alignment):
    """Suppress excluded generation outcomes while preserving original IDs."""
    if logits.shape[-1] != alignment["vocabulary_size"]:
        raise ValueError("Model output vocabulary size differs from the audited alignment")
    if not alignment["excluded_ids"]:
        return logits
    return logits.masked_fill(~allowed_token_mask(alignment, logits.device), float("-inf"))


def assert_supported_token_ids(token_ids, alignment, context="input", ignore_index=None):
    """Reject excluded or out-of-range tokens before either model sees them."""
    import torch
    ids = torch.as_tensor(token_ids)
    if ids.numel() == 0:
        return
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{context} must contain integer token IDs")
    flat = ids.reshape(-1).long()
    if ignore_index is not None:
        flat = flat[flat != ignore_index]
    out_of_range = (flat < 0) | (flat >= alignment["vocabulary_size"])
    if bool(out_of_range.any()):
        raise ValueError(f"{context} contains out-of-range token IDs: {flat[out_of_range].unique().tolist()}")
    permitted = allowed_token_mask(alignment, flat.device)[flat]
    if not bool(permitted.all()):
        raise ValueError(f"{context} contains tokens outside shared vocabulary support: {flat[~permitted].unique().tolist()}")


def project_token_ids(token_ids, alignment, ignore_index=-100):
    """Map original target IDs to projected logit columns, preserving ignore."""
    import torch
    ids = torch.as_tensor(token_ids)
    assert_supported_token_ids(ids, alignment, context="targets", ignore_index=ignore_index)
    if not alignment["excluded_ids"]:
        return ids
    mapping = torch.full((alignment["vocabulary_size"],), -1, dtype=torch.long, device=ids.device)
    indices = support_indices(alignment, ids.device)
    mapping[indices] = torch.arange(indices.numel(), device=ids.device)
    projected = torch.full_like(ids, ignore_index)
    selected = ids != ignore_index
    projected[selected] = mapping[ids[selected].long()].to(ids.dtype)
    return projected
