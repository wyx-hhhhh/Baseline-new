"""Generated-answer ROUGE-L shared by training and standalone evaluation.

The metric is the arithmetic mean of each answer's ``rougeL`` F1, multiplied
by 100. It is neither token accuracy nor the sentence-union ``rougeLsum``.
English mode uses Google Research's rouge-score 0.1.2 tokenizer and Porter
stemming, matching the archived baseline protocol. Unicode mode is explicit:
NFKC/casefold normalization, alphanumeric words, individual CJK characters,
and no stemming. No reference answer is truncated to the generation budget.

Primary implementation and definitions:
https://github.com/google-research/google-research/tree/master/rouge
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib.metadata
import math
import random
import re
import unicodedata

import numpy as np
import torch
from rouge_score import rouge_scorer, tokenizers

from .models import check_supported, generate_response, render_prompt


class UnicodeTokenizer:
    """An explicit multilingual alternative to the ASCII English tokenizer.

    This is a deterministic tokenization convention, not language-specific
    word segmentation. Han ideographs, kana and Hangul syllables are separate
    tokens; other scripts use maximal Unicode alphanumeric runs. Punctuation
    separates tokens. Unicode normalization follows this Python runtime.
    """

    def tokenize(self, text):
        normalized = unicodedata.normalize("NFKC", text).casefold()
        separated = "".join(
            f" {char} " if unicodedata.name(char, "").startswith((
                "CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH",
                "HIRAGANA", "KATAKANA", "HANGUL SYLLABLE",
            )) else char
            for char in normalized
        )
        return re.findall(r"[^\W_]+", separated, flags=re.UNICODE)


@lru_cache(maxsize=2)
def _metric(tokenizer):
    if tokenizer not in {"english", "unicode"}:
        raise ValueError("ROUGE tokenizer must be 'english' or 'unicode'")
    metric_tokenizer = (tokenizers.DefaultTokenizer(use_stemmer=True)
                        if tokenizer == "english" else UnicodeTokenizer())
    return rouge_scorer.RougeScorer(["rougeL"], tokenizer=metric_tokenizer), metric_tokenizer


def metric_definition(tokenizer="english"):
    """Return JSON-safe metric provenance suitable for result manifests."""
    _metric(tokenizer)
    result = {
        "name": "rouge_l", "variant": "rougeL", "statistic": "fmeasure",
        "implementation": "rouge-score", "version": importlib.metadata.version("rouge-score"),
        "aggregation": "arithmetic_mean_over_examples", "range": [0, 100],
        "higher_is_better": True, "tokenizer": tokenizer,
        "use_stemmer": tokenizer == "english", "truncate_references": False,
        "normalization": ("lowercase ASCII alphanumeric tokens; Porter stemming for tokens longer than 3 characters"
                          if tokenizer == "english" else
                          "NFKC and casefold; Unicode alphanumeric words; individual Han, kana and Hangul syllables"),
    }
    if tokenizer == "unicode":
        result["unicode_version"] = unicodedata.unidata_version
    else:
        result["stemmer"] = {"implementation": "nltk.stem.porter.PorterStemmer",
                             "version": importlib.metadata.version("nltk")}
    return result


def _validate_reference(reference, tokenizer):
    if not isinstance(reference, str):
        raise ValueError("ROUGE reference must be text")
    _, metric_tokenizer = _metric(tokenizer)
    if (tokenizer == "english" and any(char.isalnum() for char in reference)
            and not metric_tokenizer.tokenize(reference)):
        raise ValueError("Reference has no English ROUGE tokens; set eval_rouge_tokenizer='unicode' for this dataset")


def rouge_l_score(prediction, reference, tokenizer="english"):
    """Score one generated answer's LCS F1 in [0, 100]; empty text scores 0."""
    if not isinstance(prediction, str):
        raise ValueError("ROUGE prediction must be text")
    _validate_reference(reference, tokenizer)
    scorer, _ = _metric(tokenizer)
    return 100.0 * scorer.score(reference, prediction)["rougeL"].fmeasure


@contextmanager
def _preserve_evaluation_state(model):
    """Evaluation must not perturb later dropout, sampling or data ordering."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    # A CUDA model initializes CUDA before entering evaluation. Do not create
    # GPU contexts merely because a CPU-only evaluation can see a GPU host.
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        # Preserve deliberately mixed submodule modes as well as model.training.
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cached_result(cached, row, prompt_ids, tokenizer, metric_tokenizer, max_new_tokens):
    if not isinstance(cached, dict):
        raise ValueError(f"Invalid cached prediction for {row['id']}")
    for key, expected in (("id", row["id"]), ("prompt", row["prompt"]),
                          ("reference", row["response"]), ("prompt_ids", prompt_ids)):
        if cached.get(key) != expected:
            raise ValueError(f"Cached prediction {key} differs for {row['id']}")
    ids = cached.get("response_ids")
    if (not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids)
            or len(ids) > max_new_tokens):
        raise ValueError(f"Invalid cached response_ids for {row['id']}")
    count = cached.get("generated_tokens")
    if type(count) is not int or count != len(ids):
        raise ValueError(f"Invalid cached generated_tokens for {row['id']}")
    check_supported(ids, tokenizer, "cached evaluation response")
    prediction = tokenizer.decode(ids, skip_special_tokens=True)
    if cached.get("prediction") != prediction:
        raise ValueError(f"Cached decoded prediction differs for {row['id']}")
    score = rouge_l_score(prediction, row["response"], metric_tokenizer)
    saved_score = cached.get("rouge_l")
    if (type(saved_score) not in {int, float} or not math.isfinite(saved_score)
            or not math.isclose(saved_score, score, rel_tol=0, abs_tol=1e-10)):
        raise ValueError(f"Cached rouge_l differs for {row['id']}")
    return {**cached, "rouge_l": score}


def evaluate_records(model, tokenizer, rows, cfg, *, max_examples=None,
                     on_prediction=None, cached_predictions=None):
    """Greedily generate and score canonical prompt/response records.

    ``max_examples=None`` evaluates the entire supplied set; training callers
    explicitly pass their configured validation limit. ``on_prediction`` gets
    a JSON-safe record only for newly generated answers. A cache is a mapping
    from example ID to such a record. The caller must separately bind that
    cache to checkpoint, dataset and generation-setting identities. A raised
    callback (for interruption, for example) restores RNG and model modes.
    """
    rows = list(rows)
    if max_examples is not None:
        rows = rows[:_positive_integer(max_examples, "max_examples")]
    if not rows:
        raise ValueError("ROUGE-L evaluation requires at least one example")
    metric_tokenizer = cfg.get("eval_rouge_tokenizer", "english")
    _metric(metric_tokenizer)
    max_new_tokens = _positive_integer(cfg.get("eval_max_new_tokens") or cfg["max_new_tokens"],
                                       "eval_max_new_tokens")
    max_prompt_tokens = _positive_integer(cfg["max_prompt_tokens"], "max_prompt_tokens")
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"] or not isinstance(row.get("prompt"), str)):
            raise ValueError("Evaluation requires canonical text records with nonempty IDs")
        if row["id"] in seen:
            raise ValueError(f"Duplicate evaluation ID: {row['id']}")
        seen.add(row["id"])
        _validate_reference(row.get("response"), metric_tokenizer)
    cached_predictions = {} if cached_predictions is None else cached_predictions
    if not isinstance(cached_predictions, dict) or set(cached_predictions) - seen:
        raise ValueError("Cached predictions must map IDs from this evaluation subset")

    scores, generated_tokens = [], 0
    with _preserve_evaluation_state(model):
        for row in rows:
            prompt_ids = render_prompt(tokenizer, row["prompt"], max_prompt_tokens,
                                       cfg.get("enable_thinking", False))
            context_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
            if (type(context_limit) is int and context_limit > 0
                    and len(prompt_ids) + max_new_tokens > context_limit):
                raise ValueError(
                    f"Evaluation prompt + response budget ({len(prompt_ids)} + {max_new_tokens}) "
                    f"exceeds model context ({context_limit}) for example {row['id']}"
                )
            if row["id"] in cached_predictions:
                result = _cached_result(cached_predictions[row["id"]], row, prompt_ids,
                                        tokenizer, metric_tokenizer, max_new_tokens)
            else:
                response_ids = generate_response(model, tokenizer, prompt_ids, max_new_tokens,
                    {"temperature": 0.0, "top_p": 1.0, "top_k": 0})
                prediction = tokenizer.decode(response_ids, skip_special_tokens=True)
                result = {
                    "id": row["id"], "prompt": row["prompt"], "reference": row["response"],
                    "prompt_ids": prompt_ids, "response_ids": response_ids,
                    "prediction": prediction,
                    "rouge_l": rouge_l_score(prediction, row["response"], metric_tokenizer),
                    "generated_tokens": len(response_ids),
                }
                if on_prediction is not None:
                    on_prediction(result)
            scores.append(result["rouge_l"])
            generated_tokens += result["generated_tokens"]
    return {"rouge_l": math.fsum(scores) / len(scores), "examples": len(scores),
            "generated_tokens": generated_tokens}
