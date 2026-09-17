"""Complete-prompt generation and full-reference, best-reference ROUGE-L.

One greedy response is scored against every reference. Each example contributes
its highest reference ROUGE-L F1, and the benchmark reports their macro mean.
Neither prompts nor references are silently shortened.
"""
from __future__ import annotations

import math
import sys

from .data import validate_records
from .evaluation import (metric_definition as single_reference_definition,
                         rouge_l_score, _validate_reference, _preserve_evaluation_state)
from .models import check_supported, generate_response, render_prompt


METADATA_FIELDS = ("source_group", "source_rows", "topic")


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def metric_definition(tokenizer="english"):
    return {**single_reference_definition(tokenizer),
            "multiple_references": "maximum_fmeasure_over_all_references_per_example",
            "implementation_protocol": "natural-language-multireference-v1"}


def prepare_nl_records(records, benchmark, rouge_tokenizer="english"):
    """Validate the whole canonical source before applying a pilot limit."""
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError("benchmark must be nonempty text")
    metric_definition(rouge_tokenizer)
    records = validate_records(records)
    for row in records:
        if row.get("benchmark") != benchmark:
            raise ValueError(f"Benchmark identity differs for {row['id']!r}")
        references = row.get("references")
        if (not isinstance(references, list) or not references
                or any(not isinstance(ref, str) or not ref.strip() for ref in references)):
            raise ValueError(f"{row['id']!r}: references must be a nonempty list of nonempty strings")
        if row["response"] != references[0]:
            raise ValueError(f"{row['id']!r}: response must equal the first reference")
        for reference in references:
            _validate_reference(reference, rouge_tokenizer)
    return records


def nl_prompt(tokenizer, text, max_prompt_tokens, enable_thinking=False):
    """Render the exact input with the saved chat policy, refusing truncation."""
    positive_integer(max_prompt_tokens, "max_prompt_tokens")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Evaluation prompt must be nonempty text")
    ids = render_prompt(tokenizer, text, sys.maxsize, enable_thinking)
    if len(ids) > max_prompt_tokens:
        raise ValueError(f"Evaluation prompt requires {len(ids)} tokens, exceeds max_prompt_tokens="
                         f"{max_prompt_tokens}; increase --max-prompt-tokens to preserve the complete input")
    return ids


def reference_scores(prediction, references, rouge_tokenizer="english"):
    scores = [rouge_l_score(prediction, reference, rouge_tokenizer) for reference in references]
    best = max(scores)
    return {"rouge_l": best, "reference_rouge_l": scores,
            "best_reference_index": scores.index(best)}


def validate_prediction(row, source, tokenizer, benchmark, max_prompt_tokens,
                        max_new_tokens, vocabulary_size=None, rouge_tokenizer="english",
                        enable_thinking=False):
    if not isinstance(row, dict):
        raise ValueError("Persisted evaluation prediction must be an object")
    metadata = {"id": source["id"], "prompt": source["prompt"],
                "reference": source["response"], "references": source["references"],
                "benchmark": benchmark,
                **{key: source[key] for key in METADATA_FIELDS if key in source}}
    expected_keys = set(metadata) | {"prompt_ids", "response_ids", "prediction", "generated_tokens",
                                     "rouge_l", "reference_rouge_l", "best_reference_index"}
    if set(row) != expected_keys:
        raise ValueError(f"Persisted evaluation fields differ for {source['id']!r}")
    for key, expected in metadata.items():
        if row[key] != expected:
            raise ValueError(f"Persisted evaluation {key} differs for {source['id']!r}")
    for key, limit in (("prompt_ids", max_prompt_tokens), ("response_ids", max_new_tokens)):
        ids = row[key]
        if (not isinstance(ids, list) or (key == "prompt_ids" and not ids) or len(ids) > limit
                or any(type(token) is not int or token < 0
                       or (vocabulary_size is not None and token >= vocabulary_size) for token in ids)):
            raise ValueError(f"Invalid persisted evaluation {key} for {source['id']!r}")
        check_supported(ids, tokenizer, f"persisted evaluation {key}")
    if row["prompt_ids"] != nl_prompt(tokenizer, source["prompt"], max_prompt_tokens, enable_thinking):
        raise ValueError(f"Persisted prompt token IDs differ for {source['id']!r}")
    prediction = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
    if row["prediction"] != prediction:
        raise ValueError(f"Persisted prediction/token IDs disagree for {source['id']!r}")
    if type(row["generated_tokens"]) is not int or row["generated_tokens"] != len(row["response_ids"]):
        raise ValueError(f"Persisted generated token count differs for {source['id']!r}")
    scores = reference_scores(prediction, source["references"], rouge_tokenizer)
    if (type(row["best_reference_index"]) is not int
            or row["best_reference_index"] != scores["best_reference_index"]):
        raise ValueError(f"Persisted best_reference_index differs for {source['id']!r}")
    stored_scores = row["reference_rouge_l"]
    if not isinstance(stored_scores, list) or len(stored_scores) != len(source["references"]):
        raise ValueError(f"Persisted reference_rouge_l differs for {source['id']!r}")
    for saved, expected in zip([row["rouge_l"], *stored_scores],
                               [scores["rouge_l"], *scores["reference_rouge_l"]]):
        if type(saved) not in (int, float) or not math.isfinite(saved) or abs(saved - expected) > 1e-12:
            raise ValueError(f"Persisted ROUGE-L score differs for {source['id']!r}")
    return row


def summarize_predictions(rows, benchmark):
    rows = list(rows)
    if not rows:
        raise ValueError("ROUGE-L evaluation requires at least one prediction")
    if any(row.get("benchmark") != benchmark for row in rows):
        raise ValueError("ROUGE-L metrics require exactly one benchmark")
    return {"benchmark": benchmark, "rouge_l": math.fsum(row["rouge_l"] for row in rows) / len(rows),
            "examples": len(rows), "references": sum(len(row["references"]) for row in rows),
            "multiple_reference_examples": sum(len(row["references"]) > 1 for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows)}


def evaluate_nl_records(model, tokenizer, records, benchmark, *, max_prompt_tokens,
                        max_new_tokens, rouge_tokenizer="english", enable_thinking=False,
                        cached_predictions=None, on_prediction=None):
    records = prepare_nl_records(records, benchmark, rouge_tokenizer)
    positive_integer(max_prompt_tokens, "max_prompt_tokens")
    positive_integer(max_new_tokens, "max_new_tokens")
    cached = {} if cached_predictions is None else cached_predictions
    if not isinstance(cached, dict) or set(cached) - {row["id"] for row in records}:
        raise ValueError("Cached predictions must map IDs from the supplied benchmark")
    prepared = [(row, nl_prompt(tokenizer, row["prompt"], max_prompt_tokens, enable_thinking))
                for row in records]
    context_limit = getattr(model.config, "max_position_embeddings", None)
    for row, ids in prepared:
        if type(context_limit) is int and context_limit > 0 and len(ids) + max_new_tokens > context_limit:
            raise ValueError(f"Prompt + response exceeds model context for {row['id']!r}; "
                             "reduce --max-new-tokens or use a model with a larger context")
    results = []
    with _preserve_evaluation_state(model):
        for source, prompt_ids in prepared:
            if source["id"] in cached:
                result = validate_prediction(cached[source["id"]], source, tokenizer, benchmark,
                    max_prompt_tokens, max_new_tokens, model.config.vocab_size, rouge_tokenizer, enable_thinking)
            else:
                response_ids = generate_response(model, tokenizer, prompt_ids, max_new_tokens,
                                                 {"temperature": 0.0, "top_p": 1.0, "top_k": 0})
                prediction = tokenizer.decode(response_ids, skip_special_tokens=True)
                result = {"id": source["id"], "prompt": source["prompt"], "reference": source["response"],
                          "references": source["references"], "benchmark": benchmark,
                          "prompt_ids": prompt_ids, "response_ids": response_ids,
                          "prediction": prediction, "generated_tokens": len(response_ids),
                          **{key: source[key] for key in METADATA_FIELDS if key in source},
                          **reference_scores(prediction, source["references"], rouge_tokenizer)}
                if on_prediction is not None:
                    on_prediction(result)
            results.append(result)
    return summarize_predictions(results, benchmark)
