"""Greedy math inference with explicit final-answer grading and cache validation.

Questions are kept separately from the evaluation instruction so benchmark
identity and leakage checks always use the original question. No reference is
truncated and no example with an invalid reference enters the denominator.
"""
from __future__ import annotations

from .data import validate_records
from .math_metrics import math_score, metric_definition, validate_reference
from .models import check_supported, generate_response, render_prompt


MATH_INSTRUCTION = r"Solve the problem step by step. Put your final answer inside \boxed{}."


def render_math_prompt_text(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Math question must be nonempty text")
    return question + "\n\n" + MATH_INSTRUCTION


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prepare_math_records(records, benchmark):
    """Validate every gold, including examples outside a requested pilot limit."""
    metric_definition(benchmark)
    records = validate_records(records)
    for row in records:
        if row.get("benchmark", benchmark) != benchmark:
            raise ValueError(f"Benchmark identity differs for {row['id']!r}")
        try:
            validate_reference(row["response"], benchmark)
        except ValueError as error:
            raise ValueError(f"Invalid {benchmark} reference for {row['id']!r}: {error}") from error
    return records


def math_prompt(tokenizer, question, max_prompt_tokens):
    """Preserve the complete math question and instruction, or reject the budget.

    Silent question truncation would change a benchmark problem. A larger
    --max-prompt-tokens should be selected when an input does not fit.
    """
    positive_integer(max_prompt_tokens, "max_prompt_tokens")
    text = render_math_prompt_text(question)
    # Give the shared renderer sufficient room to apply the chat template
    # without shortening the question. The byte-derived bound is conservative
    # even for tokenizers with byte fallback and includes template overhead.
    ids = render_prompt(tokenizer, text, max(len(text.encode("utf-8")) * 4 + 16384,
                                             max_prompt_tokens), False)
    if len(ids) > max_prompt_tokens:
        raise ValueError(f"Math prompt requires {len(ids)} tokens, exceeds max_prompt_tokens="
                         f"{max_prompt_tokens}; increase --max-prompt-tokens to preserve the full problem")
    return ids


def validate_prediction(row, source, tokenizer, benchmark, max_prompt_tokens,
                        max_new_tokens, vocabulary_size=None):
    if not isinstance(row, dict):
        raise ValueError("Persisted math prediction must be an object")
    for key, expected in (("id", source["id"]), ("prompt", source["prompt"]),
                          ("reference", source["response"]), ("benchmark", benchmark),
                          ("source_group", source.get("source_group"))):
        if row.get(key) != expected:
            raise ValueError(f"Persisted math {key} differs for {source['id']!r}")
    for key, limit in (("prompt_ids", max_prompt_tokens), ("response_ids", max_new_tokens)):
        ids = row.get(key)
        if (not isinstance(ids, list) or (key == "prompt_ids" and not ids) or len(ids) > limit
                or any(type(token) is not int or token < 0
                       or (vocabulary_size is not None and token >= vocabulary_size) for token in ids)):
            raise ValueError(f"Invalid persisted math {key} for {source['id']!r}")
        check_supported(ids, tokenizer, f"persisted math {key}")
    if row["prompt_ids"] != math_prompt(tokenizer, source["prompt"], max_prompt_tokens):
        raise ValueError(f"Persisted math prompt token IDs differ for {source['id']!r}")
    prediction = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
    if row.get("prediction") != prediction:
        raise ValueError(f"Persisted math prediction/token IDs disagree for {source['id']!r}")
    if type(row.get("generated_tokens")) is not int or row["generated_tokens"] != len(row["response_ids"]):
        raise ValueError(f"Persisted math token count differs for {source['id']!r}")
    expected_score = math_score(prediction, source["response"], benchmark)
    metadata_keys = {"id", "prompt", "reference", "benchmark", "source_group", "prompt_ids",
                     "response_ids", "prediction", "generated_tokens"}
    if set(row) - metadata_keys != set(expected_score):
        raise ValueError(f"Persisted math grade fields differ for {source['id']!r}")
    if type(row.get("correct")) is not bool:
        raise ValueError(f"Persisted math correctness must be boolean for {source['id']!r}")
    for key, expected in expected_score.items():
        if key not in row or row[key] != expected or type(row[key]) is not type(expected):
            raise ValueError(f"Persisted math grade {key} differs for {source['id']!r}")
    return row


def summarize_predictions(rows, benchmark):
    rows = list(rows)
    if not rows:
        raise ValueError("Math evaluation requires at least one prediction")
    if any(row.get("benchmark") != benchmark or type(row.get("correct")) is not bool for row in rows):
        raise ValueError("Math metrics require one benchmark and boolean correctness")
    correct = sum(row["correct"] for row in rows)
    accuracy = 100.0 * correct / len(rows)
    return {"benchmark": benchmark, "accuracy_percent": accuracy,
            "pass_at_1": accuracy, "correct_count": correct, "total": len(rows),
            "examples": len(rows),
            "unparseable_predictions": sum(
                bool(row.get("extraction_failed")) or
                (bool(row.get("parse_error")) and not str(row["parse_error"]).startswith("verification_"))
                for row in rows),
            "extraction_failures": sum(bool(row.get("extraction_failed")) for row in rows),
            "parsing_failures": sum(bool(row.get("parse_error"))
                and not row.get("extraction_failed")
                and not str(row["parse_error"]).startswith("verification_") for row in rows),
            "parsing_timeouts": sum(row.get("parse_error") == "parsing_timeout" for row in rows),
            "verification_errors": sum(str(row.get("parse_error", "")).startswith("verification_") for row in rows),
            "verification_timeouts": sum(row.get("parse_error") == "verification_timeout" for row in rows),
            "reference_override_count": sum(row.get("reference_override") is not None for row in rows),
            "generated_tokens": sum(row["generated_tokens"] for row in rows)}


def evaluate_math_records(model, tokenizer, records, benchmark, *, max_prompt_tokens=1024,
                          max_new_tokens=2048, cached_predictions=None, on_prediction=None):
    records = prepare_math_records(records, benchmark)
    positive_integer(max_prompt_tokens, "max_prompt_tokens")
    positive_integer(max_new_tokens, "max_new_tokens")
    cached = {} if cached_predictions is None else cached_predictions
    if not isinstance(cached, dict) or set(cached) - {row["id"] for row in records}:
        raise ValueError("Cached math predictions must map IDs from the supplied benchmark")
    prepared = [(row, math_prompt(tokenizer, row["prompt"], max_prompt_tokens)) for row in records]
    context_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    for row, ids in prepared:
        if type(context_limit) is int and context_limit > 0 and len(ids) + max_new_tokens > context_limit:
            raise ValueError(f"Math prompt + response exceeds model context for {row['id']!r}")
    results = []
    for source, prompt_ids in prepared:
        if source["id"] in cached:
            result = validate_prediction(cached[source["id"]], source, tokenizer, benchmark,
                                         max_prompt_tokens, max_new_tokens, model.config.vocab_size)
        else:
            response_ids = generate_response(model, tokenizer, prompt_ids, max_new_tokens,
                                             {"temperature": 0.0, "top_p": 1.0, "top_k": 0})
            prediction = tokenizer.decode(response_ids, skip_special_tokens=True)
            result = {"id": source["id"], "prompt": source["prompt"], "reference": source["response"],
                      "benchmark": benchmark, "prompt_ids": prompt_ids, "response_ids": response_ids,
                      "prediction": prediction, "generated_tokens": len(response_ids),
                      **math_score(prediction, source["response"], benchmark)}
            if "source_group" in source:
                result["source_group"] = source["source_group"]
            if on_prediction is not None:
                on_prediction(result)
        results.append(result)
    return summarize_predictions(results, benchmark)
