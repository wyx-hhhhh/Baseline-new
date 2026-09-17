#!/usr/bin/env python3
"""Evaluate one exported student on GSM8K or MATH with durable greedy pass@1."""
from __future__ import annotations

import argparse
from importlib.metadata import version as package_version
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_common.config import load_config, validate_config
from baseline_common.data import assert_disjoint, file_sha256, load_records
from baseline_common.math_evaluation import (evaluate_math_records, math_prompt,
    positive_integer, prepare_math_records, render_math_prompt_text,
    summarize_predictions, validate_prediction)
from baseline_common.math_metrics import metric_definition
from baseline_common.models import bind_vocabulary, load_model, load_tokenizer, model_fingerprint
from baseline_common.orchestration import model_export_complete
from baseline_common.pair_progress import append_jsonl, atomic_json, read_progress_rows
from scripts.evaluate import (EvaluationInterrupted, StopRequest, _digest, _require_identity,
    _signals, _validate_alignment, evaluation_lock, evaluation_paths)


def _runtime_identity(device):
    import torch
    selected = torch.device(device)
    runtime = {name: package_version(name) for name in ("torch", "transformers", "tokenizers", "numpy")}
    runtime.update(python=platform.python_version(), cuda=torch.version.cuda,
                   device_type=selected.type,
                   float32_matmul_precision=torch.get_float32_matmul_precision())
    if selected.type == "cuda":
        runtime["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return runtime


def _validate_predictions(rows, records, tokenizer, benchmark, prompt_budget, response_budget,
                          vocabulary_size, *, complete=False):
    if len(rows) > len(records):
        raise ValueError("Persisted math evaluation has too many records")
    for row, source in zip(rows, records):
        validate_prediction(row, source, tokenizer, benchmark, prompt_budget,
                            response_budget, vocabulary_size)
    if complete and len(rows) != len(records):
        raise ValueError(f"Incomplete math evaluation: {len(rows)}/{len(records)}")


def _metrics(rows, identity):
    return {**summarize_predictions(rows, identity["benchmark"]),
            **{key: identity[key] for key in ("metric_definition", "model", "dataset",
                                             "evaluation", "training_run")}}


def _validate_completed(paths, identity, records, tokenizer, vocabulary_size):
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Invalid math evaluation completion manifest")
    _require_identity(manifest.get("identity"), identity)
    if manifest.get("records") != len(records):
        raise ValueError("Completed math evaluation record count changed")
    for key in ("predictions", "metrics"):
        path = paths[key]
        if not path.is_file() or manifest.get("files", {}).get(path.name) != file_sha256(path):
            raise ValueError(f"Completed math artifact missing or checksum changed: {path}")
    rows = read_progress_rows(paths["predictions"])
    settings = identity["evaluation"]
    _validate_predictions(rows, records, tokenizer, identity["benchmark"],
                          settings["max_prompt_tokens"], settings["max_new_tokens"],
                          vocabulary_size, complete=True)
    if json.loads(paths["metrics"].read_text()) != _metrics(rows, identity):
        raise ValueError("Completed math metrics disagree with persisted predictions")


def run_evaluation(cfg, *, model_path, output, data_path, benchmark, device=None, dtype=None,
                   max_prompt_tokens=1024, max_new_tokens=2048, max_examples=None, resume=False):
    """Return True for verified completion, False after a graceful interruption.

    The training config identifies initialization and data only. Its natural
    language ROUGE settings do not control this separate math evaluator.
    """
    cfg = validate_config(cfg)
    training_config = dict(cfg)
    positive_integer(max_prompt_tokens, "max_prompt_tokens")
    positive_integer(max_new_tokens, "max_new_tokens")
    if max_examples is not None:
        positive_integer(max_examples, "max_examples")
    definition = metric_definition(benchmark)
    if dtype is not None:
        cfg = validate_config({**cfg, "dtype": dtype})
    device = device or cfg["device"]
    data_path = Path(data_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    paths = evaluation_paths(output)
    stop = StopRequest()
    with evaluation_lock(paths["lock"]), _signals(stop):
        existing = [path for key, path in paths.items() if key != "lock" and path.exists()]
        if existing and not resume:
            raise FileExistsError(f"Math evaluation artifact exists: {existing[0]}; pass --resume or use a new output directory")
        if not model_export_complete(model_path):
            raise ValueError(f"Model export is missing or incomplete: {model_path}")
        dataset_sha256 = file_sha256(data_path)
        train_sha256 = file_sha256(cfg["train_file"])
        records = prepare_math_records(load_records(data_path), benchmark)
        train_records = load_records(cfg["train_file"], paired=cfg["method"] == "distillm2")
        assert_disjoint(train_records, records)
        # A dev split must never be silently relabeled a test benchmark.
        dev_sha256 = file_sha256(cfg["validation_file"])
        dev_records = load_records(cfg["validation_file"])
        assert_disjoint(dev_records, records)
        total_examples = len(records)
        if max_examples is not None:
            records = records[:max_examples]
        tokenizer = load_tokenizer(model_path)
        vocabulary_size, alignment = _validate_alignment(model_path, tokenizer, cfg)
        # Fail before loading weights, and before partially evaluating a test
        # set, if any full question does not fit the declared prompt budget.
        for row in records:
            math_prompt(tokenizer, row["prompt"], max_prompt_tokens)
        identity = {
            "benchmark": benchmark, "model": model_fingerprint(model_path),
            "dataset": {"path": str(data_path), "sha256": dataset_sha256,
                        "available_examples": total_examples, "benchmark": benchmark},
            "training_data": {"path": str(Path(cfg["train_file"]).resolve()), "sha256": train_sha256},
            "development_data": {"path": str(Path(cfg["validation_file"]).resolve()), "sha256": dev_sha256},
            "training_config_sha256": _digest(training_config),
            "training_run": {key: cfg[key] for key in ("name", "pair", "method", "dataset", "seed")},
            "evaluation": {"max_prompt_tokens": max_prompt_tokens, "max_new_tokens": max_new_tokens,
                           "max_examples": max_examples, "enable_thinking": False,
                           "dtype": cfg["dtype"], "decoding": "greedy", "batch_size": 1,
                           "samples_per_question": 1, "prompt_instruction": render_math_prompt_text("{question}"),
                           "prompt_truncation": "forbidden", "pass_at_1_range": [0, 100],
                           "vocabulary_policy": cfg["vocabulary_policy"], "vocabulary_alignment": alignment},
            "metric_definition": definition, "runtime": _runtime_identity(device),
            "implementation_sha256": {name: file_sha256(ROOT / name) for name in (
                "baseline_common/math_evaluation.py", "baseline_common/math_metrics.py",
                "baseline_common/models.py", "baseline_common/vocabulary.py",
                "baseline_common/data.py", "baseline_common/pair_progress.py",
                "scripts/evaluate_math.py", "scripts/evaluate.py")},
        }

        def check_inputs():
            if (model_fingerprint(model_path) != identity["model"]
                    or file_sha256(data_path) != dataset_sha256
                    or file_sha256(cfg["train_file"]) != train_sha256
                    or file_sha256(cfg["validation_file"]) != dev_sha256):
                raise RuntimeError("Model or dataset changed during math evaluation; use a new output directory")

        if paths["manifest"].exists():
            _validate_completed(paths, identity, records, tokenizer, vocabulary_size)
            check_inputs()
            print(f"Validated {len(records)} existing {benchmark} predictions; already complete: {paths['metrics']}", flush=True)
            return True
        if paths["progress"].exists():
            progress = json.loads(paths["progress"].read_text())
            if progress.get("schema_version") != 1 or progress.get("records") != len(records):
                raise ValueError("Invalid math evaluation progress manifest")
            _require_identity(progress.get("identity"), identity)
        else:
            if existing:
                raise ValueError("Incomplete math artifacts lack a progress manifest; use a new output directory")
            atomic_json(paths["progress"], {"schema_version": 1, "records": len(records), "identity": identity})
        predictions = read_progress_rows(paths["predictions"], recover_tail=True)
        _validate_predictions(predictions, records, tokenizer, benchmark,
                              max_prompt_tokens, max_new_tokens, vocabulary_size)
        completed = len(predictions)
        if completed < len(records):
            if stop.signum is not None:
                return False
            print(f"Loading student {model_path} on {device}; {completed}/{len(records)} {benchmark} responses durable", flush=True)
            model = load_model(model_path, cfg["dtype"], device)
            bind_vocabulary(model, tokenizer)
            model.eval()
            model.requires_grad_(False)
            if stop.signum is not None:
                return False

            def persist(row):
                nonlocal completed
                append_jsonl(paths["predictions"], row)
                predictions.append(row)
                completed += 1
                if completed % 25 == 0 or completed == len(records):
                    print(f"Evaluated {benchmark} {completed}/{len(records)}", flush=True)
                if stop.signum is not None and completed < len(records):
                    raise EvaluationInterrupted

            try:
                evaluate_math_records(model, tokenizer, records, benchmark,
                    max_prompt_tokens=max_prompt_tokens, max_new_tokens=max_new_tokens,
                    on_prediction=persist, cached_predictions={row["id"]: row for row in predictions})
            except EvaluationInterrupted:
                print(f"Saved {completed}/{len(records)} responses. Rerun with --resume.", flush=True)
                return False
            finally:
                del model
        _validate_predictions(predictions, records, tokenizer, benchmark,
                              max_prompt_tokens, max_new_tokens, vocabulary_size, complete=True)
        check_inputs()
        metrics = _metrics(predictions, identity)
        atomic_json(paths["metrics"], metrics)
        atomic_json(paths["manifest"], {"schema_version": 1, "complete": True, "identity": identity,
            "records": len(records),
            "files": {paths[key].name: file_sha256(paths[key]) for key in ("predictions", "metrics")}})
        print(f"{benchmark} pass@1: {metrics['accuracy_percent']:.4f}% "
              f"({metrics['correct_count']}/{metrics['total']}); {paths['metrics']}", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved training configuration JSON")
    parser.add_argument("--model", required=True, help="Exported student directory, usually RUN/final")
    parser.add_argument("--data", required=True, help="Canonical held-out test JSONL")
    parser.add_argument("--benchmark", required=True, choices=("gsm8k", "math"))
    parser.add_argument("--output", required=True, help="Dedicated benchmark result directory")
    parser.add_argument("--device", help="Student inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", "--max-examples", dest="max_examples", type=int,
                        help="Explicit pilot subset; default scores every test question")
    parser.add_argument("--resume", action="store_true", help="Continue durable responses or validate completed output")
    args = parser.parse_args()
    try:
        complete = run_evaluation(load_config(args.config), model_path=args.model, output=args.output,
            data_path=args.data, benchmark=args.benchmark, device=args.device, dtype=args.dtype,
            max_prompt_tokens=args.max_prompt_tokens, max_new_tokens=args.max_new_tokens,
            max_examples=args.max_examples, resume=args.resume)
    except KeyboardInterrupt:
        print("Math evaluation interrupted; completed responses are durable.", file=sys.stderr)
        return 75
    return 0 if complete else 75


if __name__ == "__main__":
    raise SystemExit(main())
