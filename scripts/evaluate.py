#!/usr/bin/env python3
"""Evaluate an exported student on held-out responses with durable ROUGE-L progress."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import platform
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_common.config import load_config, validate_config
from baseline_common.data import assert_disjoint, file_sha256, load_records
from baseline_common.models import (bind_vocabulary, check_supported, load_model,
                                    load_tokenizer, model_fingerprint, render_prompt)
from baseline_common.orchestration import model_export_complete
from baseline_common.pair_progress import append_jsonl, atomic_json, read_progress_rows


class EvaluationInterrupted(Exception):
    """An interrupt was honored after the current response became durable."""


class StopRequest:
    def __init__(self):
        self.signum = None

    def receive(self, signum, _frame):
        if self.signum is not None:
            raise KeyboardInterrupt
        self.signum = signum
        print("Evaluation interruption requested; saving the current response before stopping.", flush=True)


@contextmanager
def _signals(stop):
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, stop.receive)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def evaluation_lock(path):
    """Keep the lock inode in place to exclude concurrent append writers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another evaluation holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def evaluation_paths(output):
    output = Path(output).expanduser().resolve()
    return {key: output / name for key, name in {
        "predictions": "predictions.jsonl", "metrics": "metrics.json",
        "progress": "progress.json", "manifest": "manifest.json", "lock": ".lock",
    }.items()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _runtime_identity(device):
    import torch
    selected = torch.device(device)
    runtime = {name: package_version(name) for name in
               ("torch", "transformers", "tokenizers", "numpy", "nltk", "rouge-score")}
    runtime.update(python=platform.python_version(), cuda=torch.version.cuda,
                   device_type=selected.type,
                   float32_matmul_precision=torch.get_float32_matmul_precision())
    if selected.type == "cuda":
        # This also runs when validating completed artifacts. Do not initialize
        # a GPU context or require available hardware before a cache hit.
        runtime["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return runtime


def _require_identity(actual, expected):
    if not isinstance(actual, dict):
        raise ValueError("Evaluation progress has no identity")
    changed = sorted(key for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key))
    if changed:
        raise ValueError(f"Evaluation inputs changed: {', '.join(changed)}; use a new output directory")


def _validate_alignment(model_path, tokenizer, cfg):
    model_config = json.loads((Path(model_path) / "config.json").read_text())
    saved_model = model_config.get("baseline_vocab_alignment")
    saved_tokenizer = getattr(tokenizer, "_baseline_vocab_alignment", None)
    if saved_model != saved_tokenizer:
        raise ValueError("Exported model/tokenizer vocabulary policies disagree")
    policy = (saved_tokenizer or {}).get("policy", "full")
    if policy != cfg["vocabulary_policy"]:
        raise ValueError("Saved vocabulary policy does not match the training configuration")
    if saved_tokenizer and saved_tokenizer.get("vocabulary_size") != model_config["vocab_size"]:
        raise ValueError("Saved vocabulary support does not match the model output size")
    return model_config["vocab_size"], saved_tokenizer


def _validate_predictions(rows, records, tokenizer, cfg, vocabulary_size, *, complete=False):
    from baseline_common.evaluation import rouge_l_score
    if len(rows) > len(records):
        raise ValueError("Persisted evaluation has too many records")
    for index, row in enumerate(rows):
        source = records[index]
        if row.get("id") != source["id"]:
            raise ValueError("Persisted evaluation IDs are duplicated, unknown, or out of order")
        for key, expected in (("prompt", source["prompt"]), ("reference", source["response"]),
                              ("source_group", source.get("source_group"))):
            if row.get(key) != expected:
                raise ValueError(f"Persisted evaluation {key} changed for {source['id']!r}")
        for key, limit in (("prompt_ids", cfg["max_prompt_tokens"]),
                           ("response_ids", cfg.get("eval_max_new_tokens") or cfg["max_new_tokens"])):
            ids = row.get(key)
            if (not isinstance(ids, list) or not ids or len(ids) > limit
                    or any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids)):
                raise ValueError(f"Invalid persisted evaluation {key} for {source['id']!r}")
            check_supported(ids, tokenizer, f"persisted evaluation {key}")
        prompt_ids = render_prompt(tokenizer, source["prompt"], cfg["max_prompt_tokens"], cfg["enable_thinking"])
        if row["prompt_ids"] != prompt_ids:
            raise ValueError(f"Persisted prompt token IDs changed for {source['id']!r}")
        prediction = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
        if row.get("prediction") != prediction:
            raise ValueError(f"Persisted evaluation prediction/token IDs disagree for {source['id']!r}")
        if type(row.get("generated_tokens")) is not int or row["generated_tokens"] != len(row["response_ids"]):
            raise ValueError(f"Persisted generated token count changed for {source['id']!r}")
        expected_score = rouge_l_score(prediction, source["response"], tokenizer=cfg.get("eval_rouge_tokenizer", "english"))
        score = row.get("rouge_l")
        if type(score) not in (int, float) or not math.isfinite(score) or abs(score - expected_score) > 1e-12:
            raise ValueError(f"Persisted evaluation score changed for {source['id']!r}")
    if complete and len(rows) != len(records):
        raise ValueError(f"Incomplete evaluation: {len(rows)}/{len(records)} records")


def _metrics(rows, identity):
    return {
        "rouge_l": math.fsum(row["rouge_l"] for row in rows) / len(rows),
        "examples": len(rows), "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "metric_definition": identity["metric_definition"], "model": identity["model"],
        "dataset": identity["dataset"], "evaluation": identity["evaluation"],
        "training_run": identity["training_run"],
    }


def _validate_completed(paths, identity, records, tokenizer, cfg, vocabulary_size):
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Invalid evaluation completion manifest")
    _require_identity(manifest.get("identity"), identity)
    if manifest.get("records") != len(records):
        raise ValueError("Completed evaluation record count changed")
    for key in ("predictions", "metrics"):
        path = paths[key]
        if not path.is_file() or manifest.get("files", {}).get(path.name) != file_sha256(path):
            raise ValueError(f"Completed evaluation artifact missing or checksum changed: {path}")
    rows = read_progress_rows(paths["predictions"])
    _validate_predictions(rows, records, tokenizer, cfg, vocabulary_size, complete=True)
    if json.loads(paths["metrics"].read_text()) != _metrics(rows, identity):
        raise ValueError("Completed evaluation metrics disagree with persisted predictions")


def run_evaluation(cfg, *, model_path, output, data_path=None, device=None, dtype=None,
                   max_prompt_tokens=None, max_new_tokens=None, max_examples=None,
                   rouge_tokenizer=None, resume=False):
    """Return True for completed/verified results and False for a safe interruption."""
    from baseline_common.evaluation import evaluate_records, metric_definition
    cfg = validate_config(cfg)
    if cfg["evaluation_metric"] != "rouge_l":
        raise ValueError("This evaluator reports ROUGE-L; use scripts/evaluate_math.py for math_accuracy")
    training_config = dict(cfg)
    if max_examples is not None and (type(max_examples) is not int or max_examples < 1):
        raise ValueError("max_examples must be a positive integer")
    if max_prompt_tokens is not None:
        cfg["max_prompt_tokens"] = max_prompt_tokens
    if max_new_tokens is not None:
        cfg["eval_max_new_tokens"] = max_new_tokens
    if dtype is not None:
        cfg["dtype"] = dtype
    if rouge_tokenizer is not None:
        cfg["eval_rouge_tokenizer"] = rouge_tokenizer
    cfg = validate_config(cfg)
    device = device or cfg["device"]
    data_path = Path(data_path or cfg["validation_file"]).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    paths = evaluation_paths(output)
    stop = StopRequest()
    with evaluation_lock(paths["lock"]), _signals(stop):
        existing = [path for key, path in paths.items() if key != "lock" and path.exists()]
        if existing and not resume:
            raise FileExistsError(f"Evaluation artifact exists: {existing[0]}; pass --resume or use a new output directory")
        if not model_export_complete(model_path):
            raise ValueError(f"Model export is missing or incomplete: {model_path}")
        dataset_sha256 = file_sha256(data_path)
        train_sha256 = file_sha256(cfg["train_file"])
        records = load_records(data_path)
        # Check the complete held-out input before applying a pilot limit.
        train_records = load_records(cfg["train_file"], paired=cfg["method"] == "distillm2")
        assert_disjoint(train_records, records)
        total_examples = len(records)
        if max_examples is not None:
            records = records[:max_examples]
        tokenizer = load_tokenizer(model_path)
        vocabulary_size, alignment = _validate_alignment(model_path, tokenizer, cfg)
        identity = {
            "model": model_fingerprint(model_path),
            "dataset": {"path": str(data_path), "sha256": dataset_sha256, "available_examples": total_examples},
            "training_data": {"path": str(Path(cfg["train_file"]).resolve()), "sha256": train_sha256},
            "training_config_sha256": _digest(training_config),
            "training_run": {key: cfg[key] for key in ("name", "pair", "method", "dataset", "seed")},
            "evaluation": {"max_prompt_tokens": cfg["max_prompt_tokens"],
                           "max_new_tokens": cfg.get("eval_max_new_tokens") or cfg["max_new_tokens"],
                           "max_examples": max_examples, "enable_thinking": cfg["enable_thinking"],
                           "dtype": cfg["dtype"], "decoding": "greedy", "batch_size": 1,
                           "vocabulary_policy": cfg["vocabulary_policy"], "vocabulary_alignment": alignment},
            "metric_definition": metric_definition(cfg.get("eval_rouge_tokenizer", "english")),
            "runtime": _runtime_identity(device),
            "implementation_sha256": {name: file_sha256(ROOT / "baseline_common" / name)
                                       for name in ("evaluation.py", "models.py", "vocabulary.py")},
        }

        def check_inputs():
            if (model_fingerprint(model_path) != identity["model"]
                    or file_sha256(data_path) != dataset_sha256
                    or file_sha256(cfg["train_file"]) != train_sha256):
                raise RuntimeError("Model or dataset changed during evaluation; use a new output directory")

        if paths["manifest"].exists():
            _validate_completed(paths, identity, records, tokenizer, cfg, vocabulary_size)
            check_inputs()
            print(f"Validated {len(records)} existing predictions; evaluation already complete: {paths['metrics']}", flush=True)
            return True
        if paths["progress"].exists():
            progress = json.loads(paths["progress"].read_text())
            if progress.get("schema_version") != 1 or progress.get("records") != len(records):
                raise ValueError("Invalid evaluation progress manifest")
            _require_identity(progress.get("identity"), identity)
        else:
            if existing:
                raise ValueError("Incomplete evaluation artifacts lack a progress manifest; use a new output directory")
            atomic_json(paths["progress"], {"schema_version": 1, "records": len(records), "identity": identity})
        predictions = read_progress_rows(paths["predictions"], recover_tail=True)
        _validate_predictions(predictions, records, tokenizer, cfg, vocabulary_size)
        completed = len(predictions)
        if completed < len(records):
            if stop.signum is not None:
                return False
            print(f"Loading student {model_path} on {device}; {completed}/{len(records)} responses already durable", flush=True)
            model = load_model(model_path, cfg["dtype"], device)
            bind_vocabulary(model, tokenizer)
            model.eval()
            model.requires_grad_(False)
            if stop.signum is not None:
                return False

            def persist(row):
                nonlocal completed
                source = records[completed]
                if "source_group" in source:
                    row = {**row, "source_group": source["source_group"]}
                append_jsonl(paths["predictions"], row)
                predictions.append(row)
                completed += 1
                if completed % 25 == 0 or completed == len(records):
                    print(f"Evaluated {completed}/{len(records)}", flush=True)
                if stop.signum is not None and completed < len(records):
                    raise EvaluationInterrupted

            try:
                evaluate_records(model, tokenizer, records, cfg, max_examples=None,
                                 on_prediction=persist,
                                 cached_predictions={row["id"]: row for row in predictions})
            except EvaluationInterrupted:
                print(f"Saved {completed}/{len(records)} responses. Rerun the same command with --resume.", flush=True)
                return False
            finally:
                del model
        _validate_predictions(predictions, records, tokenizer, cfg, vocabulary_size, complete=True)
        check_inputs()
        metrics = _metrics(predictions, identity)
        atomic_json(paths["metrics"], metrics)
        atomic_json(paths["manifest"], {
            "schema_version": 1, "complete": True, "identity": identity, "records": len(records),
            "files": {paths[key].name: file_sha256(paths[key]) for key in ("predictions", "metrics")},
        })
        print(f"ROUGE-L: {metrics['rouge_l']:.4f} ({metrics['examples']} examples); {paths['metrics']}", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved training configuration JSON")
    parser.add_argument("--model", required=True, help="Exported student directory, usually RUN/final")
    parser.add_argument("--output", required=True, help="Evaluation artifact directory")
    parser.add_argument("--data", help="Canonical held-out JSONL; defaults to config validation_file")
    parser.add_argument("--device", help="Student inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--rouge-tokenizer", choices=("english", "unicode"),
                        help="ROUGE word normalization; defaults to the training configuration")
    parser.add_argument("--limit", "--max-examples", dest="max_examples", type=int,
                        help="Explicit pilot subset; default evaluates every held-out example")
    parser.add_argument("--resume", action="store_true", help="Continue durable responses or verify completed output")
    args = parser.parse_args()
    try:
        complete = run_evaluation(load_config(args.config), model_path=args.model, output=args.output,
            data_path=args.data, device=args.device, dtype=args.dtype,
            max_prompt_tokens=args.max_prompt_tokens, max_new_tokens=args.max_new_tokens,
            max_examples=args.max_examples, rouge_tokenizer=args.rouge_tokenizer, resume=args.resume)
    except KeyboardInterrupt:
        print("Evaluation interrupted; previously completed responses are durable.", file=sys.stderr)
        return 75
    return 0 if complete else 75


if __name__ == "__main__":
    raise SystemExit(main())
