#!/usr/bin/env python3
"""Evaluate an exported student on a local natural-language test benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_common.config import load_config, validate_config
from baseline_common.data import assert_disjoint, file_sha256, load_records
from baseline_common.models import bind_vocabulary, load_model, load_tokenizer, model_fingerprint
from baseline_common.nl_data import BENCHMARKS, load_nl_records, validate_nl_pool
from baseline_common.nl_evaluation import (evaluate_nl_records, metric_definition, nl_prompt,
    positive_integer, prepare_nl_records, summarize_predictions, validate_prediction)
from baseline_common.orchestration import model_export_complete
from baseline_common.pair_progress import append_jsonl, atomic_json, read_progress_rows
from scripts.evaluate import (EvaluationInterrupted, StopRequest, _digest, _require_identity,
    _runtime_identity, _signals, _validate_alignment, evaluation_lock, evaluation_paths)


def _data_provenance(data_path, benchmark, manifest_path):
    """Bind canonical rows to the checked local source and exclusion artifacts."""
    if manifest_path is None:
        candidate = data_path.parent / "manifest.json"
        manifest_path = candidate if candidate.is_file() else None
    if manifest_path is None:
        return None
    manifest_path = Path(manifest_path).expanduser().resolve()
    if manifest_path.name != "manifest.json":
        raise ValueError("--data-manifest must name the prepared natural-language pool's manifest.json")
    manifest = validate_nl_pool(manifest_path.parent)
    entry = manifest.get("benchmarks", {}).get(benchmark)
    if not entry or (manifest_path.parent / entry["path"]).resolve() != data_path:
        raise ValueError("Evaluation benchmark/data path disagrees with --data-manifest")
    if file_sha256(data_path) != entry["sha256"]:
        raise ValueError("Evaluation data checksum disagrees with --data-manifest")
    return {"path": str(manifest_path), "sha256": file_sha256(manifest_path),
            "source_files": manifest["source_files"], "exclusion_files": manifest["exclusion_files"],
            "benchmark": {key: entry[key] for key in ("path", "sha256", "records", "ids_sha256",
                "source_records", "exact_duplicates_merged", "training_overlaps_removed",
                "overlap_source_rows_removed") if key in entry}}


def _validate_predictions(rows, records, tokenizer, identity, vocabulary_size, *, complete=False):
    if len(rows) > len(records):
        raise ValueError("Persisted evaluation has too many records")
    settings = identity["evaluation"]
    for row, source in zip(rows, records):
        validate_prediction(row, source, tokenizer, identity["benchmark"], settings["max_prompt_tokens"],
                            settings["max_new_tokens"], vocabulary_size, identity["metric_definition"]["tokenizer"],
                            settings["enable_thinking"])
    if complete and len(rows) != len(records):
        raise ValueError(f"Incomplete evaluation: {len(rows)}/{len(records)}")


def _metrics(rows, identity):
    return {**summarize_predictions(rows, identity["benchmark"]),
            **{key: identity[key] for key in ("metric_definition", "model", "dataset",
                                             "evaluation", "training_run")}}


def _validate_completed(paths, identity, records, tokenizer, vocabulary_size):
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
    _validate_predictions(rows, records, tokenizer, identity, vocabulary_size, complete=True)
    if json.loads(paths["metrics"].read_text()) != _metrics(rows, identity):
        raise ValueError("Completed evaluation metrics disagree with persisted predictions")


def run_evaluation(cfg, *, model_path, output, data_path, benchmark, data_manifest=None,
                   device=None, dtype=None, max_prompt_tokens=None, max_new_tokens=None,
                   max_examples=None, rouge_tokenizer=None, resume=False):
    """Return True on verified completion, False after a durable interruption."""
    cfg = validate_config(cfg)
    if cfg["evaluation_metric"] != "rouge_l":
        raise ValueError("Natural-language evaluation requires a ROUGE-L training configuration")
    training_config = dict(cfg)
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unknown natural-language benchmark {benchmark!r}; choose from {BENCHMARKS}")
    if max_examples is not None:
        positive_integer(max_examples, "max_examples")
    max_prompt_tokens = positive_integer(
        cfg["max_prompt_tokens"] if max_prompt_tokens is None else max_prompt_tokens, "max_prompt_tokens")
    max_new_tokens = positive_integer(
        (cfg.get("eval_max_new_tokens") or cfg["max_new_tokens"]) if max_new_tokens is None else max_new_tokens,
        "max_new_tokens")
    rouge_tokenizer = cfg.get("eval_rouge_tokenizer", "english") if rouge_tokenizer is None else rouge_tokenizer
    definition = metric_definition(rouge_tokenizer)
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
            raise FileExistsError(f"Evaluation artifact exists: {existing[0]}; pass --resume or use a new output directory")
        if not model_export_complete(model_path):
            raise ValueError(f"Model export is missing or incomplete: {model_path}")
        provenance = _data_provenance(data_path, benchmark, data_manifest)
        dataset_sha256 = file_sha256(data_path)
        train_sha256 = file_sha256(cfg["train_file"])
        records = prepare_nl_records(load_nl_records(data_path), benchmark, rouge_tokenizer)
        train_records = load_records(cfg["train_file"], paired=cfg["method"] == "distillm2")
        # Validate all references and training disjointness before a pilot limit.
        assert_disjoint(train_records, records)
        total_examples = len(records)
        tokenizer = load_tokenizer(model_path)
        vocabulary_size, alignment = _validate_alignment(model_path, tokenizer, cfg)
        model_config = json.loads((model_path / "config.json").read_text())
        context_limit = model_config.get("max_position_embeddings")
        for row in records:
            ids = nl_prompt(tokenizer, row["prompt"], max_prompt_tokens, cfg["enable_thinking"])
            if type(context_limit) is int and context_limit > 0 and len(ids) + max_new_tokens > context_limit:
                raise ValueError(f"Prompt + response ({len(ids)} + {max_new_tokens}) exceeds model context "
                                 f"({context_limit}) for {row['id']!r}; reduce --max-new-tokens or use a "
                                 "model with a larger context")
        if max_examples is not None:
            records = records[:max_examples]
        identity = {
            "benchmark": benchmark, "model": model_fingerprint(model_path),
            "dataset": {"path": str(data_path), "sha256": dataset_sha256,
                        "available_examples": total_examples, "benchmark": benchmark,
                        "source_manifest": provenance},
            "training_data": {"path": str(Path(cfg["train_file"]).resolve()), "sha256": train_sha256},
            "training_config_sha256": _digest(training_config),
            "training_run": {key: cfg[key] for key in ("name", "pair", "method", "dataset", "seed")},
            "evaluation": {"max_prompt_tokens": max_prompt_tokens, "max_new_tokens": max_new_tokens,
                           "max_examples": max_examples, "enable_thinking": cfg["enable_thinking"],
                           "dtype": cfg["dtype"], "decoding": "greedy", "batch_size": 1,
                           "samples_per_question": 1, "prompt_instruction": "canonical prompt unchanged",
                           "prompt_truncation": "forbidden", "reference_truncation": "forbidden",
                           "vocabulary_policy": cfg["vocabulary_policy"], "vocabulary_alignment": alignment},
            "metric_definition": definition, "runtime": _runtime_identity(device),
            "implementation_sha256": {name: file_sha256(ROOT / name) for name in (
                "baseline_common/nl_evaluation.py", "baseline_common/nl_data.py", "baseline_common/evaluation.py",
                "baseline_common/models.py", "baseline_common/vocabulary.py", "baseline_common/config.py",
                "baseline_common/data.py", "baseline_common/pair_progress.py",
                "scripts/evaluate_nl.py", "scripts/evaluate.py")},
        }

        def check_inputs():
            if (model_fingerprint(model_path) != identity["model"]
                    or file_sha256(data_path) != dataset_sha256
                    or file_sha256(cfg["train_file"]) != train_sha256
                    or _data_provenance(data_path, benchmark, data_manifest) != provenance):
                raise RuntimeError("Model or dataset changed during evaluation; use a new output directory")

        if paths["manifest"].exists():
            _validate_completed(paths, identity, records, tokenizer, vocabulary_size)
            check_inputs()
            print(f"Validated {len(records)} existing {benchmark} predictions; already complete: {paths['metrics']}", flush=True)
            return True
        if paths["progress"].exists():
            progress = json.loads(paths["progress"].read_text())
            if progress.get("schema_version") != 1 or progress.get("records") != len(records):
                raise ValueError("Invalid evaluation progress manifest")
            _require_identity(progress.get("identity"), identity)
        else:
            if existing:
                raise ValueError("Incomplete artifacts lack a progress manifest; use a new output directory")
            atomic_json(paths["progress"], {"schema_version": 1, "records": len(records), "identity": identity})
        predictions = read_progress_rows(paths["predictions"], recover_tail=True)
        _validate_predictions(predictions, records, tokenizer, identity, vocabulary_size)
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
                evaluate_nl_records(model, tokenizer, records, benchmark,
                    max_prompt_tokens=max_prompt_tokens, max_new_tokens=max_new_tokens,
                    rouge_tokenizer=rouge_tokenizer, enable_thinking=cfg["enable_thinking"],
                    on_prediction=persist, cached_predictions={row["id"]: row for row in predictions})
            except EvaluationInterrupted:
                print(f"Saved {completed}/{len(records)} responses. Rerun with --resume.", flush=True)
                return False
            finally:
                del model
        _validate_predictions(predictions, records, tokenizer, identity, vocabulary_size, complete=True)
        check_inputs()
        metrics = _metrics(predictions, identity)
        atomic_json(paths["metrics"], metrics)
        atomic_json(paths["manifest"], {"schema_version": 1, "complete": True, "identity": identity,
            "records": len(records),
            "files": {paths[key].name: file_sha256(paths[key]) for key in ("predictions", "metrics")}})
        print(f"{benchmark} ROUGE-L: {metrics['rouge_l']:.4f}; {paths['metrics']}", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved training configuration JSON")
    parser.add_argument("--model", required=True, help="Exported student directory, usually RUN/final")
    parser.add_argument("--data", required=True, help="Canonical natural-language benchmark JSONL")
    parser.add_argument("--data-manifest", help="Prepared pool manifest.json; defaults to the data directory's manifest")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--output", required=True, help="Dedicated benchmark result directory")
    parser.add_argument("--device", help="Student inference device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--max-prompt-tokens", type=int, help="Full input budget; never silently truncates")
    parser.add_argument("--max-new-tokens", type=int, help="Response generation budget; references stay complete")
    parser.add_argument("--rouge-tokenizer", choices=("english", "unicode"), default="unicode",
                        help="ROUGE tokenization; default Unicode keeps non-English references scoreable")
    parser.add_argument("--limit", "--max-examples", dest="max_examples", type=int,
                        help="Explicit pilot subset; default scores every test example")
    parser.add_argument("--resume", action="store_true", help="Continue durable responses or validate completed output")
    args = parser.parse_args()
    try:
        complete = run_evaluation(load_config(args.config), model_path=args.model, output=args.output,
            data_path=args.data, benchmark=args.benchmark, data_manifest=args.data_manifest,
            device=args.device, dtype=args.dtype, max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens, max_examples=args.max_examples,
            rouge_tokenizer=args.rouge_tokenizer, resume=args.resume)
    except KeyboardInterrupt:
        print("Evaluation interrupted; completed responses are durable.", file=sys.stderr)
        return 75
    return 0 if complete else 75


if __name__ == "__main__":
    raise SystemExit(main())
