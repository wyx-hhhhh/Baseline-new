#!/usr/bin/env python3
"""Evaluate saved baselines on the five local natural-language benchmarks."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_common.config import load_config, validate_config
from baseline_common.data import assert_disjoint, file_sha256, load_records
from baseline_common.nl_data import prepare_nl_pool, load_nl_records, validate_nl_pool
from baseline_common.orchestration import METHOD_ORDER, PAIR_NAMES, Runner, Stage, is_training_complete, model_export_complete, _total_steps
from baseline_common.pair_progress import atomic_json, _atomic_lines
from scripts.run_parallel import gpu_pair


BENCHMARKS = ("dolly", "selfinst", "super_natural", "unnatural", "vicuna")
NAMES = dict(zip(BENCHMARKS, ("DollyEval", "SelfInst", "Super-Natural", "Unnatural", "VicunaEval")))


def benchmark_name(value):
    compact = value.lower().replace("-", "").replace("_", "")
    aliases = {key.replace("_", ""): key for key in BENCHMARKS}
    aliases.update({name.lower().replace("-", ""): key for key, name in NAMES.items()})
    if compact not in aliases:
        raise argparse.ArgumentTypeError(f"Unknown benchmark {value!r}; choose {', '.join(NAMES.values())}")
    return aliases[compact]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepared_directory(args):
    exclusions = [Path(args.data_root).expanduser().resolve() / args.dataset / "train.jsonl"]
    exclusions.extend(Path(path).expanduser().resolve() for path in args.exclude_training_data)
    exclusions = sorted(set(exclusions))
    request = {"source_root": str(Path(args.source_root).expanduser().resolve()),
               "benchmarks": args.benchmarks,
               "excluded_training": [{"path": str(path), "sha256": file_sha256(path)} for path in exclusions]}
    directory = (Path(args.prepared_data_dir).expanduser().resolve() if args.prepared_data_dir
                 else ROOT / "artifacts/nl_eval_data" / args.dataset / _digest(request)[:16])
    return directory, exclusions


def prepare_data(args):
    directory, exclusions = prepared_directory(args)
    # Independently launched GPU groups may create the same CPU data cache.
    # Serialize publication so the second group verifies the completed pool.
    directory.parent.mkdir(parents=True, exist_ok=True)
    with directory.with_name(directory.name + ".prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest = prepare_nl_pool(args.source_root, directory, benchmarks=args.benchmarks,
                                   exclude_files=exclusions, resume=True)
    return directory, manifest


def evaluation_protocol(args, pool):
    protocol = {"version": "nl-local-v1", "pool_manifest_sha256": file_sha256(pool / "manifest.json"),
                "max_prompt_tokens": args.max_prompt_tokens, "max_new_tokens": args.max_new_tokens,
                "rouge_tokenizer": args.rouge_tokenizer, "dtype_override": args.dtype,
                "device_type": args.device.split(":", 1)[0]}
    return "nl_" + _digest(protocol)[:16]


def build_evaluation_groups(args, pool):
    """Plan each benchmark separately using the original saved training config."""
    pool = Path(pool).resolve()
    validate_nl_pool(pool)
    output = Path(args.output_root).expanduser().resolve()
    evaluation_root = Path(args.evaluation_root or args.output_root).expanduser().resolve()
    protocol = evaluation_protocol(args, pool)
    state = evaluation_root / "evaluation_orchestration/nl" / args.dataset / protocol
    evaluation_data = {name: load_nl_records(pool / f"{name}.jsonl") for name in args.benchmarks}
    checked_training = set()
    groups = {}
    for group, pair in PAIR_NAMES.items():
        if args.group not in ("all", group):
            continue
        stages = []
        for method in args.methods:
            for seed in args.seeds:
                run = output / "runs" / pair / args.dataset / method / f"seed_{seed}"
                path = run / "manifest.json"
                if path.is_file():
                    manifest = json.loads(path.read_text())
                    cfg = validate_config(manifest["config"])
                    if (cfg["pair"], cfg["method"], cfg["dataset"], cfg["seed"]) != (pair, method, args.dataset, seed):
                        raise ValueError(f"Training manifest identity disagrees with its run path: {run}")
                    if not args.dry_run:
                        if not is_training_complete(run, _total_steps(cfg, manifest.get("world_size", 1))):
                            raise ValueError(f"Training is incomplete: {run}. Finish training or select completed --methods / --group.")
                        train = Path(cfg["train_file"]).resolve()
                        if file_sha256(train) != manifest["train_sha256"]:
                            raise ValueError(f"Training data changed since this model was trained: {train}")
                        key = (str(train), manifest["train_sha256"], method == "distillm2")
                        if key not in checked_training:
                            records = load_records(train, paired=method == "distillm2")
                            for benchmark, heldout in evaluation_data.items():
                                try:
                                    assert_disjoint(records, heldout)
                                except ValueError as error:
                                    raise ValueError(f"{NAMES[benchmark]} overlaps {train}; add the original canonical training file with --exclude-training-data and prepare a new pool") from error
                            checked_training.add(key)
                elif args.dry_run:
                    cfg = load_config(ROOT / "configs/experiments" / f"{pair}_{method}.json")
                    data = Path(args.data_root).expanduser().resolve() / args.dataset
                    cfg.update(output_root=str(output), dataset=args.dataset, seed=seed,
                               train_file=str(data / (f"{pair}/pairs.train.jsonl" if method == "distillm2" else "train.jsonl")),
                               validation_file=str(data / "validation.jsonl"))
                    cfg = validate_config(cfg)
                else:
                    raise FileNotFoundError(f"No training result at {run}. Finish training or select completed --methods / --group.")
                if cfg["evaluation_metric"] != "rouge_l":
                    raise ValueError("Math runs use scripts/evaluate_math_all.sh; this campaign reports natural-language ROUGE-L")
                model = run / "final"
                if not args.dry_run and not model_export_complete(model):
                    raise ValueError(f"Final student export is incomplete: {model}")
                for benchmark in args.benchmarks:
                    label = benchmark if args.limit is None else f"{benchmark}_first_{args.limit}"
                    destination = evaluation_root / "evaluations/nl" / args.dataset / protocol / pair / method / f"seed_{seed}" / label
                    stages.append(Stage(group, "evaluate", f"{method}_{benchmark}", cfg,
                        state / "configs" / f"{pair}_{method}_seed_{seed}_{label}.json",
                        destination, pool / f"{benchmark}.jsonl", model))
        groups[group] = stages
    return groups, state


def evaluation_command(stage, args, pool):
    argv = [sys.executable, str(ROOT / "scripts/evaluate_nl.py"), "--config", str(stage.config_path),
            "--model", str(stage.model_path), "--data", str(stage.input_path),
            "--data-manifest", str(Path(pool) / "manifest.json"), "--benchmark", stage.input_path.stem,
            "--output", str(stage.output_path), "--device", args.device,
            "--max-prompt-tokens", str(args.max_prompt_tokens), "--max-new-tokens", str(args.max_new_tokens),
            "--rouge-tokenizer", args.rouge_tokenizer, "--resume"]
    if args.dtype is not None:
        argv += ["--dtype", args.dtype]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    return argv


def summary_destination(args, pool):
    selection = {"group": args.group, "methods": args.methods, "seeds": args.seeds,
                 "benchmarks": args.benchmarks, "limit": args.limit}
    suffix = "" if (args.group == "all" and args.methods == list(METHOD_ORDER) and args.seeds == [42]
                     and args.benchmarks == list(BENCHMARKS) and args.limit is None) else "_" + args.group + "_" + _digest(selection)[:12]
    return (Path(args.evaluation_root or args.output_root).expanduser().resolve() / "evaluations/nl" /
            args.dataset / evaluation_protocol(args, pool) / f"summary{suffix}")


def write_summary(groups, destination):
    rows = []
    for stages in groups.values():
        for stage in stages:
            metrics = json.loads((stage.output_path / "metrics.json").read_text())
            rows.append({"pair": stage.config["pair"], "training_dataset": stage.config["dataset"],
                         "method": stage.config["method"], "seed": stage.config["seed"],
                         "benchmark": stage.input_path.stem, "rouge_l": metrics["rouge_l"],
                         "examples": metrics["examples"], "generated_tokens": metrics["generated_tokens"],
                         "model": str(stage.model_path), "metrics": str(stage.output_path / "metrics.json")})
    if not rows:
        raise ValueError("There are no completed natural-language evaluations to summarize")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination.with_suffix(".json"), {"metric": "rouge_l", "range": [0, 100],
        "benchmarks_separate": True, "reference_aggregation": "maximum_f1", "results": rows})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_lines(destination.with_suffix(".csv"), [stream.getvalue()])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/nas/Users/wyx/Baseline", help="Existing trained run root")
    parser.add_argument("--evaluation-root", help="Result root; defaults to --output-root")
    parser.add_argument("--dataset", "--training-dataset", default="dolly", help="Dataset used to train the saved models")
    parser.add_argument("--data-root", default="/nas/Users/wyx/Baseline/data", help="Canonical training data root for overlap exclusion")
    parser.add_argument("--source-root", default="/nas/Datasets", help="Parent of the five downloaded evaluation directories")
    parser.add_argument("--prepared-data-dir", help="Override the immutable prepared evaluation cache directory")
    parser.add_argument("--exclude-training-data", action="append", default=[], help="Additional canonical training JSONL to exclude (repeatable)")
    parser.add_argument("--benchmarks", type=benchmark_name, choices=BENCHMARKS, nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--group", choices=("all", "qwen", "llama"), default="all")
    parser.add_argument("--methods", choices=METHOD_ORDER, nargs="+", default=list(METHOD_ORDER))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--checkpoint", choices=("final",), default="final",
                        help="Final students; legacy best-on-validation evaluation uses --validation-only")
    parser.add_argument("--qwen-gpus", type=gpu_pair, default=("0", "1"))
    parser.add_argument("--llama-gpus", type=gpu_pair, default=("2", "3"))
    parser.add_argument("--device", default="cuda:0", help="Logical student inference device within each group")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--rouge-tokenizer", choices=("english", "unicode"), default="unicode",
                        help="Unicode ROUGE-L by default, preserving non-English references")
    parser.add_argument("--limit", type=int, help="Explicit per-model, per-benchmark pilot limit")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and verify local data without requiring trained models")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/verify data and print the queue without model loading or GPU work")
    args = parser.parse_args()
    for name in ("benchmarks", "methods", "seeds"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"{name} must be unique")
    if args.group == "all" and set(args.qwen_gpus) & set(args.llama_gpus):
        parser.error("The GPU groups must not overlap")
    if min(args.max_prompt_tokens, args.max_new_tokens) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Token budgets and --limit must be positive")
    pool, manifest = prepare_data(args)
    print(f"Prepared ROUGE-L data: {pool}\nTraining overlaps excluded; exact duplicate records merged.", flush=True)
    for benchmark in args.benchmarks:
        records = load_nl_records(pool / f"{benchmark}.jsonl")
        print(f"  {NAMES[benchmark]}: {len(records)} held-out examples", flush=True)
    if args.prepare_only:
        return 0
    groups, state = build_evaluation_groups(args, pool)
    gpus = {group: getattr(args, f"{group}_gpus") for group in groups}
    for group, stages in groups.items():
        print(f"{group}: {args.device} within GPUs {','.join(gpus[group])}; {len(stages)} evaluations", flush=True)
        for stage in stages:
            print(f"  {stage.method}, seed {stage.config['seed']} -> {stage.output_path}", flush=True)
    if args.dry_run:
        return 0
    result = Runner(groups, gpus, state,
                    command_builder=lambda stage, resume: evaluation_command(stage, args, pool)).run()
    if result == 0:
        destination = summary_destination(args, pool)
        write_summary(groups, destination)
        print(f"Separate benchmark ROUGE-L results: {destination.with_suffix('.csv')}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
