#!/usr/bin/env python3
"""Fixed MetaMathQA training and separate full GSM8K/MATH benchmark queues."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_common.config import load_config, validate_config, run_directory
from baseline_common.data import file_sha256
from baseline_common.math_data import validate_math_pool
from baseline_common.orchestration import (METHOD_ORDER, PAIR_NAMES, Runner, Stage,
    atomic_json, is_training_complete, _total_steps)
from baseline_common.pair_progress import _atomic_lines
from scripts.run_parallel import gpu_pair


SHARED_CONTROLS = ("dtype", "optimizer_offload", "gradient_checkpointing", "learning_rate",
    "weight_decay", "gradient_accumulation_steps", "epochs", "max_steps", "warmup_ratio",
    "max_grad_norm", "max_prompt_tokens", "max_new_tokens", "eval_during_training", "evaluation_metric")
PAIR_CONTROLS = ("teacher_model", "student_model", "vocabulary_policy", "generation", "teacher_generation")


def _configuration(campaign, group, method):
    path = Path(campaign["config_files"][group][method])
    return load_config(path if path.is_absolute() else ROOT / path)


def validate_common_controls(campaign):
    common = None
    for group, pair in PAIR_NAMES.items():
        files = campaign["config_files"].get(group, {})
        if set(files) != set(METHOD_ORDER):
            raise ValueError("Every math model group must define all four baselines")
        pair_values = None
        for method in METHOD_ORDER:
            cfg = _configuration(campaign, group, method)
            if (cfg["pair"], cfg["method"], cfg["dataset"]) != (pair, method, campaign["dataset"]):
                raise ValueError("Math configuration identity differs from the frozen campaign")
            if cfg["evaluation_metric"] != "math_accuracy" or cfg["eval_during_training"]:
                raise ValueError("Math training must use separate final-answer evaluation")
            values = {key: cfg[key] for key in SHARED_CONTROLS}
            if common is not None and values != common:
                raise ValueError("All math baselines must share training precision, optimizer and token/update budgets")
            common = values
            values = {key: cfg[key] for key in PAIR_CONTROLS}
            if pair_values is not None and values != pair_values:
                raise ValueError("Math baselines within a model group must share original models and generation settings")
            pair_values = values


def load_campaign(path):
    campaign = json.loads(Path(path).read_text())
    if campaign.get("schema_version") != 1 or set(campaign.get("config_files", {})) != set(PAIR_NAMES):
        raise ValueError("Invalid math campaign configuration")
    if campaign.get("evaluation", {}).get("benchmarks") != ["gsm8k", "math"]:
        raise ValueError("The math campaign must evaluate GSM8K and MATH separately")
    return campaign


def check_pool(campaign, pool_dir=None):
    pool = Path(pool_dir or campaign["pool_dir"]).expanduser().resolve()
    if file_sha256(pool / "manifest.json") != campaign["pool_manifest_sha256"]:
        raise ValueError("Fixed math pool manifest changed; do not mix a different pool into this campaign")
    manifest = validate_math_pool(pool)
    for name, count in campaign["pool_counts"].items():
        if manifest["splits"][name]["records"] != count:
            raise ValueError(f"Math pool {name} count differs from the frozen campaign")
    return pool, manifest


def check_grader(campaign):
    grading = campaign.get("grading")
    if not grading or file_sha256(ROOT / "baseline_common/math_metrics.py") != grading.get("source_sha256"):
        raise ValueError("Math grader changed or is not pinned to the frozen campaign")
    audit = Path(grading["gold_audit"])
    if file_sha256(audit if audit.is_absolute() else ROOT / audit) != grading.get("gold_audit_sha256"):
        raise ValueError("Math gold-reference audit changed; revalidate the grader before evaluation")


def build_math_groups(args, campaign=None):
    campaign = campaign or load_campaign(args.campaign)
    validate_common_controls(campaign)
    pool, manifest = check_pool(campaign, getattr(args, "pool_dir", None))
    output = Path(getattr(args, "output_root", None) or campaign["output_root"]).expanduser().resolve()
    phase = args.phase
    if phase not in {"train", "evaluate"}:
        raise ValueError("Math phase must be train or evaluate")
    if phase == "evaluate":
        check_grader(campaign)
    state = output / ("orchestration" if phase == "train" else "evaluation_orchestration") / campaign["dataset"]
    groups = {}
    for group, pair in PAIR_NAMES.items():
        if getattr(args, "group", "all") not in {"all", group}:
            continue
        stages = []
        for method in args.methods:
            cfg = _configuration(campaign, group, method)
            if cfg["pair"] != pair or cfg["method"] != method or cfg["dataset"] != campaign["dataset"]:
                raise ValueError("Math configuration identity differs from the frozen campaign")
            if cfg["evaluation_metric"] != "math_accuracy" or cfg["eval_during_training"]:
                raise ValueError("Math training must use separate final-answer evaluation")
            cfg.update(output_root=str(output), validation_file=str(pool / manifest["splits"]["validation"]["path"]),
                train_file=str(output / "pairs" / pair / "pairs.train.jsonl") if method == "distillm2"
                           else str(pool / manifest["splits"]["train"]["path"]))
            if getattr(args, "device", None):
                cfg["device"] = args.device
                cfg["teacher_device"] = args.device if args.device == "cpu" else "cuda:1"
            if getattr(args, "dtype", None):
                cfg["dtype"] = args.dtype
            if phase == "train" and getattr(args, "max_steps", None) is not None:
                cfg["max_steps"] = args.max_steps
            if phase == "train" and method == "distillm2":
                producer = validate_config({**cfg, "seed": campaign["pair_seed"]})
                stages.append(Stage(group, "pairs", method, producer,
                    state / "configs" / f"{pair}_pairs_seed_{campaign['pair_seed']}.json",
                    Path(producer["train_file"]), pool / manifest["splits"]["train"]["path"]))
            for seed in args.seeds:
                current = validate_config({**cfg, "seed": seed})
                run = run_directory(current)
                if phase == "train":
                    stages.append(Stage(group, "train", method, current,
                        state / "configs" / f"{pair}_{method}_seed_{seed}.json", run))
                    continue
                if not args.dry_run:
                    manifest_file = run / "manifest.json"
                    if not manifest_file.is_file():
                        raise ValueError(f"Math training is not complete; missing training manifest: {run}")
                    previous = json.loads(manifest_file.read_text())
                    recorded = validate_config(previous["config"])
                    for key in ("pair", "method", "dataset", "seed", "name", "teacher_model", "student_model",
                                "vocabulary_policy", "evaluation_metric", "train_file", "validation_file"):
                        if recorded[key] != current[key]:
                            raise ValueError(f"Training manifest {key} differs from this fixed math campaign")
                    for key in (*[name for name in SHARED_CONTROLS if name not in {"max_steps", "dtype"}],
                                "generation", "teacher_generation"):
                        if recorded[key] != current[key]:
                            raise ValueError(f"Training manifest {key} differs from the shared math controls")
                    if not is_training_complete(run, _total_steps(recorded, previous.get("world_size", 1))):
                        raise ValueError(f"Math training is not complete: {run}")
                    if (file_sha256(recorded["train_file"]) != previous["train_sha256"]
                            or file_sha256(recorded["validation_file"]) != previous["validation_sha256"]):
                        raise ValueError("Math training/development files changed since training")
                    # Evaluation uses the actual recorded schedule, including a
                    # completed pilot, without requiring training flags again.
                    current = recorded
                    if getattr(args, "device", None):
                        current["device"] = args.device
                    if getattr(args, "dtype", None):
                        current["dtype"] = args.dtype
                for benchmark in campaign["evaluation"]["benchmarks"]:
                    output_name = benchmark if getattr(args, "limit", None) is None else f"{benchmark}_first_{args.limit}"
                    destination = output / "evaluations" / pair / campaign["dataset"] / method / f"seed_{seed}" / output_name
                    # Unique benchmark stage names keep logs/status from mixing.
                    stages.append(Stage(group, "evaluate", f"{method}_{benchmark}", current,
                        state / "configs" / f"{pair}_{method}_seed_{seed}_{output_name}.json",
                        destination, pool / manifest["splits"][benchmark]["path"], run / "final"))
        groups[group] = stages
    return groups, state


def math_command(stage, resume, campaign, *, limit=None, interpreter=None):
    python = interpreter or sys.executable
    if stage.kind != "evaluate":
        argv = Runner._command(stage, resume)
        argv[0] = python
        return argv
    benchmark = stage.method.rsplit("_", 1)[1]
    command = [python, str(ROOT / "scripts/evaluate_math.py"), "--config", str(stage.config_path),
        "--model", str(stage.model_path), "--data", str(stage.input_path), "--benchmark", benchmark,
        "--output", str(stage.output_path), "--device", stage.config["device"],
        "--max-prompt-tokens", str(campaign["evaluation"]["max_prompt_tokens"]),
        "--max-new-tokens", str(campaign["evaluation"]["max_new_tokens"]), "--resume"]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


class MathRunner(Runner):
    def __init__(self, groups, gpu_groups, state_dir, *, campaign, pool_dir=None, limit=None,
                 command_builder=None, **kwargs):
        self.campaign = campaign
        self.pool_dir = pool_dir
        super().__init__(groups, gpu_groups, state_dir,
            command_builder=command_builder or (lambda stage, resume: math_command(stage, resume, campaign, limit=limit)), **kwargs)

    def _prepare_training(self, stage):
        check_pool(self.campaign, self.pool_dir)
        return super()._prepare_training(stage)

    def _run_child(self, stage, resume):
        check_pool(self.campaign, self.pool_dir)
        if stage.kind == "evaluate":
            check_grader(self.campaign)
        return super()._run_child(stage, resume)


def write_math_summary(groups, path):
    rows = []
    for stages in groups.values():
        for stage in stages:
            metrics = json.loads((stage.output_path / "metrics.json").read_text())
            rows.append({"pair": stage.config["pair"], "method": stage.config["method"],
                "seed": stage.config["seed"], "benchmark": metrics["benchmark"],
                "accuracy_percent": metrics["accuracy_percent"], "pass_at_1": metrics["pass_at_1"],
                "correct_count": metrics["correct_count"], "total": metrics["total"],
                "unparseable_predictions": metrics["unparseable_predictions"], "model": str(stage.model_path),
                "metrics": str(stage.output_path / "metrics.json")})
    path = Path(path)
    atomic_json(path.with_suffix(".json"), {"metric": "math_accuracy", "range": [0, 100],
                "benchmarks_separate": True, "results": rows})
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_lines(path.with_suffix(".csv"), [text.getvalue()])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(ROOT / "configs/math_campaign.json"))
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument("--group", choices=("all", "qwen", "llama"), default="all")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--qwen-gpus", type=gpu_pair, default=("0", "1"))
    parser.add_argument("--llama-gpus", type=gpu_pair, default=("2", "3"))
    parser.add_argument("--output-root")
    parser.add_argument("--pool-dir")
    parser.add_argument("--device", help="Override logical device (CPU tests only for full-size models)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--max-steps", type=int, help="Explicit pilot schedule; use a separate output root")
    parser.add_argument("--limit", type=int, help="Explicit evaluation pilot subset; full tests by default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Methods and seeds must be unique")
    if args.group == "all" and set(args.qwen_gpus) & set(args.llama_gpus):
        parser.error("The GPU groups must not overlap")
    if args.limit is not None and (args.limit < 1 or args.phase != "evaluate"):
        parser.error("--limit is a positive evaluation-only option")
    if args.phase == "evaluate" and args.max_steps is not None:
        parser.error("--max-steps is for training; evaluation reads the completed run's recorded budget")
    campaign = load_campaign(args.campaign)
    groups, state = build_math_groups(args, campaign)
    gpu_groups = {group: getattr(args, f"{group}_gpus") for group in groups}
    print(f"Fixed pool: {args.pool_dir or campaign['pool_dir']}\nMath phase: {args.phase}", flush=True)
    for group, stages in groups.items():
        print(f"{group}: GPUs {','.join(gpu_groups[group])}", flush=True)
        for stage in stages:
            print(f"  {stage.kind}: {stage.method}, seed {stage.config['seed']} -> {stage.output_path}", flush=True)
    if args.dry_run:
        return 0
    code = MathRunner(groups, gpu_groups, state, campaign=campaign, pool_dir=args.pool_dir, limit=args.limit).run()
    if code == 0 and args.phase == "evaluate":
        root = Path(args.output_root or campaign["output_root"])
        selection = []
        if args.group != "all":
            selection.append(args.group)
        if args.methods != list(METHOD_ORDER):
            selection.append("methods_" + "_".join(args.methods))
        if args.seeds != [42]:
            selection.append("seeds_" + "_".join(map(str, args.seeds)))
        if args.limit:
            selection.append(f"first_{args.limit}")
        suffix = "_" + "__".join(selection) if selection else ""
        path = root / "evaluations" / campaign["dataset"] / f"summary{suffix}"
        write_math_summary(groups, path)
        print(f"Separate GSM8K/MATH results: {path.with_suffix('.csv')}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
