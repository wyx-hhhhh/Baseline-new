#!/usr/bin/env python3
"""Evaluate all saved baselines with ROUGE-L, in two resumable family queues."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_common.config import load_config, validate_config, run_directory
from baseline_common.data import file_sha256
from baseline_common.orchestration import (METHOD_ORDER, PAIR_NAMES, Runner, Stage,
                                           atomic_json, is_training_complete, model_export_complete,
                                           _checkpoint_info, _total_steps)
from scripts.run_parallel import gpu_pair


def build_evaluation_groups(args):
    output = Path(args.output_root).expanduser().resolve()
    evaluation_root = Path(getattr(args, "evaluation_root", None) or output).expanduser().resolve()
    data = Path(args.data_root).expanduser().resolve() / args.dataset
    state = evaluation_root / "evaluation_orchestration" / args.dataset
    groups = {}
    for group, pair in PAIR_NAMES.items():
        stages = []
        for method in args.methods:
            for seed in args.seeds:
                run = output / "runs" / pair / args.dataset / method / f"seed_{seed}"
                manifest_file = run / "manifest.json"
                if manifest_file.is_file():
                    manifest = json.loads(manifest_file.read_text())
                    cfg = validate_config(manifest["config"])
                    if (cfg["pair"], cfg["method"], cfg["dataset"], cfg["seed"]) != (pair, method, args.dataset, seed):
                        raise ValueError(f"Training manifest identity disagrees with run path: {run}")
                    if not args.dry_run:
                        if not is_training_complete(run, _total_steps(cfg, manifest.get("world_size", 1))):
                            raise ValueError(f"Training is incomplete; finish or resume the training command first: {run}")
                        if file_sha256(cfg["train_file"]) != manifest["train_sha256"]:
                            raise ValueError(f"Recorded training data changed; cannot establish held-out evaluation: {run}")
                elif args.dry_run:
                    cfg = load_config(ROOT / "configs/experiments" / f"{pair}_{method}.json")
                    cfg.update(output_root=str(output), dataset=args.dataset, seed=seed,
                               train_file=str(data / (f"{pair}/pairs.train.jsonl" if method == "distillm2" else "train.jsonl")),
                               validation_file=str(data / "validation.jsonl"))
                else:
                    raise FileNotFoundError(f"Training result missing; run scripts/run_all.sh first: {run}")
                model = run / "final"
                if args.checkpoint == "best":
                    best_file = run / "best_checkpoint.json"
                    if best_file.is_file():
                        best = json.loads(best_file.read_text())
                        if "rouge_l" not in best:
                            raise ValueError("The stored best checkpoint was not selected using ROUGE-L")
                        model = (run / best["path"]).resolve()
                        if not model.is_relative_to((run / "checkpoints").resolve()):
                            raise ValueError("Best checkpoint path is outside this training run")
                        if (type(best["rouge_l"]) not in (int, float) or not math.isfinite(best["rouge_l"])
                                or not 0 <= best["rouge_l"] <= 100):
                            raise ValueError("Best checkpoint has an invalid ROUGE-L score")
                        committed = _checkpoint_info(model.parent)
                        if committed is None or committed["step"] != best.get("step"):
                            raise ValueError("Best checkpoint is not a committed step matching its selection record")
                        if manifest.get("selection_metric") != "rouge_l":
                            raise ValueError("Training did not select its best checkpoint using ROUGE-L")
                    elif args.dry_run:
                        model = run / "checkpoints" / "BEST_ROUGE_L" / "student"
                    else:
                        raise ValueError("No ROUGE-L best checkpoint; default training-only runs should use --checkpoint final")
                if not args.dry_run and not model_export_complete(model):
                    raise ValueError(f"Selected model export is incomplete: {model}")
                if cfg["evaluation_metric"] != "rouge_l":
                    raise ValueError("Math runs require scripts/evaluate_math_all.sh for separate GSM8K/MATH accuracy")
                cfg.update(output_root=str(output), device=args.device, evaluation_metric="rouge_l")
                for key, value in (("dtype", args.dtype), ("eval_max_new_tokens", args.max_new_tokens),
                                   ("max_prompt_tokens", args.max_prompt_tokens), ("eval_rouge_tokenizer", args.rouge_tokenizer)):
                    if value is not None:
                        cfg[key] = value
                cfg = validate_config(cfg)
                evaluation = evaluation_root / "evaluations" / pair / args.dataset / method / f"seed_{seed}" / args.checkpoint
                if args.limit is not None:
                    evaluation = evaluation.with_name(f"{args.checkpoint}_first_{args.limit}")
                stages.append(Stage(group, "evaluate", method, cfg,
                             state / "configs" / f"{pair}_{method}_seed_{seed}_{evaluation.name}.json",
                             evaluation, Path(cfg["validation_file"]), model))
        groups[group] = stages
    return groups, state


def write_summary(groups, destination):
    rows = []
    for group, stages in groups.items():
        for stage in stages:
            metrics = json.loads((stage.output_path / "metrics.json").read_text())
            rows.append({"pair": stage.config["pair"], "dataset": stage.config["dataset"],
                         "method": stage.method, "seed": stage.config["seed"],
                         "rouge_l": metrics["rouge_l"], "examples": metrics["examples"],
                         "generated_tokens": metrics["generated_tokens"], "model": str(stage.model_path),
                         "metrics": str(stage.output_path / "metrics.json")})
    atomic_json(destination.with_suffix(".json"), {"metric": "rouge_l", "range": [0, 100],
                "aggregation": "arithmetic_mean_over_examples", "higher_is_better": True, "results": rows})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    target = destination.with_suffix(".csv")
    temporary = target.with_suffix(".csv.tmp")
    temporary.write_text(output.getvalue())
    temporary.replace(target)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/nas/Users/wyx/Baseline")
    parser.add_argument("--evaluation-root", help="Separate result root for a new evaluation protocol; defaults to --output-root")
    parser.add_argument("--data-root", default="/nas/Users/wyx/Baseline/data")
    parser.add_argument("--dataset", default="dolly")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--checkpoint", choices=("final", "best"), default="final")
    parser.add_argument("--qwen-gpus", type=gpu_pair, default=("0", "1"))
    parser.add_argument("--llama-gpus", type=gpu_pair, default=("2", "3"))
    parser.add_argument("--device", default="cuda:0", help="Logical device inside each two-GPU group")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--limit", type=int, help="Explicit per-model subset; default scores all held-out rows")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--rouge-tokenizer", choices=("english", "unicode"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Methods and seeds must be unique")
    if set(args.qwen_gpus) & set(args.llama_gpus):
        parser.error("The GPU groups must not overlap")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    groups, state = build_evaluation_groups(args)
    gpus = {"qwen": args.qwen_gpus, "llama": args.llama_gpus}
    for group, stages in groups.items():
        print(f"{group}: inference on {args.device} within GPUs {','.join(gpus[group])}", flush=True)
        for stage in stages:
            print(f"  ROUGE-L: {stage.method}, seed {stage.config['seed']} -> {stage.output_path}", flush=True)
    if args.dry_run:
        return 0
    def command(stage, resume):
        argv = Runner._command(stage, resume)
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        return argv
    result = Runner(groups, gpus, state, command_builder=command).run()
    if result == 0:
        suffix = f"_first_{args.limit}" if args.limit else ""
        destination = Path(args.evaluation_root or args.output_root).expanduser().resolve() / "evaluations" / args.dataset / f"summary_{args.checkpoint}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = write_summary(groups, destination)
        print(f"ROUGE-L complete for {len(rows)} runs: {destination.with_suffix('.csv')}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
