#!/usr/bin/env python3
"""Run Qwen and Llama concurrently, each on two GPUs, with automatic recovery."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_common.config import load_config, validate_config, run_directory
from baseline_common.orchestration import METHOD_ORDER, PAIR_NAMES, Runner, Stage


def gpu_pair(value):
    values = value.split(",")
    if len(values) != 2 or any(not v.isdigit() for v in values):
        raise argparse.ArgumentTypeError("Specify two distinct GPU indices, for example 0,1")
    result = tuple(str(int(v)) for v in values)
    if len(set(result)) != 2:
        raise argparse.ArgumentTypeError("Specify two distinct GPU indices, for example 0,1")
    return result


def build_groups(args):
    output = Path(args.output_root).expanduser().resolve()
    data = Path(args.data_root).expanduser().resolve() / args.dataset
    state = output / "orchestration" / args.dataset
    groups = {}
    for group, pair in PAIR_NAMES.items():
        if getattr(args, "group", "all") not in ("all", group):
            continue
        stages = []
        for method in args.methods:
            cfg = load_config(ROOT / "configs/experiments" / f"{pair}_{method}.json")
            cfg.update(output_root=str(output), dataset=args.dataset, device="cuda:0", teacher_device="cuda:1",
                       train_file=str(data / (f"{pair}/pairs.train.jsonl" if method == "distillm2" else "train.jsonl")),
                       validation_file=str(data / "validation.jsonl"))
            for key in ("max_steps", "save_steps", "eval_steps", "max_prompt_tokens", "max_new_tokens", "gradient_accumulation_steps",
                        "eval_during_training", "eval_max_new_tokens", "eval_max_examples", "eval_rouge_tokenizer"):
                value = getattr(args, key, None)
                if value is not None:
                    cfg[key] = value
            if method == "distillm2":
                producer = validate_config({**cfg, "seed": args.pair_seed})
                stages.append(Stage(group, "pairs", method, producer,
                         state / "configs" / f"{pair}_pairs_seed_{args.pair_seed}.json",
                         Path(producer["train_file"]), data / "train.jsonl"))
            for seed in args.seeds:
                resolved = validate_config({**cfg, "seed": seed})
                stages.append(Stage(group, "train", method, resolved,
                         state / "configs" / f"{pair}_{method}_seed_{seed}.json", run_directory(resolved)))
        groups[group] = stages
    return groups, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "qwen", "llama"), default="all",
                        help="Run both groups or select one for a separate terminal")
    parser.add_argument("--qwen-gpus", type=gpu_pair, default=("0", "1"))
    parser.add_argument("--llama-gpus", type=gpu_pair, default=("2", "3"))
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--dataset", default="dolly")
    parser.add_argument("--data-root", default="/nas/Users/wyx/Baseline/data")
    parser.add_argument("--output-root", default="/nas/Users/wyx/Baseline")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching jobs or changing artifacts")
    parser.add_argument("--eval-during-training", action="store_true", default=None)
    parser.add_argument("--eval-rouge-tokenizer", choices=("english", "unicode"))
    for name in ("max-steps", "save-steps", "eval-steps", "max-prompt-tokens", "max-new-tokens", "gradient-accumulation-steps", "eval-max-new-tokens", "eval-max-examples"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Methods and seeds must not contain duplicates")
    if args.group == "all" and set(args.qwen_gpus) & set(args.llama_gpus):
        parser.error("The two GPU groups must not overlap")
    groups, state = build_groups(args)
    gpus = {group: getattr(args, f"{group}_gpus") for group in groups}
    for group, stages in groups.items():
        print(f"{group}: student GPU {gpus[group][0]}, teacher GPU {gpus[group][1]}", flush=True)
        for stage in stages:
            print(f"  {stage.kind}: {stage.method}, seed {stage.config['seed']} -> {stage.output_path}", flush=True)
    print(f"Logs and progress: {state}", flush=True)
    if args.dry_run:
        return 0
    return Runner(groups, gpus, state).run()


if __name__ == "__main__":
    raise SystemExit(main())
