#!/usr/bin/env python3
"""Run selected experiment configurations sequentially; preview by default."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_common.config import load_config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=["all", "qwen3_8b_1p7b", "llama3_8b_llama32_1b"], default="all")
    parser.add_argument("--methods", nargs="+", choices=["kd", "abkd", "skd", "distillm2"], default=["kd", "abkd", "skd", "distillm2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--dataset", default="dolly")
    parser.add_argument("--data-root", default="/nas/Users/wyx/Baseline/data")
    parser.add_argument("--output-root", default="/nas/Users/wyx/Baseline")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    for path in sorted((ROOT / "configs/experiments").glob("*.json")):
        cfg = load_config(path)
        if cfg["method"] not in args.methods or args.pair not in {"all", cfg["pair"]}:
            continue
        data = Path(args.data_root) / args.dataset
        train = data / cfg["pair"] / "pairs.train.jsonl" if cfg["method"] == "distillm2" else data / "train.jsonl"
        for seed in args.seeds:
            command = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(path),
                       "--dataset", args.dataset, "--train-file", str(train), "--validation-file", str(data / "validation.jsonl"),
                       "--output-root", args.output_root, "--seed", str(seed)]
            if args.max_steps:
                command += ["--max-steps", str(args.max_steps)]
            print(shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, check=True, cwd=ROOT)

if __name__ == "__main__":
    main()
