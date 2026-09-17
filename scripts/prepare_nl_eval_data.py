#!/usr/bin/env python3
"""Prepare local ROUGE-L benchmarks without starting model evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_common.nl_data import BENCHMARKS, prepare_nl_pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="/nas/Datasets")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts/nl_eval_data"))
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--exclude-file", action="append", default=None,
                        help="Canonical training/pair JSONL to exclude by prompt; repeat for the union of training files")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Validate and reuse an identical complete pool (default: enabled)")
    args = parser.parse_args()
    exclusions = args.exclude_file if args.exclude_file is not None else ["/nas/Users/wyx/Baseline/data/dolly/train.jsonl"]
    manifest = prepare_nl_pool(args.source_root, args.output_dir, args.benchmarks, exclusions, args.resume)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "benchmarks": {
        name: {key: entry[key] for key in ("source_records", "exact_duplicates_merged", "training_overlaps_removed", "records")}
        for name, entry in manifest["benchmarks"].items()}}, indent=2))


if __name__ == "__main__":
    main()
