#!/usr/bin/env python3
"""Prepare local instruction corpora without reading evaluation splits for training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_common.data import DATASET_SOURCES, DEFAULT_DATA_ROOT, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["dolly", "metamathqa", "gsm8k", "auto"], default="dolly")
    parser.add_argument("--source", help="Local JSON/JSONL/parquet file, split directory, or HF save_to_disk directory")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--output-dir")
    parser.add_argument("--validation-fraction", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, help="Deterministic subset for pilot runs; hash all source files")
    parser.add_argument("--exclude-source", action="append", default=[],
                        help="Explicit held-out JSON/parquet file; remove matching prompts before splitting (repeatable)")
    args = parser.parse_args()
    source = args.source or DATASET_SOURCES.get(args.dataset)
    if source is None:
        parser.error("--source is required with --dataset auto")
    manifest = prepare_dataset(source, args.output_dir or DEFAULT_DATA_ROOT / args.dataset,
                               dataset=args.dataset, source_split=args.source_split,
                               validation_fraction=args.validation_fraction, seed=args.seed,
                               max_records=args.max_records, exclude_sources=args.exclude_source)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
