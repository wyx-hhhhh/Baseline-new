#!/usr/bin/env python3
"""Freeze one MetaMathQA pool and complete GSM8K/MATH tests for all methods."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_common.math_data import (DEFAULT_GSM8K_SOURCE, DEFAULT_MATH_POOL,
                                       DEFAULT_MATH_SOURCE, DEFAULT_METAMATH_SOURCE,
                                       prepare_math_pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metamath-source", default=str(DEFAULT_METAMATH_SOURCE))
    parser.add_argument("--gsm8k-source", default=str(DEFAULT_GSM8K_SOURCE))
    parser.add_argument("--math-source", default=str(DEFAULT_MATH_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_MATH_POOL))
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Validate and reuse an identical complete pool")
    args = parser.parse_args()
    manifest = prepare_math_pool(args.metamath_source, args.gsm8k_source, args.math_source,
                                 args.output_dir, train_size=args.train_size,
                                 validation_size=args.validation_size, seed=args.seed, resume=args.resume)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()),
                      "source_records": manifest["source_records"],
                      "filter_counts": manifest["filter_counts"],
                      "selection_counts": manifest["selection_counts"],
                      "splits": {name: {key: value for key, value in split.items() if key != "selected_ids"}
                                 for name, split in manifest["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
