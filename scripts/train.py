#!/usr/bin/env python3
"""Launch one of the eight experiments; all data and checkpoints remain local."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_common.config import load_config, validate_config, run_directory

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Completed checkpoints/step_XXXXXX directory")
    parser.add_argument("--dry-run", action="store_true", help="Resolve settings only; does not claim runtime readiness")
    parser.add_argument("--eval-during-training", action="store_true", default=None,
                        help="Optionally select checkpoints using generated ROUGE-L; default trains only")
    parser.add_argument("--eval-rouge-tokenizer", choices=("english", "unicode"))
    for flag in ("train-file", "validation-file", "output-root", "dataset", "device", "teacher-device", "dtype"):
        parser.add_argument("--" + flag)
    for flag in ("seed", "max-steps", "max-prompt-tokens", "max-new-tokens", "gradient-accumulation-steps", "save-steps", "eval-steps", "eval-max-new-tokens", "eval-max-examples"):
        parser.add_argument("--" + flag, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in vars(args).items():
        if key not in {"config", "resume", "dry_run"} and value is not None:
            cfg[key] = value
    cfg = validate_config(cfg)
    if args.dry_run:
        print(json.dumps({"config": cfg, "run_directory": str(run_directory(cfg))}, indent=2))
        return
    from baseline_common.train import train
    output = train(cfg, resume=args.resume)
    result = json.loads((output / "result.json").read_text())
    if result.get("interrupted") and not result["complete"]:
        print(f"Training interrupted safely at step {result['step']}; rerun the suite to resume.", file=sys.stderr)
        return 75
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
