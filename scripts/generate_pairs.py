#!/usr/bin/env python3
"""Generate immutable teacher/student pairs with exact completion IDs locally."""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version as package_version
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_common.config import load_config, validate_config
from baseline_common.data import (assert_disjoint, file_sha256, join_pair_records,
                                  load_records)
from baseline_common.models import (generate_response, load_model, model_fingerprint,
                                    render_prompt, verify_tokenizers, bind_vocabulary)
from baseline_common.pair_progress import (append_jsonl, atomic_json, atomic_jsonl,
                                          generation_identity, pair_lock, pair_paths,
                                          read_progress_rows, require_identity,
                                          validate_role_records)


def generate_role(model, tokenizer, records, cfg, generation, *, on_record=None):
    """Keep IDs returned by generate: decoded text is for inspection only."""
    import torch
    generated = []
    started = time.monotonic()
    for index, row in enumerate(records):
        # Per-record seeds make continuation order irrelevant and make tiny
        # subsets reproducible without sharing teacher/student RNG state.
        seed = (cfg["seed"] + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16)) % (2**32)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        prompt_ids = render_prompt(tokenizer, row["prompt"], cfg["max_prompt_tokens"], cfg["enable_thinking"])
        response_ids = generate_response(model, tokenizer, prompt_ids, cfg["max_new_tokens"], generation)
        if not response_ids:
            raise ValueError(f"Generation produced no tokens for {row['id']}")
        result = {"id": row["id"], "prompt": row["prompt"], "prompt_ids": prompt_ids,
                  "response_ids": response_ids,
                  "response": tokenizer.decode(response_ids, skip_special_tokens=True),
                  **({"source_group": row["source_group"]} if "source_group" in row else {})}
        if on_record is not None:
            on_record(result)
        generated.append(result)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"Generated {index + 1}/{len(records)} in {time.monotonic() - started:.1f}s", flush=True)
    return generated


def _validate_completed(paths, identity, records, tokenizer, cfg, vocabulary_sizes):
    manifest = json.loads(paths["manifest"].read_text())
    expected = identity
    if "generation_runtime" not in manifest["provenance"]:
        # No further rows will be generated, so complete legacy datasets can
        # retain their historical unknown runtime without mixing producers.
        expected = {**identity, "generation_runtime": None}
        print("Legacy complete pairs: generation runtime is historically unknown; validating immutable artifacts", flush=True)
    require_identity(generation_identity(manifest["provenance"], manifest["effective_config"]), expected)
    if manifest["records"] != len(records):
        raise ValueError("Completed pair manifest has the wrong record count")
    for name in ("output", "teacher", "student"):
        path = paths[name]
        if not path.is_file() or manifest.get("files", {}).get(path.name) != file_sha256(path):
            raise ValueError(f"Completed pair artifact missing or checksum changed: {path}")
    generated = {}
    for role in ("teacher", "student"):
        generated[role] = read_progress_rows(paths[role])
        validate_role_records(generated[role], records, tokenizer, cfg,
                              vocabulary_size=vocabulary_sizes[role], complete=True)
    expected = join_pair_records(generated["teacher"], generated["student"], provenance=manifest["provenance"])
    if load_records(paths["output"], paired=True) != expected:
        raise ValueError("Completed pairs do not match teacher/student sidecars and manifest provenance")


def _require_unchanged_inputs(cfg, input_path, identity):
    if (model_fingerprint(cfg["teacher_model"]) != identity["teacher"]
            or model_fingerprint(cfg["student_model"]) != identity["student"]
            or file_sha256(input_path) != identity["source_sha256"]
            or file_sha256(cfg["validation_file"]) != identity["validation_sha256"]):
        raise RuntimeError("Checkpoint or source identity changed during generation; use a new output path")


def run_generation(cfg, *, config_path, input_path, output, resume=False, max_examples=None,
                   device_override=None):
    """Run or continue both producer roles while holding one output-wide lock."""
    paths = pair_paths(Path(output).expanduser())
    with pair_lock(paths["lock"]):
        existing = [path for key, path in paths.items() if key != "lock" and path.exists()]
        if existing and not resume:
            raise FileExistsError(f"Generation artifact exists: {existing[0]}; pass --resume or use a new output path")
        source_sha256 = file_sha256(input_path)
        validation_sha256 = file_sha256(cfg["validation_file"])
        records = load_records(input_path)
        validation = load_records(cfg["validation_file"])
        assert_disjoint(records, validation)
        if max_examples is not None:
            records = records[:max_examples]
        teacher_identity = model_fingerprint(cfg["teacher_model"])
        student_identity = model_fingerprint(cfg["student_model"])
        tokenizer = verify_tokenizers(cfg["teacher_model"], cfg["student_model"], policy=cfg["vocabulary_policy"])
        vocabulary_sizes = {role: json.loads((Path(cfg[f"{role}_model"]) / "config.json").read_text())["vocab_size"]
                            for role in ("teacher", "student")}
        provenance = {
            "vocabulary_policy": cfg["vocabulary_policy"],
            "vocabulary_alignment": getattr(tokenizer, "_baseline_vocab_alignment", None),
            "teacher": teacher_identity, "student": student_identity,
            "tokenizer": {"path": student_identity["path"],
                          "metadata_sha256": {key: value for key, value in student_identity["metadata_sha256"].items()
                                              if "token" in key or key.endswith(".jinja")}},
            "input_config_sha256": file_sha256(config_path),
            "effective_config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "source_sha256": source_sha256,
            "validation_sha256": validation_sha256,
            "seed": cfg["seed"], "enable_thinking": cfg["enable_thinking"],
            "max_prompt_tokens": cfg["max_prompt_tokens"], "max_new_tokens": cfg["max_new_tokens"],
            "generation": cfg["generation"], "teacher_generation": cfg["teacher_generation"],
            "generation_dtype": cfg["dtype"], "max_examples": max_examples,
            "generation_runtime": {package: package_version(package)
                                   for package in ("torch", "transformers", "tokenizers")},
        }
        identity = generation_identity(provenance, cfg)
        if paths["manifest"].exists():
            _validate_completed(paths, identity, records, tokenizer, cfg, vocabulary_sizes)
            _require_unchanged_inputs(cfg, input_path, identity)
            print(f"Validated {len(records)} existing pairs in {paths['output']}; generation already complete", flush=True)
            return paths["output"]
        if paths["progress"].exists():
            progress = json.loads(paths["progress"].read_text())
            require_identity(generation_identity(progress["provenance"], progress["effective_config"]), identity)
            if progress["records"] != len(records):
                raise ValueError("Pair progress has the wrong record count")
            # Keep the original producer provenance even when the caller has
            # changed unrelated training settings or its output directory.
            provenance = progress["provenance"]
        else:
            if existing:
                raise ValueError("Incomplete pair artifacts lack a progress manifest; use a new output path")
            progress = {"schema_version": 1, "records": len(records), "provenance": provenance,
                        "effective_config": cfg, "generation_costs": {}, "elapsed_seconds": 0.0}
            atomic_json(paths["progress"], progress)
        generated = {}
        # Validate both roles before loading a model or appending any new row.
        for role in ("teacher", "student"):
            generated[role] = read_progress_rows(paths[f"{role}_partial"], recover_tail=True)
            validate_role_records(generated[role], records, tokenizer, cfg, vocabulary_size=vocabulary_sizes[role])
        generation_costs = progress["generation_costs"]
        started = time.monotonic()
        import torch
        for role, path, generation in [("teacher", cfg["teacher_model"], cfg["teacher_generation"]),
                                       ("student", cfg["student_model"], cfg["generation"])]:
            have = {row["id"] for row in generated[role]}
            missing = [row for row in records if row["id"] not in have]
            device = device_override or (cfg["teacher_device"] if role == "teacher" else None) or cfg["device"]
            load_seconds = generation_seconds = 0.0
            if missing:
                print(f"Loading {role}: {path} on {device}; {len(have)}/{len(records)} records already durable", flush=True)
                load_started = time.monotonic()
                model = load_model(path, cfg["dtype"], device)
                try:
                    bind_vocabulary(model, tokenizer)
                    model.eval()
                    model.requires_grad_(False)
                    generation_started = time.monotonic()
                    load_seconds = generation_started - load_started
                    def persist(row):
                        append_jsonl(paths[f"{role}_partial"], row)
                    generated[role].extend(generate_role(model, tokenizer, missing, cfg, generation, on_record=persist))
                    generation_seconds = time.monotonic() - generation_started
                finally:
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                print(f"{role}: all {len(records)} records already durable", flush=True)
            validate_role_records(generated[role], records, tokenizer, cfg,
                                  vocabulary_size=vocabulary_sizes[role], complete=True)
            if role not in generation_costs:
                generation_costs[role] = {
                    "load_seconds": load_seconds, "generation_seconds": generation_seconds,
                    "records": len(generated[role]),
                    "prompt_tokens": sum(len(row["prompt_ids"]) for row in generated[role]),
                    "generated_tokens": sum(len(row["response_ids"]) for row in generated[role]),
                    "device": str(device), "resumed_records": len(have),
                    "timing_incomplete": bool(have),
                }
            progress["generation_costs"] = generation_costs
            atomic_json(paths["progress"], progress)
        provenance = {**provenance, "offline_generation": generation_costs}
        pairs = join_pair_records(generated["teacher"], generated["student"], provenance=provenance)
        _require_unchanged_inputs(cfg, input_path, identity)
        # Every destination is replaced atomically. The manifest is published
        # last and is the only completion marker. A kill between these writes
        # is harmless: the original append logs remain available on restart.
        atomic_jsonl(paths["teacher"], generated["teacher"])
        atomic_jsonl(paths["student"], generated["student"])
        atomic_jsonl(paths["output"], pairs)
        manifest = {"schema_version": 1, "records": len(pairs), "provenance": provenance,
                    "effective_config": progress["effective_config"], "generation_costs": generation_costs,
                    "elapsed_seconds": progress.get("elapsed_seconds", 0.0) + time.monotonic() - started,
                    "files": {paths[name].name: file_sha256(paths[name]) for name in ("output", "teacher", "student")}}
        atomic_json(paths["manifest"], manifest)
        print(f"Wrote {len(pairs)} pairs to {paths['output']}", flush=True)
        return paths["output"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Canonical training JSONL produced by prepare_data.py")
    parser.add_argument("--output", required=True, help="Destination paired JSONL")
    parser.add_argument("--resume", action="store_true", help="Continue matching durable progress or validate completed pairs")
    parser.add_argument("--device", help="Override generation device, e.g. cuda:0")
    parser.add_argument("--validation-file", help="Override the held-out canonical validation JSONL")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-examples", type=int, help="First N canonical records for an explicitly limited pilot")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ("validation_file", "seed", "max_prompt_tokens", "max_new_tokens"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.device:
        cfg["device"] = cfg["teacher_device"] = args.device
    cfg = validate_config(cfg)
    if cfg["method"] != "distillm2":
        parser.error("Pair generation requires a distillm2 config")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    run_generation(cfg, config_path=args.config, input_path=args.input, output=args.output,
                   resume=args.resume, max_examples=args.max_examples, device_override=args.device)


if __name__ == "__main__":
    main()
