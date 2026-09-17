#!/usr/bin/env python3
"""Read-only runtime/model audit; no downloads, installations, or model loading.

The optional --output report is the only filesystem write. Safetensors headers
are inspected without loading multi-gigabyte tensors. Exit 1 means at least one
required preflight failed, not necessarily that the physical host lacks GPUs:
sandboxes/containers may hide otherwise working devices.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


PINS = {
    "torch": "2.6.0", "transformers": "4.51.3", "accelerate": "1.6.0",
    "datasets": "3.5.0", "tokenizers": "0.21.1", "numpy": "1.26.4",
    "huggingface-hub": "0.30.2", "safetensors": "0.5.3",
    "rouge-score": "0.1.2",
}
PAIRS = {
    "qwen3": ("Qwen3-8B-Instruct", "Qwen3-1.7B-Instruct"),
    "llama3": ("Meta-Llama-3-8B-Instruct", "Meta-Llama-3.2-1B-Instruct"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def command(argv):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def token_map(tokenizer):
    vocab = tokenizer["model"]["vocab"]
    if not isinstance(vocab, dict):
        raise ValueError("Expected a BPE token-to-ID vocabulary")
    result = dict(vocab)
    for token in tokenizer.get("added_tokens", []):
        name, token_id = token["content"], token["id"]
        if name in result and result[name] != token_id:
            raise ValueError(f"Conflicting token IDs for {name!r}")
        result[name] = token_id
    if any(not isinstance(i, int) or i < 0 for i in result.values()):
        raise ValueError("Non-integer or negative token ID")
    if len(set(result.values())) != len(result):
        raise ValueError("Multiple token strings share the same ID")
    return result


def inspect_weights(path, config):
    index_path = path / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.is_file() else None
    files = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
    errors, tensors, total_bytes = [], {}, 0
    for filename in files:
        file = path / filename
        if not file.is_file():
            errors.append(f"Missing weight shard: {filename}")
            continue
        try:
            total_bytes += file.stat().st_size
            with file.open("rb") as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError("Truncated safetensors length header")
                length = struct.unpack("<Q", prefix)[0]
                if length > 100_000_000 or length + 8 > file.stat().st_size:
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(stream.read(length))
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                start, end = tensor["data_offsets"]
                if start < 0 or end < start or end + 8 + length > file.stat().st_size:
                    errors.append(f"Truncated/invalid tensor {name} in {filename}")
                if name in tensors:
                    errors.append(f"Duplicate weight tensor {name}")
                tensors[name] = {"shape": tensor["shape"], "dtype": tensor["dtype"],
                                 "file": filename}
            if index:
                for name, mapped_file in index["weight_map"].items():
                    if mapped_file == filename and name not in header:
                        errors.append(f"Index tensor missing from {filename}: {name}")
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
            errors.append(f"Invalid weight file {filename}: {exc}")
    embedding = tensors.get("model.embed_tokens.weight")
    head = tensors.get("lm_head.weight")
    expected = [config.get("vocab_size"), config.get("hidden_size")]
    if embedding is None or embedding["shape"] != expected:
        errors.append(f"Embedding shape is not configured {expected}")
    if head is None:
        if config.get("tie_word_embeddings") and embedding:
            head = {**embedding, "tied_to": "model.embed_tokens.weight"}
        else:
            errors.append("Missing untied lm_head.weight")
    if head and head["shape"] != expected:
        errors.append(f"Output head shape is not configured {expected}")
    return {"files": files, "total_file_bytes": total_bytes,
            "tensor_count": len(tensors),
            "stored_parameter_count": sum(math.prod(t["shape"]) for t in tensors.values()),
            "embedding": embedding, "output_head": head,
            "verification": "Headers, file lengths, index membership; tensor values not read or hashed",
            "errors": errors}


def inspect_model(path):
    result = {"path": str(path), "exists": path.is_dir(), "errors": []}
    required = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        result["errors"].append("Missing model metadata: " + ", ".join(missing))
        return result, None
    try:
        config = read_json(path / "config.json")
        tokenizer = read_json(path / "tokenizer.json")
        tokenizer_config = read_json(path / "tokenizer_config.json")
        generation = read_json(path / "generation_config.json")
        vocab = token_map(tokenizer)
        result["config"] = {k: config.get(k) for k in (
            "architectures", "model_type", "vocab_size", "hidden_size", "num_hidden_layers",
            "torch_dtype", "tie_word_embeddings", "bos_token_id", "eos_token_id")}
        result["generation"] = generation
        result["metadata_sha256"] = {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in required
        }
        result["tokenizer"] = {
            "tokens": len(vocab), "max_id": max(vocab.values()), "token_map_sha256": digest(vocab),
            "bos_token": tokenizer_config.get("bos_token"),
            "eos_token": tokenizer_config.get("eos_token"),
            "pad_token": tokenizer_config.get("pad_token"),
            "chat_template_sha256": digest(tokenizer_config.get("chat_template")),
            "has_chat_template": bool(tokenizer_config.get("chat_template")),
        }
        eos = generation.get("eos_token_id", config.get("eos_token_id"))
        eos = eos if isinstance(eos, list) else [eos]
        inverse = {value: key for key, value in vocab.items()}
        result["eos_tokens"] = {str(i): inverse.get(i) for i in eos}
        if not eos or any(i not in inverse for i in eos):
            result["errors"].append("EOS IDs are missing from tokenizer")
        if max(vocab.values()) >= config["vocab_size"]:
            result["errors"].append("Tokenizer ID exceeds configured output vocabulary")
        if not tokenizer_config.get("chat_template"):
            result["errors"].append("Instruct checkpoint has no chat template")
        result["weights"] = inspect_weights(path, config)
        result["errors"].extend(result["weights"]["errors"])
        return result, {"vocab": vocab, "tokenizer": tokenizer, "tokenizer_config": tokenizer_config,
                        "config": config, "eos": eos}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append(f"Model metadata error: {exc}")
        return result, None


def inspect_pair(root, family):
    names = PAIRS[family]
    teacher, teacher_raw = inspect_model(root / names[0])
    student, student_raw = inspect_model(root / names[1])
    result = {"teacher": teacher, "student": student, "compatible": False}
    if teacher_raw is None or student_raw is None:
        result["reason"] = "Complete teacher and student artifacts are required to prove compatibility"
        return result
    checks = {
        "full_token_to_id_map_equal": teacher_raw["vocab"] == student_raw["vocab"],
        "tokenizer_algorithm_equal": teacher_raw["tokenizer"] == student_raw["tokenizer"],
        "head_vocab_equal": teacher_raw["config"]["vocab_size"] == student_raw["config"]["vocab_size"],
        "generation_eos_equal": set(teacher_raw["eos"]) == set(student_raw["eos"]),
        "chat_template_equal": teacher_raw["tokenizer_config"].get("chat_template") ==
                               student_raw["tokenizer_config"].get("chat_template"),
    }
    result["checks"] = checks
    result["template_policy"] = "Both roles use the student's chat template; template equality is informational"
    required_checks = {key: value for key, value in checks.items() if key != "chat_template_equal"}
    result["vocabulary_policy"] = "llama3_shared" if family == "llama3" else "full"
    if result["vocabulary_policy"] == "llama3_shared":
        try:
            from baseline_common.models import verify_tokenizers
            tokenizer = verify_tokenizers(root / names[0], root / names[1], policy="llama3_shared")
            alignment = tokenizer._baseline_vocab_alignment
            result["vocabulary_alignment"] = alignment
            excluded = set(alignment["excluded_ids"])
            same_stops = (set(teacher_raw["eos"]) - excluded) == (set(student_raw["eos"]) - excluded)
            checks["shared_vocabulary_validated"] = True
            checks["shared_generation_eos_equal"] = same_stops
            required_checks = {"head_vocab_equal": checks["head_vocab_equal"],
                               "shared_vocabulary_validated": True, "shared_generation_eos_equal": same_stops}
        except Exception as exc:
            checks["shared_vocabulary_validated"] = False
            result["alignment_error"] = f"{type(exc).__name__}: {exc}"
            required_checks["shared_vocabulary_validated"] = False
    result["compatible"] = all(required_checks.values()) and not teacher["errors"] and not student["errors"]
    if not checks["full_token_to_id_map_equal"]:
        tokens = sorted(set(teacher_raw["vocab"]) | set(student_raw["vocab"]))
        result["token_mismatch_examples"] = [
            {"token": token, "teacher_id": teacher_raw["vocab"].get(token),
             "student_id": student_raw["vocab"].get(token)} for token in tokens
            if teacher_raw["vocab"].get(token) != student_raw["vocab"].get(token)
        ][:20]
    return result


def inspect_runtime(skip_cuda):
    errors, packages = [], {}
    for name, target in PINS.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
            errors.append(f"Missing package: {name}")
        packages[name] = {"installed": version, "target": target,
                          "matches_target": bool(version and version.split("+")[0] == target)}
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "packages": packages, "errors": errors}
    if sys.version_info < (3, 11):
        errors.append("Use Python 3.11 or later; the pinned reference environment uses 3.11")
    for name in ("numpy", "tokenizers", "safetensors", "accelerate", "datasets"):
        if packages[name]["installed"] is not None:
            try:
                importlib.import_module(name)
            except Exception as exc:
                errors.append(f"Import {name} failed: {type(exc).__name__}: {exc}")
    if packages["rouge-score"]["installed"] is not None:
        try:
            from rouge_score import rouge_scorer
            score = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score("same answer", "same answer")
            result["rouge_l_import_and_score"] = score["rougeL"].fmeasure == 1.0
            if not result["rouge_l_import_and_score"]:
                errors.append("ROUGE-L scorer self-check failed")
        except Exception as exc:
            errors.append(f"ROUGE-L import/scoring failed: {type(exc).__name__}: {exc}")
    if packages["transformers"]["installed"] is not None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, LlamaConfig
            result["transformers_models_import"] = all((AutoModelForCausalLM, AutoTokenizer,
                                                        Qwen3Config, LlamaConfig))
        except Exception as exc:
            errors.append(f"Required Transformers model import failed: {type(exc).__name__}: {exc}")
    try:
        import torch
        result["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available(), "devices": []}
        if skip_cuda:
            result["torch"]["cuda_compute_check"] = "skipped"
        elif not torch.cuda.is_available():
            errors.append("CUDA unavailable to this process; check GPU/container/sandbox access")
        else:
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    properties = torch.cuda.get_device_properties(i)
                    item = {"index": i, "name": properties.name, "total_memory_bytes": properties.total_memory,
                            "bf16_supported": torch.cuda.is_bf16_supported()}
                    try:
                        x = torch.ones((128, 128), dtype=torch.bfloat16, device=f"cuda:{i}")
                        value = (x @ x).float().mean().item()
                        torch.cuda.synchronize(i)
                        item["bf16_matmul_pass"] = value == 128.0
                        if not item["bf16_matmul_pass"] or not item["bf16_supported"]:
                            errors.append(f"GPU {i}: BF16 compute check failed")
                        del x
                    except Exception as exc:
                        item["error"] = f"{type(exc).__name__}: {exc}"
                        errors.append(f"GPU {i}: {item['error']}")
                    result["torch"]["devices"].append(item)
    except Exception as exc:
        errors.append(f"PyTorch runtime error: {type(exc).__name__}: {exc}")
    return result


def storage_report(path):
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    result = {"requested_output_root": str(path), "exists": path.is_dir(),
              "checked_ancestor": str(existing),
              "permission_check_writable": os.access(existing, os.W_OK),
              "verification": "Read-only permission/free-space check; no write attempted"}
    try:
        usage = shutil.disk_usage(existing)
        result.update(total_bytes=usage.total, free_bytes=usage.free)
    except OSError as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("/nas/Models"))
    parser.add_argument("--pair", choices=["all", *PAIRS], default="all")
    parser.add_argument("--output-root", type=Path, default=Path("/nas/Users/wyx/Baseline"))
    parser.add_argument("--output", type=Path, help="Write the JSON audit (parent must already exist)")
    parser.add_argument("--skip-cuda", action="store_true", help="Metadata/import audit only; does not certify GPU readiness")
    parser.add_argument("--strict-pins", action="store_true", help="Treat deviations from requirements-training.txt as failures")
    args = parser.parse_args()
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": inspect_runtime(args.skip_cuda),
              "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.free",
                                      "--format=csv,noheader,nounits"]),
              "storage": storage_report(args.output_root)}
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        report["host_memory_kib"] = {line.split(":")[0]: int(line.split()[1])
                                     for line in meminfo.read_text().splitlines()
                                     if line.split(":")[0] in ("MemTotal", "MemAvailable", "SwapTotal")}
    report["model_pairs"] = {family: inspect_pair(args.model_root, family)
                             for family in (PAIRS if args.pair == "all" else [args.pair])}
    pins_match = all(p["matches_target"] for p in report["runtime"]["packages"].values())
    if args.strict_pins and not pins_match:
        report["runtime"]["errors"].append("Installed versions differ from reference pins")
    report["reference_pins_match"] = pins_match
    report["preflight_pass"] = (not report["runtime"]["errors"]
                                 and all(pair["compatible"] for pair in report["model_pairs"].values())
                                 and report["storage"]["permission_check_writable"])
    report["gpu_preflight_pass"] = report["preflight_pass"] and not args.skip_cuda
    report["limitations"] = [
        "No training step, full model load, convergence, or distributed test is performed by this script.",
        "Output write access is not tested by creating a file; permissions may differ inside a sandbox.",
        "Weight headers and lengths are checked; checkpoint identity/corruption requires hashing tensor data.",
        "VRAM fit depends on sequence lengths, optimizer placement, activations, and generation caches.",
    ]
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["preflight_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
