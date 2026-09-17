#!/usr/bin/env python3
"""One-update real-Qwen helper-path smoke; not a complete experiment/trainer run.

Uses the production sample/loss/CPU-optimizer helpers on fresh pretrained
students. Exports and reloads HF checkpoints, but does not test full-loop
checkpoint scheduling, optimizer resume, dataset evaluation, or convergence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers

from baseline_common.config import validate_config
from baseline_common.models import (generate_response, load_model, model_fingerprint,
                                    render_prompt, verify_tokenizers)
from baseline_common.optim import CPUAdamW
from baseline_common.train import check_data, response_loss, sample_record, seed_everything


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-device", default="cuda:2")
    parser.add_argument("--teacher-device", default="cuda:3")
    parser.add_argument("--methods", nargs="+", choices=["abkd", "skd", "distillm2"],
                        default=["abkd", "skd", "distillm2"])
    parser.add_argument("--output-root", type=Path,
                        default=Path("/nas/Users/wyx/Baseline/smoke/qwen3_8b_1p7b"))
    parser.add_argument("--report", type=Path, default=Path("docs/evidence/qwen-method-smoke.json"))
    args = parser.parse_args()
    for method in args.methods:
        destination = args.output_root / method / "student"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite smoke export {destination}")
    torch.set_num_threads(8)
    torch.cuda.set_device(args.student_device)
    cfg = validate_config(dict(
        name="qwen_real_helper_smoke", pair="qwen3_8b_1p7b", method="abkd", dataset="synthetic_smoke",
        teacher_model="/nas/Models/Qwen3-8B-Instruct", student_model="/nas/Models/Qwen3-1.7B-Instruct",
        train_file=str(args.output_root / "synthetic.train.jsonl"),
        validation_file=str(args.output_root / "synthetic.validation.jsonl"),
        output_root=str(args.output_root), device=args.student_device, teacher_device=args.teacher_device,
        dtype="bfloat16", optimizer_offload=True, gradient_checkpointing=True,
        max_prompt_tokens=64, max_new_tokens=8, gradient_accumulation_steps=1,
        max_steps=1, loss_chunk_size=4,
    ))
    row = {"id": "smoke-arithmetic", "prompt": "What is 2 + 2? Answer briefly.", "response": "4."}
    validation = [{"id": "smoke-validation", "prompt": "What is 3 + 4?", "response": "7."}]
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one update per method through production sample_record/response_loss/CPUAdamW helpers",
        "limitations": ["Synthetic prompt, not prepared training datasets or scientific results",
                        "Does not exercise full train loop, optimizer resume, DDP, or validation",
                        "64 prompt / 8 response token budget, not full workload memory certification"],
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda_build": torch.version.cuda, "config": cfg.copy(), "input": row,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "methods": {}, "complete": False,
    }
    tokenizer = verify_tokenizers(cfg["teacher_model"], cfg["student_model"])
    identities = {"teacher": model_fingerprint(cfg["teacher_model"]),
                  "student": model_fingerprint(cfg["student_model"])}
    report["checkpoints"] = identities
    save_report(args.report, report)
    print("Loading frozen teacher on " + args.teacher_device, flush=True)
    teacher = load_model(cfg["teacher_model"], "bfloat16", args.teacher_device)
    teacher.eval().requires_grad_(False)
    teacher_probe_name, teacher_probe = next((name, p) for name, p in teacher.named_parameters() if "q_proj.weight" in name)
    teacher_before = teacher_probe.detach().cpu().clone()
    teacher_forward_calls = [0]

    def frozen_teacher_hook(module, _inputs):
        if module.training or torch.is_grad_enabled():
            raise AssertionError("Teacher forward must be eval/no-grad")
        teacher_forward_calls[0] += 1

    teacher.register_forward_pre_hook(frozen_teacher_hook)
    for method in args.methods:
        cfg["method"] = method
        seed_everything(cfg["seed"])
        started = time.monotonic()
        result = {"status": "running", "fresh_student_from_original": True}
        report["methods"][method] = result
        save_report(args.report, report)
        print(f"[{method}] Loading fresh student on {args.student_device}", flush=True)
        torch.cuda.reset_peak_memory_stats(args.student_device)
        torch.cuda.reset_peak_memory_stats(args.teacher_device)
        teacher_calls_before = teacher_forward_calls[0]
        student = load_model(cfg["student_model"], "bfloat16", args.student_device, True)
        student.train()
        student_probe_name, student_probe = next((name, p) for name, p in student.named_parameters() if "q_proj.weight" in name)
        student_before = student_probe.detach().cpu().clone()
        counters = {"teacher_training_tokens": 0, "response_tokens": 0}
        current_row = row.copy()
        if method == "distillm2":
            print(f"[{method}] Generating teacher and initial-student pair IDs", flush=True)
            prefix = render_prompt(tokenizer, row["prompt"], cfg["max_prompt_tokens"], False)
            current_row.update(
                prompt_ids=prefix,
                chosen_ids=generate_response(teacher, tokenizer, prefix, 8, cfg["teacher_generation"]),
                rejected_ids=generate_response(student, tokenizer, prefix, 8, cfg["generation"]),
                provenance={**identities, "tokenizer": identities["student"]["metadata_sha256"],
                            **{key: cfg[key] for key in ("enable_thinking", "max_prompt_tokens", "max_new_tokens", "generation", "teacher_generation")}},
            )
            check_data(cfg, [current_row], validation)
            result["paired_record"] = current_row
        print(f"[{method}] Producing samples and allocating CPU FP32 Adam state", flush=True)
        records = sample_record(current_row, cfg, student, teacher, tokenizer, counters)
        optimizer = CPUAdamW(student.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.999),
                             eps=1e-8, weight_decay=cfg["weight_decay"])
        optimizer.zero_grad()
        loss_value = 0.0
        denominator = 1 if method == "distillm2" else sum(t != -100 for t in records[0]["labels"][1:])
        for branch, record in enumerate(records):
            objective = ("skew_forward" if branch == 0 else "skew_reverse") if method == "distillm2" else method
            reduction = "sequence_mean" if method == "distillm2" else "sum"
            print(f"[{method}] Forward/backward branch {branch}", flush=True)
            loss = response_loss(student, teacher, record, cfg, objective, reduction, counters) / denominator
            loss.backward()
            loss_value += loss.detach().float().item()
            del loss
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
        if not torch.isfinite(norm) or norm <= 0:
            raise AssertionError("Expected finite nonzero student gradient norm")
        print(f"[{method}] Applying CPU optimizer update", flush=True)
        optimizer.step()
        changed = int(student_probe.detach().cpu().ne(student_before).sum())
        if changed <= 0:
            raise AssertionError("Student probe did not change after the optimizer step")
        teacher_frozen = all(not p.requires_grad and p.grad is None for p in teacher.parameters())
        if not teacher_frozen or not torch.equal(teacher_probe.detach().cpu(), teacher_before):
            raise AssertionError("Teacher changed during distillation")
        expected_probe = student_probe.detach().cpu().clone()
        destination = args.output_root / method / "student"
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{method}] Saving HF export to {destination}", flush=True)
        student.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        result.update(loss=loss_value, gradient_norm_before_clip=float(norm),
                      changed_probe_parameter=student_probe_name, changed_probe_elements=changed,
                      teacher_probe_parameter=teacher_probe_name, teacher_probe_unchanged=True,
                      teacher_frozen=teacher_frozen, teacher_forward_calls=teacher_forward_calls[0] - teacher_calls_before,
                      counters=counters, response_lengths=[sum(t != -100 for t in r["labels"]) for r in records],
                      peak_student_gpu_bytes=torch.cuda.max_memory_allocated(args.student_device),
                      peak_teacher_gpu_bytes=torch.cuda.max_memory_allocated(args.teacher_device),
                      export_path=str(destination))
        del student_probe, student_before, student, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{method}] Reloading exported weights", flush=True)
        reloaded = load_model(str(destination), "bfloat16", args.student_device)
        reloaded_probe = dict(reloaded.named_parameters())[student_probe_name]
        if not torch.equal(reloaded_probe.detach().cpu(), expected_probe):
            raise AssertionError("Saved/reloaded student probe differs")
        reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(destination, local_files_only=True)
        if reloaded_tokenizer.get_vocab() != tokenizer.get_vocab() or reloaded_tokenizer.chat_template != tokenizer.chat_template:
            raise AssertionError("Saved tokenizer metadata changed")
        if not reloaded.config.tie_word_embeddings or reloaded.get_input_embeddings().weight.data_ptr() != reloaded.get_output_embeddings().weight.data_ptr():
            raise AssertionError("Qwen student lost tied embeddings on save/reload")
        result.update(status="passed", export_reload_probe_equal=True, export_tokenizer_equal=True,
                      tied_embeddings_preserved=True, elapsed_seconds=time.monotonic() - started)
        save_report(args.report, report)
        print(f"[{method}] Passed: loss={loss_value:.7f}, grad_norm={float(norm):.7f}, changed={changed}", flush=True)
        del reloaded_probe, reloaded, reloaded_tokenizer, expected_probe, norm, records
        gc.collect()
        torch.cuda.empty_cache()
    report["complete"] = True
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.report, report)


if __name__ == "__main__":
    main()
