"""Training and exact-step resume for KD, ABKD, online SKD and paired DistiLLM-2.

Run through scripts/train.py. Microbatch is one prompt; accumulation is normalized
by actual response targets across the entire optimizer step and all DDP ranks.
"""
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import run_directory, validate_config
from .data import load_records, file_sha256, prompt_key
from .losses import causal_distillation_loss
from .models import (encode_reference, load_model, model_fingerprint, render_prompt,
                     stop_ids, verify_tokenizers, bind_vocabulary, check_supported)
from .optim import CPUAdamW
from .sampling import SamplingConfig, skd_generate


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, required_cuda_devices=()):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    for device in required_cuda_devices:
        if str(device).startswith("cuda"):
            index = torch.device(device).index
            index = torch.cuda.current_device() if index is None else index
            if index >= len(state["cuda"]):
                raise ValueError(f"Checkpoint has no CUDA RNG state for required device {device}")
    if state["cuda"]:
        # Legacy unmasked runs captured every GPU, including unused devices.
        # A pair-isolated restart keeps logical device 0/1 and their exact RNGs;
        # states for now-invisible, unused devices must not be restored.
        for index, rng in enumerate(state["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng, device=index)


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_artifacts(*paths):
    """Make this export's closed files durable before publishing its marker.

    Only explicitly supplied files/directories are traversed. In particular,
    checkpoint siblings and older exports are never recursively flushed.
    """
    files, directories = set(), set()
    for item in paths:
        path = Path(item)
        directories.add(path.parent)
        if path.is_dir():
            directories.add(path)
            for child in path.rglob("*"):
                (directories if child.is_dir() else files).add(child)
        else:
            files.add(path)
    for path in sorted(files):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    # Child directory entries become durable before their parent's entries.
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        _sync_directory(path)


def write_json(path, value):
    """Commit metadata atomically; a killed writer leaves the previous JSON intact."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_run_manifest(cfg, world=1, tokenizer=None):
    """Read-only identity shared by the trainer and resume coordinator."""
    cfg = validate_config(cfg)
    if tokenizer is None:
        tokenizer = verify_tokenizers(cfg["teacher_model"], cfg["student_model"], policy=cfg["vocabulary_policy"])
    if cfg["evaluation_metric"] == "math_accuracy":
        evaluation_protocol = {"name": "math_accuracy", "stage": "separate_after_training",
                               "benchmarks": ["gsm8k", "math"], "higher_is_better": True,
                               "grader": "baseline_common.math_metrics", "range": [0, 100]}
    else:
        from .evaluation import metric_definition
        evaluation_protocol = metric_definition(cfg["eval_rouge_tokenizer"])
    return {"config": cfg, "world_size": world, "teacher": model_fingerprint(cfg["teacher_model"]),
            "vocabulary_alignment": getattr(tokenizer, "_baseline_vocab_alignment", None),
            "student": model_fingerprint(cfg["student_model"]), "train_sha256": file_sha256(cfg["train_file"]),
            "validation_sha256": file_sha256(cfg["validation_file"]),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "tokenizers")},
            "evaluation_metric": cfg["evaluation_metric"],
            "evaluation_protocol": evaluation_protocol,
            "selection_metric": cfg["evaluation_metric"] if cfg["eval_during_training"] else None,
            "selection_direction": "maximize" if cfg["eval_during_training"] else None,
            "optimizer": "CPU FP32 master AdamW" if cfg["optimizer_offload"] else "torch AdamW (model dtype states)"}


def validate_run_manifest(previous, expected):
    """Reject changed schedules/data/models while accepting older full-vocab metadata."""
    previous = {**previous, "config": validate_config(previous["config"])}
    previous.setdefault("vocabulary_alignment", None)
    if previous != expected:
        raise ValueError("Resume manifest differs (config, data, model, runtime or world size changed)")


class _StopRequest:
    def __init__(self):
        self.signum = None

    def handler(self, signum, _frame):
        # Never raise from a signal handler: accumulation/checkpoint writes must
        # finish as a unit. The training loop synchronizes at safe boundaries.
        self.signum = signum

    def pending(self, device, world):
        if world > 1:
            requested = torch.tensor(self.signum or 0, device=device)
            dist.all_reduce(requested, op=dist.ReduceOp.MAX)
            self.signum = int(requested.item()) or None
        return self.signum is not None


@contextmanager
def _graceful_interrupt():
    request = _StopRequest()
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request.handler)
    try:
        yield request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def check_data(cfg, train, validation):
    if not train or not validation:
        raise ValueError("Train and held-out validation must both be nonempty")
    for key, values_a, values_b in (
        ("id", {r["id"] for r in train}, {r["id"] for r in validation}),
        ("prompt", {prompt_key(r["prompt"]) for r in train}, {prompt_key(r["prompt"]) for r in validation}),
        ("source_group", {r["source_group"] for r in train if "source_group" in r}, {r["source_group"] for r in validation if "source_group" in r}),
    ):
        if values_a & values_b:
            raise ValueError(f"Training/validation leakage by {key}")
    if cfg["method"] == "distillm2":
        expected = {"teacher": model_fingerprint(cfg["teacher_model"]),
                    "student": model_fingerprint(cfg["student_model"])}
        for record in train:
            provenance = record.get("provenance", {})
            if provenance.get("vocabulary_policy", "full") != cfg["vocabulary_policy"]:
                raise ValueError("Pair vocabulary policy differs; regenerate pairs")
            for key, fingerprint in expected.items():
                if provenance.get(key) != fingerprint:
                    raise ValueError(f"Pair {record['id']} missing/mismatched {key} checkpoint provenance; regenerate pairs")
            for key in ("enable_thinking", "max_prompt_tokens", "max_new_tokens", "generation", "teacher_generation"):
                if provenance.get(key) != cfg[key]:
                    raise ValueError(f"Pair {record['id']} generation protocol differs: {key}")


def tensor_record(record, device):
    ids = torch.tensor([record["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([record["labels"]], dtype=torch.long, device=device)
    return ids, labels


def sample_record(row, cfg, student, teacher, tokenizer, counters):
    prefix = render_prompt(tokenizer, row["prompt"], cfg["max_prompt_tokens"], cfg["enable_thinking"])
    if cfg["method"] == "distillm2":
        alignment = getattr(tokenizer, "_baseline_vocab_alignment", None)
        if alignment and row.get("provenance", {}).get("vocabulary_alignment") != alignment:
            raise ValueError(f"Pair {row['id']} vocabulary alignment differs; regenerate pairs")
        if row["prompt_ids"] != prefix:
            raise ValueError(f"Pair {row['id']} prompt token IDs differ from current renderer")
        output = []
        for name in ("chosen_ids", "rejected_ids"):
            response = row[name]
            if not response or len(response) > cfg["max_new_tokens"]:
                raise ValueError(f"Invalid paired response length for {row['id']}")
            if any(type(t) is not int or t < 0 or t >= unwrap(student).config.vocab_size for t in response):
                raise ValueError("Pair has out-of-range token IDs")
            check_supported(response, tokenizer, "paired response")
            output.append({"input_ids": prefix + response, "labels": [-100] * len(prefix) + response})
        return output
    if cfg["method"] == "skd":
        full, stats = skd_generate(unwrap(student), teacher, torch.tensor([prefix], device=next(student.parameters()).device),
               max_new_tokens=cfg["max_new_tokens"], eos_token_ids=stop_ids(teacher, tokenizer),
               proposal_config=SamplingConfig(**cfg["generation"]), teacher_config=SamplingConfig(**cfg["teacher_generation"]),
               acceptance_k=cfg["acceptance_k"], block_size=cfg["proposal_block_size"], return_stats=True)
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                counters[f"skd_{key}"] = counters.get(f"skd_{key}", 0) + value
        ids = full[0].tolist()
        return [{"input_ids": ids, "labels": [-100] * len(prefix) + ids[len(prefix):]}]
    return [encode_reference(tokenizer, row["prompt"], row["response"], cfg["max_prompt_tokens"], cfg["max_new_tokens"], cfg["enable_thinking"])]


def response_loss(student, teacher, record, cfg, method, reduction, counters, active=True):
    student_device = next(student.parameters()).device
    teacher_device = next(teacher.parameters()).device
    ids, labels = tensor_record(record, student_device)
    check_supported(ids, unwrap(student), "training sequence")
    alignment = getattr(unwrap(student), "_baseline_vocab_alignment", None)
    if cfg["vocabulary_policy"] != "full" and not alignment:
        raise ValueError("Shared-token training requires validated model vocabulary binding")
    if alignment != getattr(teacher, "_baseline_vocab_alignment", None):
        raise ValueError("Teacher/student vocabulary policies differ")
    if alignment:
        from .vocabulary import assert_supported_token_ids
        assert_supported_token_ids(labels, alignment, context="training targets", ignore_index=-100)
    with torch.no_grad():
        teacher_logits = teacher(input_ids=ids.to(teacher_device), attention_mask=torch.ones_like(ids, device=teacher_device), use_cache=False).logits.to(student_device)
    student_logits = student(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    if alignment:
        from .vocabulary import project_logits
        student_logits = project_logits(student_logits, alignment)
        teacher_logits = project_logits(teacher_logits, alignment)
    loss = causal_distillation_loss(student_logits, teacher_logits, labels, method=method,
             reduction=reduction, chunk_size=cfg["loss_chunk_size"], alpha=cfg["alpha"], beta=cfg["beta"],
             skew_alpha=cfg["alpha_1"] if method == "skew_forward" else cfg["alpha_2"])
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Nonfinite active-token distillation loss")
    counters["teacher_training_tokens"] += ids.numel()
    if active:
        counters["response_tokens"] += int(labels[:, 1:].ne(-100).sum())
    return loss


def evaluate(student, tokenizer, rows, cfg):
    """Optional generated-text validation; default training does not call it."""
    from .evaluation import evaluate_records
    return evaluate_records(student, tokenizer, rows, cfg, max_examples=cfg["eval_max_examples"])


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, state, rank, world):
    path = Path(path)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
        unwrap(model).save_pretrained(path / "student", safe_serialization=True)
        tokenizer.save_pretrained(path / "student")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "state": state}, path / "training.pt")
        sync_artifacts(path / "student", path / "training.pt")
        _sync_directory(path.parent)
        _sync_directory(path.parent.parent)
    if world > 1:
        dist.barrier()
    torch.save(rng_state(), path / f"rng_rank{rank}.pt")
    sync_artifacts(path / f"rng_rank{rank}.pt")
    if world > 1:
        dist.barrier()
    if rank == 0:
        write_json(path / "complete.json", {"step": state["step"], "world_size": world})


def train(config, resume=None, stop_after_steps=None):
    """Execute a resolved experiment. stop_after_steps pauses a fixed schedule for tests."""
    with _graceful_interrupt() as interruption:
        return _train(config, resume, stop_after_steps, interruption)


def _train(config, resume, stop_after_steps, interruption):
    cfg = validate_config(config)
    world, rank, local_rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if cfg["device"].startswith("cuda") else "gloo")
    device = f"cuda:{local_rank}" if world > 1 and cfg["device"].startswith("cuda") else cfg["device"]
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    teacher_device = cfg["teacher_device"] or device
    if teacher_device == "paired":
        teacher_device = f"cuda:{local_rank + world}"
    elif world > 1 and str(teacher_device).startswith("cuda"):
        raise ValueError("DDP on this profile requires teacher_device='paired' (2 GPUs per rank)")
    seed_everything(cfg["seed"] + rank)
    tokenizer = verify_tokenizers(cfg["teacher_model"], cfg["student_model"], policy=cfg["vocabulary_policy"])
    train_rows = load_records(cfg["train_file"], paired=cfg["method"] == "distillm2")
    validation = load_records(cfg["validation_file"])
    check_data(cfg, train_rows, validation)
    output = run_directory(cfg)
    manifest = build_run_manifest(cfg, world=world, tokenizer=tokenizer)
    checkpoint = Path(resume) if resume else None
    if checkpoint:
        if checkpoint.resolve().parent != (output / "checkpoints").resolve():
            raise ValueError("Resume checkpoint must belong to this run's checkpoints directory")
        if not (checkpoint / "complete.json").is_file():
            raise ValueError("Resume requires a completed checkpoint (complete.json)")
        previous = json.loads((output / "manifest.json").read_text())
        validate_run_manifest(previous, manifest)
        completion = json.loads((checkpoint / "complete.json").read_text())
        later = [p for p in (output / "checkpoints").glob("step_*")
                 if (p / "complete.json").is_file() and json.loads((p / "complete.json").read_text())["step"] > completion["step"]]
        if later:
            raise ValueError("A later completed checkpoint exists; resume the latest checkpoint")
        if rank == 0:
            # Preserve uncommitted logs/checkpoints from a crash, then resume the
            # last fully committed step. No checkpoint is silently overwritten.
            for name in ("metrics.jsonl", "validation.jsonl"):
                log_path = output / name
                if log_path.exists():
                    lines = log_path.read_text().splitlines()
                    committed = []
                    for line in lines:
                        try:
                            row = json.loads(line)
                            if row["step"] <= completion["step"]:
                                committed.append(line)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            # A killed append can leave an incomplete trailing row.
                            continue
                    if committed != lines:
                        log_path.rename(output / f"{name}.before_resume_{time.time_ns()}")
                        log_path.write_text("\n".join(committed) + "\n")
            for path in (output / "checkpoints").glob("step_*"):
                if not (path / "complete.json").is_file() and ".incomplete_" not in path.name:
                    path.rename(path.with_name(path.name + f".incomplete_{time.time_ns()}"))
    elif rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f"Output is nonempty; resume explicitly or select a new output root: {output}")
        write_json(output / "manifest.json", manifest)
    if world > 1:
        dist.barrier()

    def stop_before_training():
        if not interruption.pending(device, world):
            return False
        if rank == 0:
            write_json(output / "result.json", {"step": completion["step"] if checkpoint else 0,
                       "complete": False, "interrupted": True, "signal": interruption.signum,
                       "last_checkpoint": str(checkpoint) if checkpoint else None})
        if world > 1:
            dist.barrier()
        return True

    if stop_before_training():
        return output
    student = load_model(str(checkpoint / "student") if checkpoint else cfg["student_model"], cfg["dtype"], device, cfg["gradient_checkpointing"])
    if stop_before_training():
        return output
    teacher = load_model(cfg["teacher_model"], cfg["dtype"], teacher_device)
    if stop_before_training():
        return output
    bind_vocabulary(student, tokenizer)
    bind_vocabulary(teacher, tokenizer)
    teacher.eval().requires_grad_(False)
    if teacher.get_output_embeddings().weight.shape[0] != student.get_output_embeddings().weight.shape[0]:
        raise ValueError("Teacher/student output heads differ")
    context_limit = min(student.config.max_position_embeddings, teacher.config.max_position_embeddings)
    if cfg["max_prompt_tokens"] + cfg["max_new_tokens"] > context_limit:
        raise ValueError("Prompt + response budget exceeds model context")
    if world > 1:
        student = DistributedDataParallel(student, device_ids=[local_rank] if device.startswith("cuda") else None, broadcast_buffers=False)
    optimizer_cls = CPUAdamW if cfg["optimizer_offload"] else torch.optim.AdamW
    optimizer = optimizer_cls(student.parameters(), lr=cfg["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg["weight_decay"])
    global_batch = world * cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_rows) / global_batch)
    total_steps = cfg["max_steps"] or steps_per_epoch * cfg["epochs"]
    warmup = int(total_steps * cfg["warmup_ratio"])
    def lr_factor(step):
        if warmup and step < warmup:
            return float(step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer if cfg["optimizer_offload"] else optimizer, lr_factor)
    counters = {"teacher_training_tokens": 0, "response_tokens": 0, "unique_epoch_examples": 0}
    state = {"step": 0, "best_score": None, "best_step": None, "counters": counters, "elapsed_seconds": 0.}
    if checkpoint:
        saved = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["state"]
        if state["step"] != completion["step"] or completion["world_size"] != world:
            raise ValueError("Checkpoint completion marker and training state disagree")
        if rank == 0 and state.get("best_step") is not None:
            write_json(output / "best_checkpoint.json", {"path": f"checkpoints/step_{state['best_step']:06d}/student",
                       "step": state["best_step"], "rouge_l": state["best_score"]})
        counters = dict(state["counters"])
        # Counters in state are totals; partition the prior total before a later reduction.
        counters = {k: v / world for k, v in counters.items()}
        restore_rng(torch.load(checkpoint / f"rng_rank{rank}.pt", map_location="cpu", weights_only=False),
                    required_cuda_devices=(device, teacher_device))
    student.train()
    started = time.monotonic()
    elapsed_before = state["elapsed_seconds"]
    last_saved = None

    def save_current_step():
        nonlocal last_saved
        target = output / "checkpoints" / f"step_{state['step']:06d}"
        # A stop between steps may already have a committed checkpoint. Never
        # overwrite one or save an untrained step zero with no optimizer state.
        if state["step"] == 0 or target == last_saved or (checkpoint and target.resolve() == checkpoint.resolve()):
            return
        totals = [None] * world if world > 1 else [counters]
        if world > 1:
            dist.all_gather_object(totals, counters)
        state["counters"] = {k: sum(x.get(k, 0) for x in totals) for k in set().union(*(x.keys() for x in totals))}
        state["elapsed_seconds"] = elapsed_before + time.monotonic() - started
        if rank == 0:
            # A durable optimizer step must not outlive its corresponding logs
            # after a reboot. Only newly appended pages need storage flushing.
            sync_artifacts(*(output / name for name in ("metrics.jsonl", "validation.jsonl")
                             if (output / name).exists()))
        save_checkpoint(target, student, tokenizer, optimizer, scheduler, state, rank, world)
        last_saved = target

    for step in range(state["step"], total_steps):
        if interruption.pending(device, world):
            save_current_step()
            break
        optimizer.zero_grad()
        epoch, batch_index = divmod(step, steps_per_epoch)
        order = list(range(len(train_rows)))
        random.Random(cfg["seed"] + epoch).shuffle(order)
        indices = order[batch_index * global_batch:(batch_index + 1) * global_batch]
        local_indices = indices[rank::world]
        slots = math.ceil(len(indices) / world)
        samples = [sample_record(train_rows[i], cfg, student, teacher, tokenizer, counters) for i in local_indices]
        local_denominator = len(samples) if cfg["method"] == "distillm2" else sum(sum(t != -100 for t in sample[0]["labels"][1:]) for sample in samples)
        denominator = torch.tensor(float(local_denominator), device=device)
        if world > 1:
            dist.all_reduce(denominator)
        if denominator.item() <= 0:
            raise ValueError("No response targets in optimizer step")
        dummy = encode_reference(tokenizer, "Continue", "OK", cfg["max_prompt_tokens"], cfg["max_new_tokens"])
        step_loss = torch.zeros((), device=device)
        for slot in range(slots):
            active = slot < len(samples)
            responses = samples[slot] if active else [dummy] * (2 if cfg["method"] == "distillm2" else 1)
            for branch, response in enumerate(responses):
                last = slot == slots - 1 and branch == len(responses) - 1
                sync = student.no_sync() if world > 1 and not last else nullcontext()
                method = ("skew_forward" if branch == 0 else "skew_reverse") if cfg["method"] == "distillm2" else cfg["method"]
                reduction = "sequence_mean" if cfg["method"] == "distillm2" else "sum"
                with sync:
                    loss = response_loss(student, teacher, response, cfg, method, reduction, counters, active=active)
                    scaled = loss * (world / denominator) * int(active)
                    scaled.backward()
                step_loss += loss.detach() * int(active)
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise FloatingPointError("Student gradients are nonfinite")
        optimizer.step()
        scheduler.step()
        counters["unique_epoch_examples"] += len(local_indices)
        state["step"] = step + 1
        if world > 1:
            dist.all_reduce(step_loss)
        if rank == 0:
            log = {"step": state["step"], "epoch": epoch, "loss": float(step_loss / denominator),
                   "grad_norm": float(norm), "learning_rate": scheduler.get_last_lr()[0]}
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(log) + "\n")
            print(json.dumps(log), flush=True)
        final = state["step"] == total_steps
        paused = stop_after_steps is not None and state["step"] >= stop_after_steps
        should_eval = cfg["eval_during_training"] and (final or state["step"] % cfg["eval_steps"] == 0)
        should_save = final or paused or state["step"] % cfg["save_steps"] == 0 or should_eval
        if should_eval and rank == 0:
            metrics = evaluate(unwrap(student), tokenizer, validation, cfg)
            metrics["step"] = state["step"]
            with (output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if state["best_score"] is None or metrics["rouge_l"] > state["best_score"]:
                state["best_score"] = metrics["rouge_l"]
                state["best_step"] = state["step"]
                write_json(output / "best_checkpoint.json", {"path": f"checkpoints/step_{state['step']:06d}/student", **metrics})
        if world > 1:
            objects = [state["best_score"], state["best_step"]]
            dist.broadcast_object_list(objects, src=0)
            state["best_score"] = objects[0]
            state["best_step"] = objects[1]
        stopping = interruption.pending(device, world)
        if should_save or stopping:
            save_current_step()
        # Signals received during checkpoint serialization are deferred until
        # complete.json exists, so the coordinator can resume this exact step.
        stopping = interruption.pending(device, world)
        if stopping:
            save_current_step()
        if paused or stopping:
            break
    if rank == 0:
        result = {**state, "complete": state["step"] == total_steps,
                  "interrupted": interruption.signum is not None and state["step"] < total_steps,
                  "signal": interruption.signum,
                  "last_checkpoint": str(last_saved or checkpoint) if last_saved or checkpoint else None,
                  "peak_student_gpu_bytes_rank0": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}
        if state["step"] == total_steps:
            # A recovery can itself be killed. Revoke an older successful
            # result before touching its export so stale metadata cannot make
            # a partially rewritten model look complete on the next restart.
            write_json(output / "result.json", {**result, "complete": False, "export_pending": True})
            marker = output / "final" / "complete.json"
            if marker.exists():
                marker.unlink()
                _sync_directory(marker.parent)
            unwrap(student).save_pretrained(output / "final", safe_serialization=True)
            tokenizer.save_pretrained(output / "final")
            sync_artifacts(output / "final")
            write_json(marker, {"step": state["step"], "world_size": world})
        write_json(output / "result.json", result)
    if world > 1:
        dist.barrier()
    return output
