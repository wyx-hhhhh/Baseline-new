"""Run both complete method queues using native tiny HF models and real CLIs."""
import importlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, load_records, write_jsonl
from baseline_common.orchestration import (METHOD_ORDER, Runner, is_training_complete,
                                           latest_checkpoint, model_export_complete)
from baseline_common.pair_progress import pair_paths


WRAPPER = '''\
import json
import os
from pathlib import Path
import runpy
import sys

events, group, kind, method = sys.argv[1:5]
with Path(events).open("a") as stream:
    stream.write(json.dumps(dict(group=group, kind=kind, method=method,
                                cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\\n")
sys.argv = sys.argv[5:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
import baseline_common.models as models
original_load_model = models.load_model
def observe_load(path, dtype="bfloat16", device="cuda:0", *args, **kwargs):
    model = original_load_model(path, dtype, device, *args, **kwargs)
    with Path(events).with_name("model_devices.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(group=group, kind=kind, method=method,
                                    role=Path(path).name, requested_device=str(device),
                                    actual_device=str(next(model.parameters()).device))) + "\\n")
    return model
models.load_model = observe_load
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def _checkpoints(root, architecture):
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>",
              "one", "two", "three", "four", "answer", "short", "long", "heldout", "Continue", "OK"]
    backend = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
        bos_token="<bos>", eos_token="<eos>", additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = (
        "{{ bos_token }} {% for message in messages %}"
        "{{ '<user>' if message['role'] == 'user' else '<assistant>' }} {{ message['content'] }} "
        "{% if message['role'] == 'assistant' %}{{ eos_token }} {% endif %}"
        "{% endfor %}{% if add_generation_prompt %}<assistant> {% endif %}"
    )
    for role, width, seed in [("teacher", 16, 39), ("student", 8, 43)]:
        torch.manual_seed(seed)
        config = getattr(transformers, architecture + "Config")(
            vocab_size=32, hidden_size=width, intermediate_size=width * 2,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=width // 2, max_position_embeddings=64, bos_token_id=2,
            eos_token_id=3, pad_token_id=0, attention_dropout=0.1,
        )
        model = getattr(transformers, architecture + "ForCausalLM")(config)
        model.save_pretrained(root / role)
        tokenizer.save_pretrained(root / role)


@pytest.mark.parametrize("profile", ["cpu", "cuda"])
def test_two_real_queues_generate_train_all_methods_and_skip_finished_on_restart(tmp_path, monkeypatch, profile):
    if profile == "cuda":
        if torch.cuda.device_count() < 4:
            pytest.skip("Four visible GPUs are required for the physical two-group smoke test")
        if any(torch.cuda.get_device_capability(index)[0] < 8 for index in range(4)):
            pytest.skip("Four BF16-capable GPUs are required for the CUDA smoke profile")
    expected_dtype = torch.bfloat16 if profile == "cuda" else torch.float32
    launcher = importlib.import_module("scripts.run_parallel")
    data = tmp_path / "data" / "fixture"
    write_jsonl(data / "train.jsonl", [
        {"id": "a", "prompt": "one", "response": "short"},
        {"id": "b", "prompt": "two", "response": "long answer"},
        {"id": "c", "prompt": "three", "response": "long long answer"},
    ])
    write_jsonl(data / "validation.jsonl", [
        {"id": "v", "prompt": "heldout four", "response": "answer"},
    ])
    base = {}
    for group, architecture in [("qwen", "Qwen3"), ("llama", "Llama")]:
        root = tmp_path / "models" / group
        _checkpoints(root, architecture)
        base[group] = validate_config(dict(
            name=group, pair=group, method="kd", dataset="fixture",
            teacher_model=str(root / "teacher"), student_model=str(root / "student"),
            train_file=str(data / "train.jsonl"), validation_file=str(data / "validation.jsonl"),
            dtype="bfloat16" if profile == "cuda" else "float32",
            optimizer_offload=True, gradient_checkpointing=True,
            learning_rate=0.003, loss_chunk_size=2, acceptance_k=2, proposal_block_size=2,
            generation=dict(temperature=0.7, top_p=1.0, top_k=2),
            teacher_generation=dict(temperature=0.7, top_p=1.0, top_k=2),
        ))

    def local_config(path):
        group = "qwen" if Path(path).name.startswith("qwen") else "llama"
        method = Path(path).stem.rsplit("_", 1)[1]
        return validate_config({**base[group], "method": method, "name": f"{group}_{method}"})

    monkeypatch.setattr(launcher, "load_config", local_config)
    # Tiny models keep both profiles small. The wrapper observes actual model
    # placement as well as each process's mask without replacing computation.
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    args = SimpleNamespace(
        output_root=str(tmp_path / "outputs"), data_root=str(tmp_path / "data"), dataset="fixture",
        methods=list(METHOD_ORDER), seeds=[42], pair_seed=42,
        max_steps=1, save_steps=1, eval_steps=1, max_prompt_tokens=16, max_new_tokens=3,
        gradient_accumulation_steps=1,
    )
    groups, state = launcher.build_groups(args)
    for stages in groups.values():
        for stage in stages:
            if profile == "cpu":
                stage.config["device"] = stage.config["teacher_device"] = "cpu"
        assert [(stage.kind, stage.method) for stage in stages] == [
            ("train", "kd"), ("train", "abkd"), ("train", "skd"),
            ("pairs", "distillm2"), ("train", "distillm2"),
        ]
    wrapper = tmp_path / "record_and_run.py"
    wrapper.write_text(WRAPPER)
    events = tmp_path / "workers.jsonl"
    launches = []

    def command(stage, resume):
        original = Runner._command(stage, resume)
        launches.append((stage.key, resume))
        return [original[0], str(wrapper), str(events), stage.group, stage.kind, stage.method, *original[1:]]

    def runner():
        return Runner(groups, {"qwen": ("0", "1"), "llama": ("2", "3")}, state,
                      command_builder=command, lock_dir=tmp_path / "locks")

    assert runner().run() == 0
    assert len(launches) == 10
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    devices_path = events.with_name("model_devices.jsonl")
    model_devices = [json.loads(line) for line in devices_path.read_text().splitlines()]
    assert len(model_devices) == 20
    for row in model_devices:
        expected_device = ("cuda:1" if row["role"] == "teacher" else "cuda:0") if profile == "cuda" else "cpu"
        assert row["requested_device"] == row["actual_device"] == expected_device
    pair_snapshots = {}
    training_snapshots = {}
    training_evidence = []
    for group, stages in groups.items():
        actual = [row for row in rows if row["group"] == group]
        assert [(row["kind"], row["method"]) for row in actual] == [(s.kind, s.method) for s in stages]
        assert all(row["cuda"] == ("0,1" if group == "qwen" else "2,3") for row in actual)
        for stage in stages:
            if stage.kind == "pairs":
                paths = pair_paths(stage.output_path)
                assert len(load_records(stage.output_path, paired=True)) == 3
                manifest = json.loads(paths["manifest"].read_text())
                assert manifest["records"] == 3 and manifest["provenance"]["seed"] == 42
                for key in ("output", "teacher", "student", "manifest", "teacher_partial", "student_partial"):
                    path = paths[key]
                    pair_snapshots[path] = (file_sha256(path), path.stat().st_mtime_ns)
            else:
                assert is_training_complete(stage.output_path, expected_steps=1)
                assert model_export_complete(stage.output_path / "final")
                checkpoint = latest_checkpoint(stage.output_path)
                assert checkpoint.name == "step_000001"
                saved = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=False)
                assert saved["state"]["step"] == 1
                rng = torch.load(checkpoint / "rng_rank0.pt", map_location="cpu", weights_only=False)
                if profile == "cuda":
                    assert len(rng["cuda"]) == 2
                metrics = [json.loads(line) for line in (stage.output_path / "metrics.jsonl").read_text().splitlines()]
                assert len(metrics) == 1 and metrics[0]["step"] == 1
                assert math.isfinite(metrics[0]["loss"])
                from safetensors.torch import load_file
                exported = load_file(stage.output_path / "final/model.safetensors")
                assert all(tensor.dtype == expected_dtype and torch.isfinite(tensor).all() for tensor in exported.values())
                result = json.loads((stage.output_path / "result.json").read_text())
                if profile == "cuda":
                    assert result["peak_student_gpu_bytes_rank0"] > 0
                training_evidence.append(dict(group=group, method=stage.method, step=1,
                                              loss=metrics[0]["loss"], finite_export=True,
                                              cuda_rng_states=len(rng["cuda"]),
                                              peak_student_gpu_bytes=result["peak_student_gpu_bytes_rank0"]))
                for path in (stage.output_path / "result.json", stage.output_path / "final/model.safetensors"):
                    training_snapshots[path] = (file_sha256(path), path.stat().st_mtime_ns)
    launches.clear()
    assert runner().run() == 0
    assert len(launches) == 2 and all(key.endswith(":pairs") for key, _resume in launches)
    for path, expected in {**pair_snapshots, **training_snapshots}.items():
        assert (file_sha256(path), path.stat().st_mtime_ns) == expected
    rerun = [json.loads(line) for line in events.read_text().splitlines()][len(rows):]
    assert len(rerun) == 2 and all(row["kind"] == "pairs" for row in rerun)
    assert [json.loads(line) for line in devices_path.read_text().splitlines()] == model_devices
    for group in groups:
        log = state / "logs" / group / "distillm2_seed_42_pairs.log"
        assert "generation already complete" in log.read_text()
    evidence_path = os.environ.get("BASELINE_GPU_SMOKE_EVIDENCE")
    if profile == "cuda" and evidence_path:
        from datetime import datetime, timezone
        hardware = []
        for index in range(4):
            properties = torch.cuda.get_device_properties(index)
            hardware.append(dict(index=index, name=properties.name, total_memory_bytes=properties.total_memory,
                                 compute_capability=[properties.major, properties.minor]))
        evidence = dict(
            verified_at=datetime.now(timezone.utc).isoformat(), status="passed", profile="cuda",
            scope="Random tiny Qwen3 and Llama checkpoints, 3 training records, 1 validation record; no full-corpus or NAS model training",
            hardware=hardware, runtime={name: package_version(name) for name in ("torch", "transformers", "tokenizers")},
            cuda_runtime=torch.version.cuda, dtype="bfloat16", optimizer="CPU FP32 master AdamW",
            gpu_groups={"qwen": [0, 1], "llama": [2, 3]}, methods=list(METHOD_ORDER),
            first_launch_workers=10, completed_training_runs=8, completed_pair_producers=2,
            restart_training_workers=0, restart_pair_validation_workers=2,
            restart_preserved_artifact_hashes_and_mtimes=True, restart_loaded_models=0,
            training=training_evidence, observed_model_devices=model_devices,
        )
        destination = Path(evidence_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(evidence, indent=2) + "\n")
