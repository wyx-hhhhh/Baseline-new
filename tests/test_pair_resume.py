"""Pair producer restart tests exercise the real CLI orchestration and RNG seeds."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, load_records, write_jsonl
from baseline_common.pair_progress import (atomic_json, atomic_jsonl, pair_lock,
                                          pair_paths, read_progress_rows)


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1, ord(messages[0]["content"][-1]), 2]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class TinyModel:
    def __init__(self, role):
        self.role = role

    def eval(self):
        return self

    def requires_grad_(self, enabled):
        return self


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    module = importlib.import_module("scripts.generate_pairs")
    for role in ("teacher", "student"):
        directory = tmp_path / role
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"vocab_size": 256}))
        (directory / "tokenizer.json").write_text(json.dumps({"fixture": True}))
        (directory / "model.safetensors").write_bytes(b"fixture")
    rows = [{"id": letter, "prompt": f"prompt {letter}", "response": "answer"} for letter in "abc"]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "validation.jsonl", [{"id": "heldout", "prompt": "heldout z", "response": "answer"}])
    cfg = validate_config(dict(
        name="pairs", pair="tiny", method="distillm2", dataset="fixture",
        teacher_model=str(tmp_path / "teacher"), student_model=str(tmp_path / "student"),
        train_file=str(tmp_path / "pairs.jsonl"), validation_file=str(tmp_path / "validation.jsonl"),
        device="cpu", teacher_device="cpu", dtype="float32", max_new_tokens=4, max_prompt_tokens=16,
    ))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    events = {"loads": [], "generations": [], "fail": None}

    def load_model(path, dtype, device):
        role = Path(path).name
        events["loads"].append((role, device))
        return TinyModel(role)

    def generate(model, tokenizer, prompt_ids, max_new_tokens, generation):
        key = (model.role, chr(prompt_ids[1]))
        events["generations"].append(key)
        if events["fail"] == key:
            raise InterruptedError("simulated interruption")
        return torch.randint(4, 64, (max_new_tokens,)).tolist()

    monkeypatch.setattr(module, "verify_tokenizers", lambda *args, **kwargs: TinyTokenizer())
    monkeypatch.setattr(module, "load_model", load_model)
    monkeypatch.setattr(module, "generate_response", generate)
    monkeypatch.setattr(module, "bind_vocabulary", lambda *args: None)

    def run(*, config=None, output=None, resume=False, max_examples=None):
        return module.run_generation(config or cfg, config_path=config_path,
                                     input_path=tmp_path / "train.jsonl",
                                     output=output or cfg["train_file"], resume=resume,
                                     max_examples=max_examples)

    return cfg, events, run, pair_paths(cfg["train_file"]), module


@pytest.mark.parametrize("role", ["teacher", "student"])
def test_interrupted_role_resumes_only_missing_rows_with_identical_tokens(experiment, role, tmp_path):
    cfg, events, run, paths, module = experiment
    events["fail"] = (role, "b")
    with pytest.raises(InterruptedError):
        run()
    assert not paths["manifest"].exists()
    assert [row["id"] for row in read_progress_rows(paths[f"{role}_partial"])] == ["a"]
    with paths[f"{role}_partial"].open("ab") as stream:
        stream.write(b'{"id": "b", "response_ids": [')
    events["fail"] = None
    events["generations"].clear()
    events["loads"].clear()
    run(resume=True)
    assert (role, "a") not in events["generations"]
    assert (role, "b") in events["generations"]
    assert (role, "c") in events["generations"]
    if role == "student":
        assert events["loads"] == [("student", "cpu")]
    backup = list(tmp_path.glob(f"pairs.{role}.partial.jsonl.truncated-*.bak"))
    assert len(backup) == 1
    assert backup[0].read_bytes() == b'{"id": "b", "response_ids": ['
    uninterrupted = run(output=tmp_path / "uninterrupted.jsonl")
    resumed_rows = load_records(paths["output"], paired=True)
    clean_rows = load_records(uninterrupted, paired=True)
    assert [{key: row[key] for key in ("id", "prompt_ids", "chosen_ids", "rejected_ids")}
            for row in resumed_rows] == [
        {key: row[key] for key in ("id", "prompt_ids", "chosen_ids", "rejected_ids")}
        for row in clean_rows]


def test_complete_resume_ignores_training_only_settings_and_never_loads_models(experiment, tmp_path):
    cfg, events, run, paths, module = experiment
    run()
    hashes = {name: file_sha256(paths[name]) for name in ("teacher", "student", "output", "manifest")}
    changed = {**cfg, "learning_rate": 0.003, "output_root": str(tmp_path / "new-runs"),
               "name": "renamed", "device": "cuda:0", "teacher_device": "cuda:1"}
    events["loads"].clear()
    events["generations"].clear()
    run(config=changed, resume=True)
    assert not events["loads"] and not events["generations"]
    assert hashes == {name: file_sha256(paths[name]) for name in hashes}
    with pytest.raises(FileExistsError, match="--resume"):
        run()


@pytest.mark.parametrize("field,value", [
    ("seed", 123), ("max_prompt_tokens", 15), ("max_new_tokens", 3), ("dtype", "bfloat16"),
    ("generation", {"temperature": 0.3, "top_k": 0, "top_p": 1.0}),
    ("teacher_generation", {"temperature": 0.3, "top_k": 0, "top_p": 1.0}),
    ("vocabulary_policy", "llama3_shared"),
])
@pytest.mark.parametrize("complete", [False, True])
def test_generation_inputs_cannot_change_on_resume(experiment, field, value, complete):
    cfg, events, run, paths, module = experiment
    if complete:
        run()
    else:
        events["fail"] = ("teacher", "b")
        with pytest.raises(InterruptedError):
            run()
    events["loads"].clear()
    with pytest.raises(ValueError, match="inputs changed"):
        run(config={**cfg, field: value}, resume=True)
    assert not events["loads"]


@pytest.mark.parametrize("change", ["source", "validation", "model", "max_examples"])
def test_external_identity_and_pilot_limit_are_frozen(experiment, tmp_path, change):
    cfg, events, run, paths, module = experiment
    events["fail"] = ("teacher", "b")
    with pytest.raises(InterruptedError):
        run()
    options = {}
    if change == "source":
        (tmp_path / "train.jsonl").write_text((tmp_path / "train.jsonl").read_text() + "\n")
    elif change == "validation":
        (tmp_path / "validation.jsonl").write_text((tmp_path / "validation.jsonl").read_text() + "\n")
    elif change == "model":
        (tmp_path / "teacher" / "model.safetensors").write_bytes(b"modified checkpoint")
    else:
        options["max_examples"] = 2
    with pytest.raises(ValueError, match="inputs changed"):
        run(resume=True, **options)


@pytest.mark.parametrize("corruption", ["id", "prompt", "source_group", "prompt_ids", "response_ids",
                                        "response_text", "duplicate", "invalid_middle", "excluded_token"])
def test_partial_rows_are_strictly_validated_before_new_generation(experiment, corruption):
    cfg, events, run, paths, module = experiment
    events["fail"] = ("student", "a")
    with pytest.raises(InterruptedError):
        run()
    rows = read_progress_rows(paths["teacher_partial"])
    if corruption == "id":
        rows[0]["id"] = "unknown"
    elif corruption == "prompt":
        rows[0]["prompt"] = "modified prompt"
    elif corruption == "source_group":
        rows[0]["source_group"] = "unexpected source"
    elif corruption == "prompt_ids":
        rows[0]["prompt_ids"] = [1, 3, 2]
    elif corruption == "response_ids":
        rows[0]["response_ids"] = [256]
    elif corruption == "response_text":
        rows[0]["response"] = "edited"
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif corruption == "excluded_token":
        rows[0]["response_ids"] = [True]
    atomic_jsonl(paths["teacher_partial"], rows)
    if corruption == "invalid_middle":
        lines = paths["teacher_partial"].read_text().splitlines()
        paths["teacher_partial"].write_text(lines[0] + "\n{broken\n" + lines[2] + "\n")
    events["loads"].clear()
    with pytest.raises(ValueError):
        run(resume=True)
    assert not events["loads"]
    assert not list(paths["output"].parent.glob("*.bak"))


def test_interrupted_publication_resumes_without_regeneration(experiment, monkeypatch):
    cfg, events, run, paths, module = experiment
    original = module.atomic_json

    def interrupt_marker(path, document):
        if path == paths["manifest"]:
            raise InterruptedError("killed just before publishing completion marker")
        return original(path, document)

    monkeypatch.setattr(module, "atomic_json", interrupt_marker)
    with pytest.raises(InterruptedError):
        run()
    assert paths["output"].exists() and not paths["manifest"].exists()
    monkeypatch.setattr(module, "atomic_json", original)
    events["loads"].clear()
    run(resume=True)
    assert not events["loads"]
    assert paths["manifest"].exists()


@pytest.mark.parametrize("corruption", ["output", "teacher", "missing_student", "rehash_prompt", "rehash_pair"])
def test_completed_artifacts_require_checksums_and_semantics(experiment, corruption):
    cfg, events, run, paths, module = experiment
    run()
    if corruption == "missing_student":
        paths["student"].unlink()
    elif corruption.startswith("rehash"):
        key = "teacher" if corruption == "rehash_prompt" else "output"
        rows = read_progress_rows(paths[key])
        if key == "teacher":
            rows[0]["prompt_ids"] = [1, 4, 2]
        else:
            rows[0]["chosen_ids"] = [4]
        atomic_jsonl(paths[key], rows)
        manifest = json.loads(paths["manifest"].read_text())
        manifest["files"][paths[key].name] = file_sha256(paths[key])
        atomic_json(paths["manifest"], manifest)
    else:
        paths[corruption].write_text(paths[corruption].read_text() + "\n")
    events["loads"].clear()
    with pytest.raises(ValueError):
        run(resume=True)
    assert not events["loads"]


def test_legacy_complete_manifest_is_reused_without_full_config_hash_match(experiment, tmp_path, capsys):
    cfg, events, run, paths, module = experiment
    run()
    manifest = json.loads(paths["manifest"].read_text())
    manifest["provenance"].pop("generation_dtype")
    manifest["provenance"].pop("generation_runtime")
    rows = load_records(paths["output"], paired=True)
    for row in rows:
        row["provenance"] = manifest["provenance"]
    atomic_jsonl(paths["output"], rows)
    manifest["files"][paths["output"].name] = file_sha256(paths["output"])
    atomic_json(paths["manifest"], manifest)
    for key in ("progress", "teacher_partial", "student_partial"):
        paths[key].unlink()
    events["loads"].clear()
    run(config={**cfg, "learning_rate": 0.002, "output_root": str(tmp_path / "changed")}, resume=True)
    assert not events["loads"]
    assert "generation runtime is historically unknown" in capsys.readouterr().out


@pytest.mark.parametrize("complete", [False, True])
def test_generation_runtime_versions_cannot_change_on_resume(experiment, monkeypatch, complete):
    cfg, events, run, paths, module = experiment
    if complete:
        run()
    else:
        events["fail"] = ("teacher", "b")
        with pytest.raises(InterruptedError):
            run()
    manifest = json.loads(paths["manifest" if complete else "progress"].read_text())
    runtime = manifest["provenance"]["generation_runtime"]
    assert set(runtime) == {"torch", "transformers", "tokenizers"}
    original_version = module.package_version
    monkeypatch.setattr(module, "package_version",
                        lambda package: "changed-runtime" if package == "torch" else original_version(package))
    events["loads"].clear()
    with pytest.raises(ValueError, match="inputs changed: generation_runtime"):
        run(resume=True)
    assert not events["loads"]


def test_partial_generation_with_unknown_runtime_cannot_mix_new_rows(experiment):
    cfg, events, run, paths, module = experiment
    events["fail"] = ("teacher", "b")
    with pytest.raises(InterruptedError):
        run()
    progress = json.loads(paths["progress"].read_text())
    progress["provenance"].pop("generation_runtime")
    atomic_json(paths["progress"], progress)
    events["loads"].clear()
    with pytest.raises(ValueError, match="inputs changed: generation_runtime"):
        run(resume=True)
    assert not events["loads"]


def test_output_lock_prevents_concurrent_pair_writers(experiment):
    cfg, events, run, paths, module = experiment
    with pair_lock(paths["lock"]):
        with pytest.raises(RuntimeError, match="Another pair generator"):
            run(resume=True)
    assert not events["loads"]


def test_missing_newline_is_completed_without_discarding_valid_row(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text('{"id": "a"}')
    assert read_progress_rows(path, recover_tail=True) == [{"id": "a"}]
    assert path.read_bytes() == b'{"id": "a"}\n'
    assert not list(tmp_path.glob("*.bak"))


def test_cli_resume_switch_and_device_override(experiment, monkeypatch, tmp_path):
    cfg, events, run, paths, module = experiment
    monkeypatch.setattr(sys, "argv", [
        "generate_pairs.py", "--config", str(tmp_path / "config.json"),
        "--input", str(tmp_path / "train.jsonl"), "--output", str(paths["output"]),
        "--resume", "--device", "cpu",
    ])
    module.main()
    assert events["loads"] == [("teacher", "cpu"), ("student", "cpu")]
    events["loads"].clear()
    module.main()
    assert not events["loads"]
