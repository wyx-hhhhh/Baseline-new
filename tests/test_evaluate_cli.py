"""Real tiny HF inference plus interruption, identity and cache validation."""
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, write_jsonl
from baseline_common.pair_progress import atomic_json, atomic_jsonl, read_progress_rows


@pytest.fixture
def experiment(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(2)
    words = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>",
             "training", "first", "second", "third", "heldout", "answer", "long"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
        unk_token="<unk>", bos_token="<bos>", eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>"])
    tokenizer.chat_template = "{{ bos_token }} <user> {{ messages[0]['content'] }} <assistant>"
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64, bos_token_id=2,
        eos_token_id=3, pad_token_id=0))
    model_path = tmp_path / "final"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    train = [{"id": "train", "prompt": "training", "response": "answer", "source_group": "training-group"}]
    heldout = [{"id": word, "prompt": word + " heldout", "response": "long answer",
                "source_group": word + "-group"} for word in ("first", "second", "third")]
    train_path, validation_path = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, heldout)
    cfg = validate_config(dict(name="tiny_kd", pair="tiny", method="kd", dataset="fixture",
        # The teacher and original student are deliberately absent. Evaluation
        # requires only the actual export, including its saved tokenizer.
        teacher_model=str(tmp_path / "missing-teacher"), student_model=str(tmp_path / "missing-student"),
        train_file=str(train_path), validation_file=str(validation_path),
        dtype="float32", device="cpu", max_prompt_tokens=16, max_new_tokens=3,
        eval_max_new_tokens=3, eval_max_examples=1))
    cfg_path = tmp_path / "config.json"
    atomic_json(cfg_path, cfg)
    module = importlib.import_module("scripts.evaluate")
    output = tmp_path / "evaluation"

    def run(**kwargs):
        return module.run_evaluation(kwargs.pop("cfg", cfg), model_path=model_path,
                                     output=kwargs.pop("output", output), **kwargs)

    command = [sys.executable, str(Path(module.__file__)), "--config", str(cfg_path),
               "--model", str(model_path), "--output", str(output), "--device", "cpu", "--resume"]
    return dict(cfg=cfg, cfg_path=cfg_path, module=module, model=model_path, output=output,
                train=train, heldout=heldout, run=run, command=command,
                paths=module.evaluation_paths(output))


def _forbid_load(*_args, **_kwargs):
    raise AssertionError("Model weights must not be loaded")


def test_real_cli_scores_all_examples_and_reuses_complete_results_without_weights(experiment, monkeypatch):
    exp = experiment
    result = subprocess.run(exp["command"], text=True, capture_output=True,
                            env={**os.environ, "OMP_NUM_THREADS": "2"}, timeout=90)
    assert result.returncode == 0, result.stderr
    paths = exp["paths"]
    rows = read_progress_rows(paths["predictions"])
    metrics = json.loads(paths["metrics"].read_text())
    assert len(rows) == metrics["examples"] == 3  # ignores training eval_max_examples=1
    assert all(row["reference"] == "long answer" for row in rows)
    assert metrics["rouge_l"] == pytest.approx(sum(row["rouge_l"] for row in rows) / 3)
    assert metrics["generated_tokens"] == sum(len(row["response_ids"]) for row in rows)
    assert metrics["metric_definition"]["variant"] == "rougeL"
    assert metrics["metric_definition"]["range"] == [0, 100]
    assert metrics["model"]["path"] == str(exp["model"])
    original = {key: (file_sha256(paths[key]), paths[key].stat().st_mtime_ns)
                for key in ("predictions", "metrics", "manifest")}
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    assert exp["run"](resume=True)
    assert original == {key: (file_sha256(paths[key]), paths[key].stat().st_mtime_ns) for key in original}


def test_real_signal_interrupt_and_resume_preserve_completed_predictions(experiment, tmp_path):
    exp = experiment
    wrapper = tmp_path / "interrupt.py"
    wrapper.write_text("\n".join([
        "import os, signal, sys", f"sys.path.insert(0, {str(Path(exp['module'].__file__).resolve().parents[1])!r})",
        "import scripts.evaluate as evaluation", "original = evaluation.append_jsonl",
        "def interrupt(path, row):", "    original(path, row)",
        "    os.kill(os.getpid(), signal.SIGTERM)",
        "evaluation.append_jsonl = interrupt", "raise SystemExit(evaluation.main())", "",
    ]))
    command = [sys.executable, str(wrapper), *exp["command"][2:]]
    result = subprocess.run(command, text=True, capture_output=True,
                            env={**os.environ, "OMP_NUM_THREADS": "2"}, timeout=90)
    assert result.returncode == 75, result.stderr
    prefix = exp["paths"]["predictions"].read_bytes()
    assert len(read_progress_rows(exp["paths"]["predictions"])) == 1
    assert not exp["paths"]["manifest"].exists()
    with exp["paths"]["predictions"].open("ab") as stream:
        stream.write(b'{"id": "torn')
    result = subprocess.run(exp["command"], text=True, capture_output=True,
                            env={**os.environ, "OMP_NUM_THREADS": "2"}, timeout=90)
    assert result.returncode == 0, result.stderr
    assert exp["paths"]["predictions"].read_bytes().startswith(prefix)
    backups = list(exp["output"].glob("predictions.jsonl.truncated-*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == b'{"id": "torn'
    fresh = tmp_path / "fresh"
    assert exp["run"](output=fresh)
    assert (fresh / "predictions.jsonl").read_bytes() == exp["paths"]["predictions"].read_bytes()
    assert json.loads((fresh / "metrics.json").read_text()) == json.loads(exp["paths"]["metrics"].read_text())


def _interrupt_after_first(exp, monkeypatch):
    original = exp["module"].append_jsonl
    def interrupt(path, row):
        original(path, row)
        raise InterruptedError("simulated crash after durable append")
    monkeypatch.setattr(exp["module"], "append_jsonl", interrupt)
    with pytest.raises(InterruptedError):
        exp["run"]()
    monkeypatch.setattr(exp["module"], "append_jsonl", original)


@pytest.mark.parametrize("mutation", ["id", "prompt", "reference", "score", "token_ids", "prediction", "duplicate", "middle_json"])
def test_partial_corruption_rejected_before_loading_weights(experiment, monkeypatch, mutation):
    exp = experiment
    _interrupt_after_first(exp, monkeypatch)
    path = exp["paths"]["predictions"]
    rows = read_progress_rows(path)
    if mutation in {"id", "prompt", "reference", "prediction"}:
        rows[0][mutation] = "wrong"
    elif mutation == "score":
        rows[0]["rouge_l"] = 101
    elif mutation == "token_ids":
        rows[0]["prompt_ids"] = [99999]
    elif mutation == "duplicate":
        rows.append(rows[0])
    atomic_jsonl(path, rows)
    if mutation == "middle_json":
        with path.open("ab") as stream:
            stream.write(b"{broken\n{}\n")
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    with pytest.raises(ValueError):
        exp["run"](resume=True)


@pytest.mark.parametrize("mutation", ["dataset", "model", "runtime", "budget", "training_data"])
def test_input_identity_changes_cannot_mix_evaluations(experiment, monkeypatch, mutation):
    exp = experiment
    _interrupt_after_first(exp, monkeypatch)
    options = {}
    if mutation == "dataset":
        changed = [dict(row) for row in exp["heldout"]]
        changed[-1]["response"] = "changed answer"
        atomic_jsonl(exp["cfg"]["validation_file"], changed)
    elif mutation == "training_data":
        atomic_jsonl(exp["cfg"]["train_file"], [{**exp["train"][0], "response": "changed answer"}])
    elif mutation == "model":
        config_path = exp["model"] / "config.json"
        cfg = json.loads(config_path.read_text())
        atomic_json(config_path, {**cfg, "test_metadata_change": True})
    elif mutation == "runtime":
        monkeypatch.setattr(exp["module"], "package_version", lambda _name: "different-version")
    else:
        options["max_new_tokens"] = 4
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    with pytest.raises(ValueError, match="Evaluation inputs changed"):
        exp["run"](resume=True, **options)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("field", ["id", "prompt", "source_group"])
def test_all_input_records_checked_for_training_leakage_before_pilot_limit(experiment, monkeypatch, paired, field):
    exp = experiment
    training = exp["train"]
    cfg = exp["cfg"]
    if paired:
        training = [{**row, "prompt_ids": [2, 4, 6, 5], "chosen_ids": [11], "rejected_ids": [12],
                     "provenance": {"teacher": "t", "student": "s", "tokenizer": "s"}} for row in training]
        cfg = {**cfg, "method": "distillm2"}
        atomic_jsonl(cfg["train_file"], training)
    heldout = [dict(row) for row in exp["heldout"]]
    heldout[-1][field] = training[0][field]
    atomic_jsonl(cfg["validation_file"], heldout)
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    with pytest.raises(ValueError, match="leakage"):
        exp["run"](cfg=cfg, max_examples=1)


@pytest.mark.parametrize("corruption", ["missing", "hash", "rehash_semantics", "rehash_metrics"])
def test_completed_results_require_checksums_and_semantics(experiment, monkeypatch, corruption):
    exp = experiment
    assert exp["run"]()
    paths = exp["paths"]
    if corruption == "missing":
        paths["metrics"].unlink()
    elif corruption == "hash":
        with paths["predictions"].open("a") as stream:
            stream.write("\n")
    else:
        key = "predictions" if corruption == "rehash_semantics" else "metrics"
        if key == "predictions":
            rows = read_progress_rows(paths[key])
            rows[0]["reference"] = "edited reference"
            atomic_jsonl(paths[key], rows)
        else:
            metrics = json.loads(paths[key].read_text())
            atomic_json(paths[key], {**metrics, "rouge_l": 999})
        manifest = json.loads(paths["manifest"].read_text())
        manifest["files"][paths[key].name] = file_sha256(paths[key])
        atomic_json(paths["manifest"], manifest)
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    with pytest.raises(ValueError):
        exp["run"](resume=True)


def test_publish_interruption_resumes_without_regenerating(experiment, monkeypatch):
    exp = experiment
    original = exp["module"].atomic_json
    def interrupt(path, document):
        if path == exp["paths"]["manifest"]:
            raise InterruptedError("publication interrupted")
        return original(path, document)
    monkeypatch.setattr(exp["module"], "atomic_json", interrupt)
    with pytest.raises(InterruptedError):
        exp["run"]()
    monkeypatch.setattr(exp["module"], "atomic_json", original)
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    assert exp["run"](resume=True)


def test_output_requires_resume_and_exclusive_writer(experiment):
    exp = experiment
    with exp["module"].evaluation_lock(exp["paths"]["lock"]):
        with pytest.raises(RuntimeError, match="Another evaluation"):
            exp["run"]()
    assert exp["run"]()
    with pytest.raises(FileExistsError, match="--resume"):
        exp["run"]()


def test_shared_policy_must_be_restored_from_export_not_teacher(experiment, monkeypatch):
    exp = experiment
    monkeypatch.setattr(exp["module"], "load_model", _forbid_load)
    with pytest.raises(ValueError, match="Saved vocabulary policy"):
        exp["run"](cfg={**exp["cfg"], "vocabulary_policy": "llama3_shared"})
