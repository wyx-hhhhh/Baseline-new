"""Two genuine tiny-model campaigns followed by separate resumable evaluation."""
import importlib
import csv
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, write_jsonl
from baseline_common.models import load_tokenizer
from baseline_common.orchestration import METHOD_ORDER, PAIR_NAMES, Runner
from test_parallel_real_pipeline import _checkpoints
from test_training import _paired_config


EVALUATION_WRAPPER = '''\
import json
import os
from pathlib import Path
import runpy
import sys

events, group, method = sys.argv[1:4]
with Path(events).open("a") as stream:
    stream.write(json.dumps(dict(group=group, method=method,
                                cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\\n")
sys.argv = sys.argv[4:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
import baseline_common.models as models
original_load = models.load_model
def observe_load(path, *args, **kwargs):
    assert Path(path).name == "final", "Evaluation must only load the exported student"
    model = original_load(path, *args, **kwargs)
    with Path(events).with_name("loaded_models.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(group=group, method=method, path=str(path),
                                    device=str(next(model.parameters()).device),
                                    dtype=str(next(model.parameters()).dtype))) + "\\n")
    return model
models.load_model = observe_load
runpy.run_path(sys.argv[0], run_name="__main__")
'''


@pytest.fixture(scope="module")
def trained_campaign(tmp_path_factory):
    from baseline_common.train import train

    root = tmp_path_factory.mktemp("evaluation_campaign")
    data = root / "data" / "fixture"
    training_rows = [
        {"id": "a", "prompt": "one", "response": "short"},
        {"id": "b", "prompt": "two", "response": "long answer"},
    ]
    validation_rows = [
        {"id": "v1", "prompt": "heldout three", "response": "answer"},
        {"id": "v2", "prompt": "heldout four", "response": "short answer"},
    ]
    write_jsonl(data / "train.jsonl", training_rows)
    write_jsonl(data / "validation.jsonl", validation_rows)
    runs, configs = {}, {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        for group, architecture in (("qwen", "Qwen3"), ("llama", "Llama")):
            model_root = root / "models" / group
            _checkpoints(model_root, architecture)
            tokenizer = load_tokenizer(model_root / "student")
            base = validate_config(dict(
                name=f"{group}_tiny", pair=PAIR_NAMES[group], method="kd", dataset="fixture",
                teacher_model=str(model_root / "teacher"), student_model=str(model_root / "student"),
                train_file=str(data / "train.jsonl"), validation_file=str(data / "validation.jsonl"),
                output_root=str(root / "outputs"), dtype="float32", device="cpu", teacher_device="cpu",
                optimizer_offload=True, gradient_checkpointing=True,
                max_prompt_tokens=16, max_new_tokens=3, eval_max_new_tokens=3,
                max_steps=1, gradient_accumulation_steps=1, learning_rate=0.003,
                save_steps=1, eval_steps=1, loss_chunk_size=2, acceptance_k=2, proposal_block_size=2,
                generation=dict(temperature=0.7, top_p=1.0, top_k=2),
                teacher_generation=dict(temperature=0.7, top_p=1.0, top_k=2),
            ))
            for method in METHOD_ORDER:
                cfg = {**base, "method": method, "name": f"{group}_{method}"}
                if method == "distillm2":
                    cfg = _paired_config(cfg, tokenizer, training_rows)
                    generated_pairs = Path(cfg["train_file"])
                    family_pairs = generated_pairs.with_name(f"pairs_{group}.jsonl")
                    generated_pairs.rename(family_pairs)
                    cfg["train_file"] = str(family_pairs)
                output = train(cfg)
                assert json.loads((output / "result.json").read_text())["complete"]
                assert not (output / "validation.jsonl").exists()
                runs[group, method], configs[group, method] = output, cfg
            # Successful evaluation must use the final student export alone.
            (model_root / "teacher").rename(model_root / "teacher_unavailable")
    finally:
        torch.set_num_threads(previous_threads)
    return {"root": root, "runs": runs, "configs": configs,
            "data": data, "output_root": root / "outputs"}


def _arguments(campaign, **changes):
    values = dict(output_root=str(campaign["output_root"]), data_root=str(campaign["root"] / "data"),
                  dataset="fixture", seeds=[42], methods=list(METHOD_ORDER), checkpoint="final",
                  device="cpu", dtype=None, limit=None, max_new_tokens=None,
                  max_prompt_tokens=None, rouge_tokenizer=None, dry_run=False,
                  evaluation_root=None,
                  qwen_gpus=("0", "1"), llama_gpus=("2", "3"))
    values.update(changes)
    return SimpleNamespace(**values)


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.mark.parametrize("profile", ["cpu", "cuda"])
def test_two_family_evaluation_queues_score_all_baselines_and_restart_without_model_loads(trained_campaign, monkeypatch, profile):
    if profile == "cuda":
        if torch.cuda.device_count() < 4:
            pytest.skip("Four visible GPUs required to verify both physical evaluation groups")
        if any(torch.cuda.get_device_capability(index)[0] < 8 for index in range(4)):
            pytest.skip("Four BF16-capable GPUs required for native CUDA evaluation")
    launcher = importlib.import_module("scripts.run_evaluation")
    campaign = trained_campaign
    artifacts = campaign["root"] / f"observe_{profile}"
    artifacts.mkdir()
    events = artifacts / "evaluation_workers.jsonl"
    wrapper = artifacts / "evaluation_wrapper.py"
    wrapper.write_text(EVALUATION_WRAPPER)
    launches = []
    device = "cuda:0" if profile == "cuda" else "cpu"
    dtype = "bfloat16" if profile == "cuda" else "float32"
    evaluation_root = campaign["root"] / "gpu_evaluations" if profile == "cuda" else campaign["output_root"]

    class ObservedRunner(Runner):
        def __init__(self, *args, **kwargs):
            original_builder = kwargs.pop("command_builder", None) or Runner._command

            def observed_command(stage, resume):
                command = original_builder(stage, resume)
                launches.append(stage.key)
                return [command[0], str(wrapper), str(events), stage.group, stage.method, *command[1:]]

            kwargs.update(command_builder=observed_command, lock_dir=artifacts / "locks")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(launcher, "Runner", ObservedRunner)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    argv = [
        "run_evaluation.py", "--output-root", str(campaign["output_root"]),
        "--data-root", str(campaign["root"] / "data"), "--dataset", "fixture", "--device", device,
    ]
    if profile == "cuda":
        argv += ["--evaluation-root", str(evaluation_root), "--dtype", dtype]
    monkeypatch.setattr(sys, "argv", argv)
    assert launcher.main() == 0
    assert len(launches) == 8
    groups, state = launcher.build_evaluation_groups(_arguments(campaign, device=device,
        dtype=dtype if profile == "cuda" else None,
        evaluation_root=str(evaluation_root) if profile == "cuda" else None))
    assert state.is_relative_to(evaluation_root)
    workers, loaded = _rows(events), _rows(events.with_name("loaded_models.jsonl"))
    assert len(workers) == len(loaded) == 8
    assert all(row["device"] == device and row["dtype"] == f"torch.{dtype}"
               and Path(row["path"]).name == "final" for row in loaded)
    snapshots = {}
    for group, stages in groups.items():
        assert [stage.method for stage in stages] == list(METHOD_ORDER)
        assert [row["method"] for row in workers if row["group"] == group] == list(METHOD_ORDER)
        assert all(row["cuda"] == ("0,1" if group == "qwen" else "2,3")
                   for row in workers if row["group"] == group)
        for stage in stages:
            assert stage.kind == "evaluate"
            assert stage.output_path.is_relative_to(evaluation_root)
            assert stage.model_path.is_relative_to(campaign["output_root"])
            assert stage.config["output_root"] == str(campaign["output_root"])
            metrics = json.loads((stage.output_path / "metrics.json").read_text())
            assert 0 <= metrics["rouge_l"] <= 100 and "reference_nll" not in metrics
            assert metrics["examples"] == 2
            assert metrics["training_run"]["method"] == stage.method
            assert metrics["metric_definition"]["name"] == "rouge_l"
            assert metrics["metric_definition"]["higher_is_better"] is True
            predictions = _rows(stage.output_path / "predictions.jsonl")
            assert [row["id"] for row in predictions] == ["v1", "v2"]
            assert metrics["rouge_l"] == pytest.approx(sum(row["rouge_l"] for row in predictions) / 2)
            for filename in ("metrics.json", "predictions.jsonl", "manifest.json"):
                path = stage.output_path / filename
                snapshots[path] = (file_sha256(path), path.stat().st_mtime_ns)
    summary_path = evaluation_root / "evaluations/fixture/summary_final.json"
    summary = json.loads(summary_path.read_text())
    assert summary["metric"] == "rouge_l" and summary["range"] == [0, 100]
    assert summary["higher_is_better"] is True
    results = summary["results"]
    assert len(results) == 8
    assert {(row["pair"], row["method"]) for row in results} == {
        (pair, method) for pair in PAIR_NAMES.values() for method in METHOD_ORDER
    }
    with summary_path.with_suffix(".csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 8 and "rouge_l" in csv_rows[0]
    for row, csv_row in zip(results, csv_rows):
        measured = json.loads(Path(row["metrics"]).read_text())
        assert row["rouge_l"] == measured["rouge_l"] == float(csv_row["rouge_l"])
    before_models = events.with_name("loaded_models.jsonl").read_bytes()
    launches.clear()
    assert launcher.main() == 0
    assert len(launches) == 8  # Workers validate durable results before skipping inference.
    assert events.with_name("loaded_models.jsonl").read_bytes() == before_models
    for path, expected in snapshots.items():
        assert (file_sha256(path), path.stat().st_mtime_ns) == expected
    assert len(_rows(events)) == 16


def test_evaluation_campaign_requires_complete_training_before_inference(trained_campaign):
    launcher = importlib.import_module("scripts.run_evaluation")
    run = trained_campaign["runs"]["qwen", "kd"]
    path = run / "result.json"
    original = path.read_bytes()
    result = json.loads(original)
    result["complete"] = False
    path.write_text(json.dumps(result))
    try:
        with pytest.raises((ValueError, RuntimeError), match="complet|train"):
            launcher.build_evaluation_groups(_arguments(trained_campaign, methods=["kd"]))
    finally:
        path.write_bytes(original)


def test_evaluation_campaign_rejects_training_dataset_changed_since_training(trained_campaign):
    launcher = importlib.import_module("scripts.run_evaluation")
    path = trained_campaign["data"] / "train.jsonl"
    original = path.read_bytes()
    path.write_bytes(original + b'\n')
    try:
        with pytest.raises(ValueError, match="chang|hash|training data"):
            launcher.build_evaluation_groups(_arguments(trained_campaign, methods=["kd"]))
    finally:
        path.write_bytes(original)


def test_evaluation_campaign_rejects_unsafe_best_checkpoint_pointer(trained_campaign):
    launcher = importlib.import_module("scripts.run_evaluation")
    run = trained_campaign["runs"]["qwen", "kd"]
    pointer = run / "best_checkpoint.json"
    assert not pointer.exists()
    pointer.write_text(json.dumps({"path": "../../outside/student", "step": 1, "rouge_l": 100}))
    try:
        with pytest.raises((ValueError, RuntimeError), match="checkpoint|path|best|ROUGE"):
            launcher.build_evaluation_groups(_arguments(trained_campaign, methods=["kd"], checkpoint="best"))
    finally:
        pointer.unlink()


def test_evaluation_campaign_rejects_uncommitted_best_checkpoint(trained_campaign):
    launcher = importlib.import_module("scripts.run_evaluation")
    run = trained_campaign["runs"]["qwen", "kd"]
    partial = run / "checkpoints/step_000002"
    pointer = run / "best_checkpoint.json"
    assert not pointer.exists() and not partial.exists()
    # Weights alone do not prove a complete optimizer step. The genuine step-1
    # final remains complete while this interrupted export must be rejected.
    shutil.copytree(run / "final", partial / "student")
    pointer.write_text(json.dumps({"path": "checkpoints/step_000002/student", "step": 2, "rouge_l": 100}))
    try:
        with pytest.raises(ValueError, match="committed step"):
            launcher.build_evaluation_groups(_arguments(trained_campaign, methods=["kd"], checkpoint="best"))
    finally:
        pointer.unlink()
        shutil.rmtree(partial)
