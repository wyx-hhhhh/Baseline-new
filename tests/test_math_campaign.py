"""Verify frozen math campaign wiring and real tiny CPU training/evaluation."""
from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from baseline_common.data import file_sha256, load_records
from baseline_common.math_data import prepare_math_pool
from baseline_common.orchestration import METHOD_ORDER, PAIR_NAMES, is_training_complete


ROOT = Path(__file__).resolve().parents[1]
GPU_GROUPS = {"qwen": ("0", "1"), "llama": ("2", "3")}


def arguments(**overrides):
    values = dict(phase="train", group="all", methods=list(METHOD_ORDER), seeds=[42],
                  dry_run=True, device=None, limit=None, output_root=None, pool_dir=None,
                  campaign=str(ROOT / "configs/math_campaign.json"),
                  qwen_gpus=("0", "1"), llama_gpus=("2", "3"))
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def frozen_fixture_pool(tmp_path):
    sources = {}
    data = {
        "metamath": [{"query": word, "response": "#### 1", "original_question": word}
                     for word in ("one", "two", "three", "four", "short", "long")],
        "gsm8k": [{"question": word + " heldout", "answer": "#### 1"}
                  for word in ("first", "second")],
        "math": [{"problem": word + " math heldout", "solution": r"\boxed{1}",
                  "type": "Algebra", "level": "Level 1"} for word in ("third", "fourth")],
    }
    for name, rows in data.items():
        sources[name] = tmp_path / f"{name}.json"
        sources[name].write_text(json.dumps(rows))
    output = tmp_path / "fixed-pool"
    manifest = prepare_math_pool(sources["metamath"], sources["gsm8k"], sources["math"], output,
                                 train_size=3, validation_size=1,
                                 expected_gsm8k_count=2, expected_math_count=2)
    return output, manifest


def snapshots(paths):
    return {path: (file_sha256(path), path.stat().st_mtime_ns) for path in paths}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_all_eight_math_configs_bind_same_pool_and_keep_dolly_separate():
    from baseline_common.config import load_config
    from baseline_common.math_data import DEFAULT_MATH_POOL
    paths = sorted((ROOT / "configs/math").glob("*.json"))
    assert len(paths) == 8
    configs = [load_config(path) for path in paths]
    assert {(cfg["pair"], cfg["method"]) for cfg in configs} == {
        (pair, method) for pair in PAIR_NAMES.values() for method in METHOD_ORDER}
    for cfg in configs:
        assert cfg["evaluation_metric"] == "math_accuracy"
        assert cfg["eval_during_training"] is False
        assert cfg["dataset"] != "dolly"
        assert Path(cfg["validation_file"]) == DEFAULT_MATH_POOL / "validation.jsonl"
        if cfg["method"] != "distillm2":
            assert Path(cfg["train_file"]) == DEFAULT_MATH_POOL / "train.jsonl"
        assert "/dolly/" not in cfg["train_file"]
        original = load_config(ROOT / "configs/experiments" / f"{cfg['pair']}_{cfg['method']}.json")
        assert original["dataset"] == "dolly"
        assert original["evaluation_metric"] == "rouge_l"
        assert cfg["student_model"] == original["student_model"]
        assert cfg["teacher_model"] == original["teacher_model"]
    shared = {cfg["train_file"] for cfg in configs if cfg["method"] != "distillm2"}
    assert shared == {str(DEFAULT_MATH_POOL / "train.jsonl")}
    # These controls define the common comparison, including equal schedules
    # and fresh initialization for each of the four methods in a model group.
    controls = ("seed", "dtype", "optimizer_offload", "max_prompt_tokens",
                "max_new_tokens", "learning_rate", "weight_decay", "epochs",
                "max_steps", "gradient_accumulation_steps", "warmup_ratio",
                "max_grad_norm", "gradient_checkpointing", "enable_thinking")
    assert len({tuple(cfg[key] for key in controls) for cfg in configs}) == 1
    assert all(cfg["epochs"] == 1 and cfg["max_steps"] is None for cfg in configs)
    for pair in PAIR_NAMES.values():
        pair_configs = [cfg for cfg in configs if cfg["pair"] == pair]
        assert len({cfg["student_model"] for cfg in pair_configs}) == 1
        assert len({cfg["teacher_model"] for cfg in pair_configs}) == 1
        assert all(Path(cfg["student_model"]).parent == Path("/nas/Models") for cfg in pair_configs)


def test_campaign_freezes_exact_training_development_and_full_benchmark_counts():
    launcher = importlib.import_module("scripts.run_math")
    campaign = launcher.load_campaign(ROOT / "configs/math_campaign.json")
    assert campaign["pool_counts"] == {
        "train": 50000, "validation": 5000, "gsm8k": 1319, "math": 5000}
    pool, manifest = launcher.check_pool(campaign)
    assert file_sha256(pool / "manifest.json") == campaign["pool_manifest_sha256"]
    assert {name: split["records"] for name, split in manifest["splits"].items()} == campaign["pool_counts"]
    assert campaign["evaluation"]["benchmarks"] == ["gsm8k", "math"]


def test_default_campaign_plans_eight_training_runs_and_sixteen_independent_tests():
    from baseline_common.config import run_directory
    from baseline_common.math_data import DEFAULT_MATH_POOL
    launcher = importlib.import_module("scripts.run_math")
    original_paths = sorted((ROOT / "configs/experiments").glob("*.json"))
    before = snapshots(original_paths)
    groups, state = launcher.build_math_groups(arguments())
    evaluations, eval_state = launcher.build_math_groups(arguments(phase="evaluate"))
    assert set(groups) == set(evaluations) == {"qwen", "llama"}
    assert sum(stage.kind == "train" for stages in groups.values() for stage in stages) == 8
    assert sum(len(stages) for stages in evaluations.values()) == 16
    assert state != eval_state
    for group, stages in groups.items():
        assert [(stage.kind, stage.method) for stage in stages] == [
            ("train", "kd"), ("train", "abkd"), ("train", "skd"),
            ("pairs", "distillm2"), ("train", "distillm2")]
        pair_stage = next(stage for stage in stages if stage.kind == "pairs")
        assert pair_stage.input_path == DEFAULT_MATH_POOL / "train.jsonl"
        paired_training = next(stage for stage in stages if stage.kind == "train" and stage.method == "distillm2")
        assert pair_stage.output_path == Path(paired_training.config["train_file"])
        for training in (stage for stage in stages if stage.kind == "train"):
            corresponding = [stage for stage in evaluations[group] if stage.config["method"] == training.method]
            assert len(corresponding) == 2
            assert {stage.input_path for stage in corresponding} == {
                DEFAULT_MATH_POOL / "tests/gsm8k.jsonl", DEFAULT_MATH_POOL / "tests/math.jsonl"}
            assert {stage.model_path for stage in corresponding} == {run_directory(training.config) / "final"}
            assert len({stage.output_path for stage in corresponding}) == 2
            assert len({stage.key for stage in corresponding}) == 2
            assert all(stage.config["evaluation_metric"] == "math_accuracy" for stage in corresponding)
            assert training.config["device"] == "cuda:0"
            assert training.config["teacher_device"] == "cuda:1"
            assert "dolly" not in str(training.output_path)
            assert training.output_path.is_relative_to(Path("/nas/Users/wyx/Baseline/math"))
    assert snapshots(original_paths) == before


@pytest.mark.parametrize("group,expected_mask", [("qwen", "0,1"), ("llama", "2,3")])
def test_group_selection_preserves_baseline_order_and_gpu_environment(group, expected_mask):
    from baseline_common.orchestration import child_environment
    launcher = importlib.import_module("scripts.run_math")
    groups, _ = launcher.build_math_groups(arguments(group=group))
    assert list(groups) == [group]
    assert [stage.method for stage in groups[group] if stage.kind == "train"] == list(METHOD_ORDER)
    assert child_environment(GPU_GROUPS[group])["CUDA_VISIBLE_DEVICES"] == expected_mask


def fixture_campaign(pool, manifest, tmp_path):
    campaign = json.loads((ROOT / "configs/math_campaign.json").read_text())
    campaign.update(pool_dir=str(pool), pool_manifest_sha256=file_sha256(pool / "manifest.json"),
                    pool_counts={name: split["records"] for name, split in manifest["splits"].items()},
                    output_root=str(tmp_path / "math-output"))
    return campaign


def test_campaign_rejects_missing_changed_or_replaced_pool(frozen_fixture_pool, tmp_path):
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    assert launcher.check_pool(campaign)[0] == pool
    with pytest.raises(FileNotFoundError):
        launcher.check_pool(campaign, tmp_path / "missing")
    changed = copy.deepcopy(campaign)
    changed["pool_counts"]["train"] += 1
    with pytest.raises(ValueError, match="count differs"):
        launcher.check_pool(changed)
    with (pool / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="artifact mismatch"):
        launcher.build_math_groups(arguments(), campaign)
    (pool / "manifest.json").write_text(json.dumps({**manifest, "changed": True}))
    with pytest.raises(ValueError, match="manifest changed"):
        launcher.check_pool(campaign)


@pytest.mark.parametrize("missing,field", [
    (True, "grading"),
    (True, "source_sha256"), (False, "source_sha256"),
    (True, "gold_audit_sha256"), (False, "gold_audit_sha256"),
])
def test_evaluation_plan_rejects_missing_or_changed_grading_pins(
        frozen_fixture_pool, tmp_path, missing, field):
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    if field == "grading":
        campaign.pop(field)
    elif missing:
        campaign["grading"].pop(field)
    else:
        campaign["grading"][field] = "0" * 64
    with pytest.raises(ValueError, match="(?i)(grader|gold-reference audit)"):
        launcher.build_math_groups(arguments(phase="evaluate"), campaign)
    assert not Path(campaign["output_root"]).exists()


@pytest.mark.parametrize("artifact", ["grader", "audit"])
@pytest.mark.parametrize("change", ["changed", "missing"])
def test_grading_check_rejects_changed_or_missing_actual_files(tmp_path, monkeypatch, artifact, change):
    launcher = importlib.import_module("scripts.run_math")
    campaign = launcher.load_campaign(ROOT / "configs/math_campaign.json")
    fixture_root = tmp_path / "code-copy"
    source = fixture_root / "baseline_common/math_metrics.py"
    source.parent.mkdir(parents=True)
    source.write_bytes((ROOT / "baseline_common/math_metrics.py").read_bytes())
    audit = fixture_root / "gold-audit.json"
    audit.write_bytes((ROOT / campaign["grading"]["gold_audit"]).read_bytes())
    campaign["grading"]["gold_audit"] = str(audit)
    monkeypatch.setattr(launcher, "ROOT", fixture_root)
    launcher.check_grader(campaign)
    target = source if artifact == "grader" else audit
    if change == "changed":
        target.write_bytes(target.read_bytes() + b"\n")
        expected = ValueError
    else:
        target.unlink()
        expected = FileNotFoundError
    with pytest.raises(expected):
        launcher.check_grader(campaign)


def test_runner_rechecks_grading_before_next_evaluation_child(frozen_fixture_pool, tmp_path, monkeypatch):
    from baseline_common.orchestration import Runner
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    audit = tmp_path / "gold-audit.json"
    audit.write_bytes((ROOT / campaign["grading"]["gold_audit"]).read_bytes())
    campaign["grading"]["gold_audit"] = str(audit)
    groups, state = launcher.build_math_groups(arguments(phase="evaluate", group="qwen"), campaign)
    entered = []

    def record_child(self, stage, resume):
        entered.append(stage.key)
        return 0, None

    # Isolate the prelaunch boundary; the integration below exercises all real
    # children, computation and artifacts with the pinned grader unchanged.
    monkeypatch.setattr(Runner, "_run_child", record_child)
    runner = launcher.MathRunner(groups, {"qwen": GPU_GROUPS["qwen"]}, state,
                                 campaign=campaign, lock_dir=tmp_path / "locks")
    first, second = groups["qwen"][:2]
    assert runner._run_child(first, None) == (0, None)
    audit.write_bytes(audit.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="gold-reference audit changed"):
        runner._run_child(second, None)
    assert entered == [first.key]


def test_evaluation_commands_use_math_accuracy_and_both_full_test_sets(frozen_fixture_pool, tmp_path):
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    groups, _ = launcher.build_math_groups(arguments(phase="evaluate"), campaign)
    for stages in groups.values():
        for stage in stages:
            command = launcher.math_command(stage, None, campaign)
            assert Path(command[1]).name == "evaluate_math.py"
            assert command[command.index("--benchmark") + 1] in {"gsm8k", "math"}
            assert command[command.index("--data") + 1] == str(stage.input_path)
            assert "--resume" in command and "--limit" not in command
            assert not any("rouge" in argument.lower() for argument in command)


def test_evaluation_before_training_reports_incomplete_run_even_without_pairs(frozen_fixture_pool, tmp_path):
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    assert not Path(campaign["output_root"]).exists()
    with pytest.raises(ValueError, match="(?i)training.*complete"):
        launcher.build_math_groups(arguments(phase="evaluate", dry_run=False,
                                              methods=["distillm2"]), campaign)
    assert not Path(campaign["output_root"]).exists()


@pytest.mark.parametrize("field,replacement", [
    ("student_model", "/tmp/a-previous-method-final"),
    ("teacher_model", "/tmp/different-teacher"),
    ("learning_rate", 0.01),
    ("gradient_accumulation_steps", 16),
    ("max_steps", 9),
])
def test_campaign_rejects_mixed_initializations_or_training_budgets(
        frozen_fixture_pool, tmp_path, field, replacement):
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    changed = json.loads((ROOT / campaign["config_files"]["qwen"]["abkd"]).read_text())
    changed[field] = replacement
    path = tmp_path / "changed-source-config.json"
    path.write_text(json.dumps(changed))
    campaign["config_files"]["qwen"]["abkd"] = str(path)
    with pytest.raises(ValueError):
        launcher.build_math_groups(arguments(), campaign)


def test_stage_data_paths_always_bind_verified_pool(frozen_fixture_pool, tmp_path):
    launcher = importlib.import_module("scripts.run_math")
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    changed = json.loads((ROOT / campaign["config_files"]["qwen"]["kd"]).read_text())
    changed.update(train_file="/tmp/unverified-other-train.jsonl",
                   validation_file="/tmp/unverified-other-dev.jsonl")
    path = tmp_path / "source-with-stale-data-paths.json"
    path.write_text(json.dumps(changed))
    campaign["config_files"]["qwen"]["kd"] = str(path)
    groups, _ = launcher.build_math_groups(arguments(), campaign)
    for stages in groups.values():
        for stage in stages:
            assert Path(stage.config["validation_file"]) == pool / "validation.jsonl"
            if stage.kind == "pairs":
                assert stage.input_path == pool / "train.jsonl"
            elif stage.method != "distillm2":
                assert Path(stage.config["train_file"]) == pool / "train.jsonl"


def test_real_cpu_campaign_interruption_resume_eight_models_and_sixteen_tests(
        frozen_fixture_pool, tmp_path, monkeypatch):
    import torch
    from baseline_common.config import load_config, validate_config
    from baseline_common.math_metrics import metric_definition
    from test_parallel_real_pipeline import _checkpoints
    # This integration is deliberately run with .venv-math-eval so genuine
    # symbolic grading is required, not skipped or replaced by a fake scorer.
    assert metric_definition("gsm8k")["name"] == "math_accuracy"
    launcher = importlib.import_module("scripts.run_math")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(2)
    pool, manifest = frozen_fixture_pool
    campaign = fixture_campaign(pool, manifest, tmp_path)
    campaign["evaluation"].update(max_prompt_tokens=64, max_new_tokens=3)
    for group, architecture in (("qwen", "Qwen3"), ("llama", "Llama")):
        models = tmp_path / "models" / group
        _checkpoints(models, architecture)
        for method, relative in campaign["config_files"][group].items():
            cfg = load_config(ROOT / relative)
            cfg.update(teacher_model=str(models / "teacher"), student_model=str(models / "student"),
                       vocabulary_policy="full", dtype="float32", device="cpu", teacher_device="cpu",
                       max_steps=None, max_prompt_tokens=16, max_new_tokens=3,
                       gradient_accumulation_steps=1, save_steps=1, eval_steps=1,
                       eval_max_new_tokens=3, optimizer_offload=True,
                       learning_rate=0.003, loss_chunk_size=2, acceptance_k=2, proposal_block_size=2,
                       generation=dict(temperature=0.7, top_p=1.0, top_k=2),
                       teacher_generation=dict(temperature=0.7, top_p=1.0, top_k=2))
            destination = tmp_path / f"{group}_{method}.json"
            destination.write_text(json.dumps(validate_config(cfg)))
            campaign["config_files"][group][method] = str(destination)
    original_dolly = snapshots(sorted((ROOT / "configs/experiments").glob("*.json")))
    frozen_before = snapshots(path for path in pool.rglob("*") if path.is_file())
    groups, state = launcher.build_math_groups(arguments(dry_run=False, max_steps=2), campaign)
    worker = tmp_path / "native_worker.py"
    worker.write_text(NATIVE_WORKER)
    events = tmp_path / "workers.jsonl"
    launches = []

    def command(stage, resume):
        native = launcher.math_command(stage, resume, campaign, interpreter=sys.executable)
        launches.append((stage.key, resume))
        return [native[0], str(worker), str(events), stage.group, stage.kind, stage.method, *native[1:]]

    def run(selected, status):
        return launcher.MathRunner(selected, {group: GPU_GROUPS[group] for group in selected}, status,
                                   campaign=campaign, command_builder=command,
                                   lock_dir=tmp_path / "locks").run()

    # One actual update is committed before SIGTERM. Llama's group continues.
    (tmp_path / "interrupt_qwen_train_abkd").touch()
    assert run(groups, state) == 1
    qwen_abkd = next(stage for stage in groups["qwen"] if stage.method == "abkd")
    interrupted = json.loads((qwen_abkd.output_path / "result.json").read_text())
    assert interrupted["step"] == 1 and interrupted["complete"] is False
    checkpoint = qwen_abkd.output_path / "checkpoints/step_000001"
    assert (checkpoint / "complete.json").is_file()
    assert all(is_training_complete(stage.output_path, 2)
               for stage in groups["llama"] if stage.kind == "train")
    committed_prefix = (qwen_abkd.output_path / "metrics.jsonl").read_bytes()
    launches.clear()
    assert run(groups, state) == 0
    assert (qwen_abkd.key, checkpoint) in launches
    assert (qwen_abkd.output_path / "metrics.jsonl").read_bytes().startswith(committed_prefix)
    training_snapshots = {}
    for group, stages in groups.items():
        source_ids = {row["id"] for row in load_records(pool / "train.jsonl")}
        for stage in stages:
            if stage.kind == "pairs":
                assert {row["id"] for row in load_records(stage.output_path, paired=True)} == source_ids
                continue
            assert is_training_complete(stage.output_path, 2)
            training_metrics = read_jsonl(stage.output_path / "metrics.jsonl")
            assert len(training_metrics) == 2
            assert all(math.isfinite(row["loss"]) for row in training_metrics)
            assert not (stage.output_path / "validation.jsonl").exists()
            from safetensors.torch import load_file
            initial = load_file(Path(stage.config["student_model"]) / "model.safetensors")
            exported = load_file(stage.output_path / "final/model.safetensors")
            assert all(torch.isfinite(value).all() for value in exported.values())
            assert any(not torch.equal(initial[key], value) for key, value in exported.items())
            training_snapshots.update(snapshots([
                stage.output_path / "result.json", stage.output_path / "final/model.safetensors"]))
    launches.clear()
    assert run(groups, state) == 0
    assert len(launches) == 2 and all(key.endswith(":pairs") for key, _ in launches)
    assert snapshots(training_snapshots) == training_snapshots

    # Evaluation reads the actual pilot budget from each completed training
    # manifest, although the source configurations still request a full epoch.
    assert all(json.loads(Path(path).read_text())["max_steps"] is None
               for family in campaign["config_files"].values() for path in family.values())
    evaluations, eval_state = launcher.build_math_groups(arguments(phase="evaluate", dry_run=False), campaign)
    assert all(stage.config["max_steps"] == 2 for stages in evaluations.values() for stage in stages)
    selected_training = groups["qwen"][0]
    recorded_path = selected_training.output_path / "manifest.json"
    original_manifest = recorded_path.read_text()
    # A finished checkpoint from another experiment must never be relabelled
    # as this campaign merely because its training file happens to match.
    for field, replacement in (("pair", PAIR_NAMES["llama"]), ("method", "skd"),
                               ("dataset", "dolly"), ("seed", 7),
                               ("student_model", "/tmp/other-student"),
                               ("teacher_model", "/tmp/other-teacher"),
                               ("learning_rate", 0.9)):
        changed = json.loads(original_manifest)
        changed["config"][field] = replacement
        recorded_path.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError):
                launcher.build_math_groups(arguments(phase="evaluate", dry_run=False), campaign)
        finally:
            recorded_path.write_text(original_manifest)
    (tmp_path / "interrupt_qwen_evaluate_kd_gsm8k").touch()
    launches.clear()
    assert run(evaluations, eval_state) == 1
    first = evaluations["qwen"][0]
    assert len(read_jsonl(first.output_path / "predictions.jsonl")) == 1
    assert not (first.output_path / "manifest.json").exists()
    predictions_prefix = (first.output_path / "predictions.jsonl").read_bytes()
    assert run(evaluations, eval_state) == 0
    assert (first.output_path / "predictions.jsonl").read_bytes().startswith(predictions_prefix)
    evaluation_snapshots = {}
    for stages in evaluations.values():
        for stage in stages:
            metrics = json.loads((stage.output_path / "metrics.json").read_text())
            predictions = read_jsonl(stage.output_path / "predictions.jsonl")
            benchmark = stage.method.rsplit("_", 1)[1]
            assert metrics["benchmark"] == benchmark
            assert len(predictions) == metrics["total"] == 2
            assert metrics["metric_definition"]["name"] == "math_accuracy"
            assert metrics["accuracy_percent"] == sum(row["correct"] for row in predictions) / 2 * 100
            assert "rouge_l" not in metrics
            evaluation_snapshots.update(snapshots([
                stage.output_path / name for name in ("predictions.jsonl", "metrics.json", "manifest.json")]))
    device_rows = read_jsonl(tmp_path / "devices.jsonl")
    assert device_rows and all(row["device"] == "cpu" for row in device_rows)
    devices_before = len(device_rows)
    assert run(evaluations, eval_state) == 0
    assert len(read_jsonl(tmp_path / "devices.jsonl")) == devices_before
    assert snapshots(evaluation_snapshots) == evaluation_snapshots
    summary = tmp_path / "summary"
    rows = launcher.write_math_summary(evaluations, summary)
    assert len(rows) == 16
    assert {row["benchmark"] for row in rows} == {"gsm8k", "math"}
    assert json.loads(summary.with_suffix(".json").read_text())["benchmarks_separate"] is True
    assert snapshots(frozen_before) == frozen_before
    assert snapshots(original_dolly) == original_dolly
    events_seen = read_jsonl(events)
    assert all(row["cuda"] == ",".join(GPU_GROUPS[row["group"]]) for row in events_seen)
    for group in groups:
        first_training = []
        for row in events_seen:
            if row["group"] == group and row["kind"] == "train" and row["method"] not in first_training:
                first_training.append(row["method"])
        assert first_training == list(METHOD_ORDER)


# Wrapping a native CLI records GPU masks and actual device placement while
# preserving its real training, generation, checkpoint and evaluation behavior.
NATIVE_WORKER = r'''
import json
import os
from pathlib import Path
import runpy
import signal
import sys

events, group, kind, method = sys.argv[1:5]
with Path(events).open("a") as stream:
    stream.write(json.dumps(dict(group=group, kind=kind, method=method,
                                cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\n")
# Record the real scheduling mask, then hide devices before importing torch.
# Even CPU training's RNG checkpoint code otherwise probes available GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = sys.argv[5:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
import baseline_common.models as models
original = models.load_model
def observe(path, dtype="bfloat16", device="cuda:0", *args, **kwargs):
    if str(device) != "cpu":
        raise RuntimeError("Math integration smoke must never load a GPU model")
    model = original(path, dtype, device, *args, **kwargs)
    with Path(events).with_name("devices.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(group=group, kind=kind, method=method,
                                    device=str(next(model.parameters()).device))) + "\n")
    return model
models.load_model = observe
interruption = Path(events).with_name(f"interrupt_{group}_{kind}_{method}")
if interruption.exists():
    interruption.unlink()
    if kind == "train":
        import importlib
        training = importlib.import_module("baseline_common.train")
        original_loss = training.response_loss
        def interrupt_loss(*args, **kwargs):
            result = original_loss(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)
            return result
        training.response_loss = interrupt_loss
    elif kind != "pairs":
        import baseline_common.pair_progress as progress
        original_append = progress.append_jsonl
        def interrupt_append(*args, **kwargs):
            original_append(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)
        progress.append_jsonl = interrupt_append
runpy.run_path(sys.argv[0], run_name="__main__")
'''
