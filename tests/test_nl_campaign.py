"""Local five-benchmark routing, saved-run guards, and genuine CPU evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, write_jsonl
from baseline_common.nl_data import SOURCE_FILES, load_nl_records, prepare_nl_pool
from baseline_common.orchestration import METHOD_ORDER, PAIR_NAMES, Runner
from scripts import run_nl_evaluation as launcher
from test_parallel_real_pipeline import _checkpoints


ROOT = Path(__file__).resolve().parents[1]


def _sources(root):
    for benchmark, files in SOURCE_FILES.items():
        for subset, relative in enumerate(files):
            write_jsonl(root / relative, [
                {"instruction": f"heldout {benchmark} {subset} three", "input": "four",
                 "output": ["answer", "short answer"], "prompt": "OLD TEMPLATE MUST NOT BE USED"},
                {"instruction": f"heldout {benchmark} {subset} answer", "input": "",
                 "output": "long answer"},
            ])


def _args(root, **changes):
    values = dict(output_root=str(root / "outputs"), evaluation_root=None,
                  data_root=str(root / "data"), dataset="fixture", source_root=str(root / "sources"),
                  prepared_data_dir=str(root / "pool"), exclude_training_data=[],
                  benchmarks=list(launcher.BENCHMARKS), group="all", methods=list(METHOD_ORDER),
                  seeds=[42], qwen_gpus=("0", "1"), llama_gpus=("2", "3"), device="cpu",
                  dtype=None, max_prompt_tokens=32, max_new_tokens=3,
                  rouge_tokenizer="english", limit=None, prepare_only=False, dry_run=False)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def local_pool(tmp_path):
    _sources(tmp_path / "sources")
    write_jsonl(tmp_path / "data/fixture/train.jsonl", [
        {"id": "a", "prompt": "one", "response": "short"},
        {"id": "b", "prompt": "two", "response": "long answer"},
    ])
    write_jsonl(tmp_path / "data/fixture/validation.jsonl", [
        {"id": "v", "prompt": "validation only", "response": "answer"},
    ])
    pool, _ = launcher.prepare_data(_args(tmp_path))
    return tmp_path, pool


@pytest.mark.parametrize("name, expected", [
    ("DollyEval", "dolly"), ("SelfInst", "selfinst"), ("Super-Natural", "super_natural"),
    ("Unnatural", "unnatural"), ("VicunaEval", "vicuna"), ("SUPER_NATURAL", "super_natural"),
])
def test_cli_accepts_downloaded_directory_names(name, expected):
    assert launcher.benchmark_name(name) == expected
    with pytest.raises(argparse.ArgumentTypeError):
        launcher.benchmark_name("unknown")


def test_five_local_schemas_produce_forty_separate_saved_student_stages(local_pool):
    root, pool = local_pool
    args = _args(root, dry_run=True)
    groups, state = launcher.build_evaluation_groups(args, pool)
    all_stages = [stage for stages in groups.values() for stage in stages]
    assert len(all_stages) == 40
    assert len({stage.key for stage in all_stages}) == 40
    assert len({stage.output_path for stage in all_stages}) == 40
    assert len({stage.config_path for stage in all_stages}) == 40
    assert state.is_relative_to(root / "outputs/evaluation_orchestration/nl/fixture")
    for group, stages in groups.items():
        assert [stage.method for stage in stages] == [
            f"{method}_{benchmark}" for method in METHOD_ORDER for benchmark in launcher.BENCHMARKS]
        for method in METHOD_ORDER:
            selected = [stage for stage in stages if stage.config["method"] == method]
            expected_model = root / "outputs/runs" / PAIR_NAMES[group] / "fixture" / method / "seed_42/final"
            assert {stage.model_path for stage in selected} == {expected_model}
            assert {stage.input_path.stem for stage in selected} == set(launcher.BENCHMARKS)
            assert all(stage.kind == "evaluate" for stage in selected)
    for benchmark in launcher.BENCHMARKS:
        rows = load_nl_records(pool / f"{benchmark}.jsonl")
        assert len(rows) == len(SOURCE_FILES[benchmark]) * 2
        assert rows[0]["references"] == ["answer", "short answer"]
        assert rows[0]["prompt"].endswith("\n\nfour")
        assert all("OLD TEMPLATE" not in row["prompt"] for row in rows)
        command = launcher.evaluation_command(next(stage for stage in all_stages
                                                   if stage.input_path.stem == benchmark), args, pool)
        assert Path(command[1]).name == "evaluate_nl.py"
        assert command[command.index("--benchmark") + 1] == benchmark
        assert command[command.index("--data-manifest") + 1] == str(pool / "manifest.json")
        assert "--resume" in command and "--limit" not in command


def test_selection_and_protocol_changes_have_distinct_summary_paths(local_pool):
    root, pool = local_pool
    choices = [dict(), dict(group="qwen"), dict(group="llama"), dict(methods=["kd"]),
               dict(seeds=[43]), dict(benchmarks=["dolly"]), dict(limit=1),
               dict(max_new_tokens=4), dict(max_prompt_tokens=33), dict(dtype="bfloat16"),
               dict(rouge_tokenizer="unicode"), dict(device="cuda:0")]
    destinations = [launcher.summary_destination(_args(root, **changes), pool) for changes in choices]
    assert len(set(destinations)) == len(choices)
    assert destinations[0].name == "summary"
    args = _args(root, group="llama", methods=["skd", "distillm2"], seeds=[42, 43],
                 benchmarks=["dolly", "vicuna"], limit=1, dry_run=True)
    groups, _ = launcher.build_evaluation_groups(args, pool)
    assert list(groups) == ["llama"]
    assert len(groups["llama"]) == 8
    assert all(stage.output_path.name.endswith("_first_1") for stage in groups["llama"])
    assert all("--limit" in launcher.evaluation_command(stage, args, pool) for stage in groups["llama"])


def test_missing_saved_model_refused_before_any_inference(local_pool):
    root, pool = local_pool
    with pytest.raises(FileNotFoundError, match="No training result"):
        launcher.build_evaluation_groups(_args(root), pool)


def test_concurrent_group_preparation_waits_for_first_pool_commit(local_pool, monkeypatch):
    from baseline_common import nl_data

    root, _ = local_pool
    pool = root / "concurrent_pool"
    args = _args(root, prepared_data_dir=str(pool))
    publishing, release, second_started = threading.Event(), threading.Event(), threading.Event()
    original_write = nl_data.write_jsonl

    def delayed_write(path, rows):
        if Path(path).parent == pool and not publishing.is_set():
            publishing.set()
            assert release.wait(5), "The test must release the first publisher"
        return original_write(path, rows)

    def second_group():
        second_started.set()
        return launcher.prepare_data(args)

    monkeypatch.setattr(nl_data, "write_jsonl", delayed_write)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(launcher.prepare_data, args)
        try:
            assert publishing.wait(5)
            assert pool.is_dir() and not (pool / "manifest.json").exists()
            second = workers.submit(second_group)
            assert second_started.wait(5)
            # The second launch sees a directory mid-publication. It must wait,
            # rather than reject it as incomplete or race its first writer.
            with pytest.raises(FutureTimeout):
                second.result(timeout=0.1)
        finally:
            release.set()
        first_result = first.result(timeout=5)
        second_result = second.result(timeout=5)
    assert first_result == second_result
    assert nl_data.validate_nl_pool(pool) == first_result[1]
    before = {path.name: (file_sha256(path), path.stat().st_mtime_ns) for path in pool.iterdir()}
    assert launcher.prepare_data(args) == first_result
    assert before == {path.name: (file_sha256(path), path.stat().st_mtime_ns) for path in pool.iterdir()}


def test_dry_run_and_prepare_only_cli_need_no_models_and_report_group_masks(local_pool, monkeypatch, capsys):
    root, pool = local_pool
    command = ["run_nl_evaluation.py", "--output-root", str(root / "outputs"),
               "--data-root", str(root / "data"), "--dataset", "fixture",
               "--source-root", str(root / "sources"), "--prepared-data-dir", str(pool)]
    monkeypatch.setattr(sys, "argv", command + ["--prepare-only"])
    assert launcher.main() == 0
    prepared = capsys.readouterr().out
    assert all(name in prepared for name in launcher.NAMES.values())
    monkeypatch.setattr(sys, "argv", command + ["--dry-run"])
    assert launcher.main() == 0
    planned = capsys.readouterr().out
    assert "qwen: cuda:0 within GPUs 0,1; 20 evaluations" in planned
    assert "llama: cuda:0 within GPUs 2,3; 20 evaluations" in planned
    assert not (root / "outputs").exists()


@pytest.mark.parametrize("options", [
    ["--qwen-gpus", "0,1", "--llama-gpus", "1,2"],
    ["--benchmarks", "DollyEval", "dolly"], ["--methods", "kd", "kd"],
    ["--seeds", "42", "42"], ["--limit", "0"], ["--max-new-tokens", "0"],
])
def test_cli_rejects_ambiguous_selection_before_source_or_model_access(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["run_nl_evaluation.py", *options])
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 2


def test_shell_wrapper_routes_new_benchmarks_and_explicit_legacy_validation():
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    current = subprocess.run(["bash", str(ROOT / "scripts/evaluate_all.sh"), "--help"],
                             env=environment, text=True, capture_output=True, check=True)
    legacy = subprocess.run(["bash", str(ROOT / "scripts/evaluate_all.sh"), "--validation-only", "--help"],
                            env=environment, text=True, capture_output=True, check=True)
    assert "--source-root" in current.stdout and "--benchmarks" in current.stdout
    assert "--checkpoint" in legacy.stdout and "--source-root" not in legacy.stdout


@pytest.fixture(scope="module")
def tiny_saved_students(tmp_path_factory):
    from baseline_common.train import train

    root = tmp_path_factory.mktemp("nl_saved_students")
    _sources(root / "sources")
    data = root / "data/fixture"
    write_jsonl(data / "train.jsonl", [
        {"id": "a", "prompt": "one", "response": "short"},
        {"id": "b", "prompt": "two", "response": "long answer"},
    ])
    write_jsonl(data / "validation.jsonl", [
        {"id": "v", "prompt": "validation only", "response": "answer"},
    ])
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    runs = {}
    try:
        for group, architecture in (("qwen", "Qwen3"), ("llama", "Llama")):
            models = root / "models" / group
            _checkpoints(models, architecture)
            cfg = validate_config(dict(
                name=f"{group}_tiny", pair=PAIR_NAMES[group], method="kd", dataset="fixture",
                teacher_model=str(models / "teacher"), student_model=str(models / "student"),
                train_file=str(data / "train.jsonl"), validation_file=str(data / "validation.jsonl"),
                output_root=str(root / "outputs"), dtype="float32", device="cpu", teacher_device="cpu",
                optimizer_offload=True, gradient_checkpointing=True, max_steps=1,
                max_prompt_tokens=16, max_new_tokens=2, eval_max_new_tokens=2,
                gradient_accumulation_steps=1, save_steps=1, eval_steps=1,
                learning_rate=0.003, loss_chunk_size=2,
            ))
            runs[group] = train(cfg)
            assert json.loads((runs[group] / "result.json").read_text())["complete"] is True
            # Inference must load the trained export without requiring the original weights.
            (models / "teacher").rename(models / "teacher_unavailable")
            (models / "student").rename(models / "student_unavailable")
    finally:
        torch.set_num_threads(prior_threads)
    pool, _ = launcher.prepare_data(_args(root))
    return root, pool, runs


def test_evaluation_overrides_preserve_exact_saved_training_configuration(tiny_saved_students):
    root, pool, runs = tiny_saved_students
    args = _args(root, methods=["kd"], dtype="bfloat16", device="cuda:0",
                 max_prompt_tokens=40, max_new_tokens=6, rouge_tokenizer="unicode",
                 evaluation_root=str(root / "elsewhere"))
    groups, _ = launcher.build_evaluation_groups(args, pool)
    for group, stages in groups.items():
        saved = json.loads((runs[group] / "manifest.json").read_text())["config"]
        for stage in stages:
            assert stage.config == saved
            assert stage.output_path.is_relative_to(root / "elsewhere")
            assert stage.model_path == runs[group] / "final"
            command = launcher.evaluation_command(stage, args, pool)
            for option, expected in (("--dtype", "bfloat16"), ("--device", "cuda:0"),
                                     ("--max-prompt-tokens", "40"), ("--max-new-tokens", "6"),
                                     ("--rouge-tokenizer", "unicode")):
                assert command[command.index(option) + 1] == expected


@pytest.mark.parametrize("mutation", ["incomplete_training", "incomplete_export", "training_hash", "manifest_identity"])
def test_saved_run_validation_rejects_changed_or_incomplete_inputs(tiny_saved_students, mutation):
    root, pool, runs = tiny_saved_students
    run = runs["qwen"]
    target = {"incomplete_training": run / "result.json", "incomplete_export": run / "final/config.json",
              "training_hash": root / "data/fixture/train.jsonl", "manifest_identity": run / "manifest.json"}[mutation]
    original = target.read_bytes()
    try:
        if mutation == "incomplete_export":
            target.unlink()
        elif mutation == "training_hash":
            target.write_bytes(original + b"\n")
        else:
            value = json.loads(original)
            if mutation == "incomplete_training":
                value["complete"] = False
            else:
                value["config"]["seed"] = 43
            target.write_text(json.dumps(value))
        with pytest.raises(ValueError, match="incomplete|changed|identity"):
            launcher.build_evaluation_groups(_args(root, methods=["kd"], group="qwen"), pool)
    finally:
        target.write_bytes(original)


def test_train_overlap_after_pilot_limit_is_rejected(tiny_saved_students, tmp_path):
    root, _, _ = tiny_saved_students
    sources = tmp_path / "sources"
    _sources(sources)
    dolly = sources / "DollyEval/valid.jsonl"
    with dolly.open("a") as stream:
        stream.write(json.dumps({"instruction": "one", "output": "short"}) + "\n")
    pool = tmp_path / "unexcluded_pool"
    prepare_nl_pool(sources, pool, benchmarks=["dolly"], exclude_files=[])
    assert load_nl_records(pool / "dolly.jsonl")[-1]["prompt"] == "one"
    with pytest.raises(ValueError, match="overlaps.*exclude-training-data"):
        launcher.build_evaluation_groups(_args(root, methods=["kd"], group="qwen",
                                               benchmarks=["dolly"], limit=1), pool)


CPU_WORKER = '''\
import json
import os
from pathlib import Path
import runpy
import sys

events, group, method = sys.argv[1:4]
with Path(events).open("a") as stream:
    stream.write(json.dumps(dict(group=group, method=method, cuda=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\\n")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = sys.argv[4:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parents[1]))
import torch
assert not torch.cuda.is_available()
torch.set_num_threads(2)
import baseline_common.models as models
original = models.load_model
def observe_load(path, *args, **kwargs):
    assert Path(path).name == "final"
    model = original(path, *args, **kwargs)
    with Path(events).with_name("loaded_models.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(group=group, method=method, path=str(path),
                                    device=str(next(model.parameters()).device))) + "\\n")
    return model
models.load_model = observe_load
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def test_native_two_family_two_benchmark_cpu_queue_and_reuse(tiny_saved_students, tmp_path, monkeypatch):
    root, pool, runs = tiny_saved_students
    args = _args(root, methods=["kd"], benchmarks=["dolly", "selfinst"],
                 evaluation_root=str(tmp_path / "results"))
    groups, state = launcher.build_evaluation_groups(args, pool)
    saved_manifests = {run / "manifest.json": (run / "manifest.json").read_bytes() for run in runs.values()}
    worker = tmp_path / "cpu_worker.py"
    worker.write_text(CPU_WORKER)
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")

    def command(stage, resume):
        original = launcher.evaluation_command(stage, args, pool)
        return [original[0], str(worker), str(events), stage.group, stage.method, *original[1:]]

    def runner():
        return Runner(groups, {"qwen": args.qwen_gpus, "llama": args.llama_gpus}, state,
                      command_builder=command, lock_dir=tmp_path / "locks")

    assert runner().run() == 0
    observed = [json.loads(line) for line in events.read_text().splitlines()]
    loaded_path = events.with_name("loaded_models.jsonl")
    loaded = [json.loads(line) for line in loaded_path.read_text().splitlines()]
    assert len(observed) == len(loaded) == 4
    assert all(row["device"] == "cpu" for row in loaded)
    snapshots = {}
    for group, stages in groups.items():
        actual = [row for row in observed if row["group"] == group]
        assert [row["method"] for row in actual] == ["kd_dolly", "kd_selfinst"]
        assert all(row["cuda"] == ("0,1" if group == "qwen" else "2,3") for row in actual)
        for stage in stages:
            metrics = json.loads((stage.output_path / "metrics.json").read_text())
            assert metrics["benchmark"] == stage.input_path.stem
            assert metrics["examples"] == 2 and metrics["multiple_reference_examples"] == 1
            assert 0 <= metrics["rouge_l"] <= 100 and "reference_nll" not in metrics
            assert metrics["evaluation"]["max_prompt_tokens"] == 32
            assert metrics["evaluation"]["max_new_tokens"] == 3
            predictions = [json.loads(line) for line in (stage.output_path / "predictions.jsonl").read_text().splitlines()]
            assert metrics["rouge_l"] == pytest.approx(sum(row["rouge_l"] for row in predictions) / 2)
            assert all(row["rouge_l"] == max(row["reference_rouge_l"]) for row in predictions)
            for name in ("manifest.json", "predictions.jsonl", "metrics.json"):
                path = stage.output_path / name
                snapshots[path] = (file_sha256(path), path.stat().st_mtime_ns)
    destination = launcher.summary_destination(args, pool)
    rows = launcher.write_summary(groups, destination)
    assert len(rows) == 4
    summary = json.loads(destination.with_suffix(".json").read_text())
    assert summary["metric"] == "rouge_l" and summary["range"] == [0, 100]
    assert summary["benchmarks_separate"] is True
    with destination.with_suffix(".csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 4
    for row in csv_rows:
        assert float(row["rouge_l"]) == json.loads(Path(row["metrics"]).read_text())["rouge_l"]
    before_loads = loaded_path.read_bytes()
    assert runner().run() == 0
    assert loaded_path.read_bytes() == before_loads
    assert len(events.read_text().splitlines()) == 8
    assert all((file_sha256(path), path.stat().st_mtime_ns) == snapshot for path, snapshot in snapshots.items())
    assert all(path.read_bytes() == original for path, original in saved_manifests.items())
