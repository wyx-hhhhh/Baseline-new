"""Exercise parallel experiment scheduling with real, lightweight child processes.

Workers never import torch or access a GPU. They use the same on-disk commit
markers as training so scheduling, recovery, environment and locks are tested
independently of expensive model loading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest


FAKE_WORKER = r'''
import json
import os
from pathlib import Path
import pickle
import struct
import sys
import time
import zipfile

job = json.loads(sys.argv[1])
root = Path(job["events"])
root.mkdir(parents=True, exist_ok=True)
stage = job["stage"]
group = job["group"]

def event(kind, **extra):
    payload = dict(event=kind, stage=stage, group=group, pid=os.getpid(),
                   time=time.monotonic(), resume=job.get("resume"),
                   cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   ranks={k: os.environ[k] for k in (
                       "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                       "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                       "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT")
                       if k in os.environ}, **extra)
    with open(root / "events.jsonl", "a") as handle:
        handle.write(json.dumps(payload) + "\n")

def export(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "llama", "vocab_size": 8}))
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text(json.dumps({"model": {
        "type": "WordLevel", "vocab": {"<unk>": 0}, "unk_token": "<unk>"}}))
    # A complete one-element safetensors file, built using its documented
    # header/offset format without importing model or GPU libraries.
    header = json.dumps({"test.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * ((-len(header)) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))

def state_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({}))
        archive.writestr("archive/version", b"3\n")

def checkpoint(step):
    path = Path(job["output"]) / "checkpoints" / f"step_{step:06d}"
    export(path / "student")
    state_archive(path / "training.pt")
    state_archive(path / "rng_rank0.pt")
    (path / "complete.json").write_text(json.dumps({"step": step, "world_size": 1}))
    return path

event("start")
if job.get("barrier"):
    (root / f"barrier_{group}").write_text(str(os.getpid()))
    deadline = time.monotonic() + 15
    while len(list(root.glob("barrier_*"))) < 2:
        if time.monotonic() > deadline:
            event("barrier_timeout")
            sys.exit(91)
        time.sleep(0.01)

output = Path(job["output"])
if job.get("kind") == "pairs":
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = output.with_suffix(".progress.jsonl")
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    resumed_rows = len(rows)
    failure = root / (stage + ".fail_once")
    for record_id in ("a", "b"):
        if any(row["id"] == record_id for row in rows):
            continue
        row = {"id": record_id, "prompt": "prompt", "chosen": "answer", "rejected": "answer"}
        rows.append(row)
        with journal.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if failure.exists():
            failure.unlink()
            event("failed", committed_rows=len(rows))
            sys.exit(17)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    event("end", resumed_rows=resumed_rows)
    sys.exit(0)

output.mkdir(parents=True, exist_ok=True)
(output / "manifest.json").write_text(json.dumps(job["manifest"]))
failure = root / (stage + ".fail_once")
if failure.exists():
    failure.unlink()
    committed = checkpoint(1)
    # An interrupted save must never beat the last fully committed checkpoint.
    (output / "checkpoints/step_000099").mkdir(parents=True, exist_ok=True)
    (output / "checkpoints/step_000099/training.pt").write_bytes(b"torn")
    (output / "result.json").write_text(json.dumps({"complete": False, "step": 1,
                                                    "last_checkpoint": str(committed)}))
    event("failed")
    sys.exit(17)

if job.get("hold"):
    checkpoint(1)
    (root / (stage + ".holding")).write_text(str(os.getpid()))
    while True:
        time.sleep(0.1)

if job.get("incomplete_success"):
    committed = checkpoint(1)
    (output / "result.json").write_text(json.dumps({"complete": False, "step": 1,
                                                    "last_checkpoint": str(committed)}))
    event("end_incomplete")
    sys.exit(0)

time.sleep(job.get("delay", 0.02))
last = checkpoint(2)
final = output / "final"
export(final)
(output / "result.json").write_text(json.dumps({"complete": True, "step": 2,
                                                "last_checkpoint": str(last)}))
event("end")
'''


@pytest.fixture
def fake_worker(tmp_path):
    path = tmp_path / "fake_worker.py"
    path.write_text(textwrap.dedent(FAKE_WORKER))
    return path


def read_events(path):
    source = Path(path) / "events.jsonl"
    return [json.loads(line) for line in source.read_text().splitlines()] if source.exists() else []


def wait_for_file(path, timeout=15):
    deadline = time.monotonic() + timeout
    while not Path(path).exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Worker did not create {path}")
        time.sleep(0.02)


@pytest.fixture
def scenario(tmp_path, fake_worker):
    from baseline_common.config import run_directory, validate_config
    from baseline_common.data import file_sha256
    from baseline_common.orchestration import Runner, Stage

    events = tmp_path / "events"
    events.mkdir()
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"id":"train","prompt":"one","response":"two"}\n')
    validation.write_text('{"id":"validation","prompt":"three","response":"four"}\n')
    state_dir = tmp_path / "state"
    lock_dir = tmp_path / "locks"
    gpu_groups = {"qwen": ("0", "1"), "llama": ("2", "3")}
    groups = {}
    for group in gpu_groups:
        groups[group] = []
        for method in ("kd", "abkd", "skd", "distillm2"):
            config = validate_config(dict(
                name=f"{group}_{method}", pair=group, method=method, dataset="fixture",
                teacher_model=str(tmp_path / f"{group}_teacher"),
                student_model=str(tmp_path / f"{group}_student"),
                train_file=str(train), validation_file=str(validation),
                output_root=str(tmp_path / "outputs"), max_steps=2,
                gradient_accumulation_steps=1, save_steps=1, eval_steps=1,
            ))
            path = tmp_path / f"{group}_{method}.json"
            path.write_text(json.dumps(config))
            groups[group].append(Stage(group=group, kind="train", method=method,
                                       config=config, config_path=path,
                                       output_path=run_directory(config)))

    def manifest_builder(config):
        return {"config": config, "world_size": 1, "vocabulary_alignment": None,
                "train_sha256": file_sha256(config["train_file"]),
                "validation_sha256": file_sha256(config["validation_file"])}

    options = {}

    def command_builder(stage, resume):
        payload = dict(stage=stage.key, group=stage.group, kind=stage.kind,
                       output=str(stage.output_path), config=stage.config,
                       manifest=manifest_builder(stage.config) if stage.kind == "train" else {},
                       events=str(events), resume=str(resume) if resume else None,
                       **options.get(stage.key, {}))
        return [sys.executable, str(fake_worker), json.dumps(payload)]

    def runner(**overrides):
        return Runner(**dict(dict(groups=groups, gpu_groups=gpu_groups,
                                  state_dir=state_dir, lock_dir=lock_dir,
                                  command_builder=command_builder,
                                  manifest_builder=manifest_builder), **overrides))

    return SimpleNamespace(groups=groups, events=events, state_dir=state_dir,
                           lock_dir=lock_dir, gpu_groups=gpu_groups, options=options,
                           runner=runner, manifest_builder=manifest_builder,
                           worker=fake_worker, tmp_path=tmp_path)


def test_groups_overlap_methods_are_ordered_and_child_environments_are_isolated(scenario, monkeypatch):
    # The first jobs cannot finish until both have started. A sequential
    # scheduler therefore fails this test instead of merely running slowly.
    for group in scenario.groups.values():
        scenario.options[group[0].key] = {"barrier": True}
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                 "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "ROLE_RANK",
                 "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"):
        monkeypatch.setenv(name, "inherited-must-be-removed")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7,6,5,4")
    assert scenario.runner().run() == 0
    events = read_events(scenario.events)
    starts = [event for event in events if event["event"] == "start"]
    assert len(starts) == 8
    first_ends = [event["time"] for event in events if event["event"] == "end" and ":kd:" in event["stage"]]
    first_starts = [event["time"] for event in starts if ":kd:" in event["stage"]]
    assert max(first_starts) < min(first_ends)
    for group, stages in scenario.groups.items():
        actual = [event for event in events if event["group"] == group]
        assert [event["stage"] for event in actual if event["event"] == "start"] == [stage.key for stage in stages]
        assert [event["event"] for event in actual] == ["start", "end"] * 4
        assert all(event["cuda"] == ",".join(scenario.gpu_groups[group]) for event in actual)
        assert all(event["ranks"] == {} for event in actual)


def test_failure_stops_only_its_group_and_restart_resumes_latest_committed_checkpoint(scenario):
    failing = scenario.groups["qwen"][1]
    (scenario.events / (failing.key + ".fail_once")).touch()
    assert scenario.runner().run() == 1
    before = read_events(scenario.events)
    assert [row["stage"] for row in before if row["event"] == "start" and row["group"] == "qwen"] == [
        stage.key for stage in scenario.groups["qwen"][:2]
    ]
    assert [row["stage"] for row in before if row["event"] == "end" and row["group"] == "llama"] == [
        stage.key for stage in scenario.groups["llama"]
    ]
    assert scenario.runner().run() == 0
    after = read_events(scenario.events)[len(before):]
    starts = [row for row in after if row["event"] == "start"]
    assert [row["stage"] for row in starts] == [stage.key for stage in scenario.groups["qwen"][1:]]
    assert Path(starts[0]["resume"]) == failing.output_path / "checkpoints/step_000001"
    assert all(row["resume"] is None for row in starts[1:])
    finished = len(read_events(scenario.events))
    assert scenario.runner().run() == 0
    assert len(read_events(scenario.events)) == finished


def test_child_exit_zero_without_completion_does_not_advance_group(scenario):
    first = scenario.groups["qwen"][0]
    scenario.options[first.key] = {"incomplete_success": True}
    assert scenario.runner().run() == 1
    starts = [row["stage"] for row in read_events(scenario.events)
              if row["event"] == "start" and row["group"] == "qwen"]
    assert starts == [first.key]
    scenario.options.clear()
    assert scenario.runner().run() == 0
    resumed = [row for row in read_events(scenario.events)
               if row["event"] == "start" and row["stage"] == first.key]
    assert Path(resumed[-1]["resume"]) == first.output_path / "checkpoints/step_000001"


def test_pair_generation_is_ordered_before_distillm2_and_resumes_its_existing_journal(scenario):
    from baseline_common.orchestration import Runner, Stage

    training = scenario.groups["qwen"][-1]
    canonical = Path(training.config["train_file"])
    paired = scenario.tmp_path / "paired" / "qwen.seed_42.jsonl"
    training.config["train_file"] = str(paired)
    training.config_path.write_text(json.dumps(training.config))
    generation = Stage(group="qwen", kind="pairs", method="distillm2", config=training.config,
                       config_path=training.config_path, input_path=canonical, output_path=paired)
    scenario.groups["qwen"].insert(-1, generation)
    assert "--resume" in Runner._command(generation, None)
    (scenario.events / (generation.key + ".fail_once")).touch()
    assert scenario.runner().run() == 1
    before = read_events(scenario.events)
    assert [row["stage"] for row in before if row["event"] == "start" and row["group"] == "qwen"] == [
        stage.key for stage in scenario.groups["qwen"][:-1]
    ]
    assert not paired.exists()
    assert len(paired.with_suffix(".progress.jsonl").read_text().splitlines()) == 1
    assert scenario.runner().run() == 0
    after = read_events(scenario.events)[len(before):]
    assert [row["stage"] for row in after if row["event"] == "start"] == [generation.key, training.key]
    assert next(row for row in after if row["event"] == "end" and row["stage"] == generation.key)["resumed_rows"] == 1
    assert [json.loads(line)["id"] for line in paired.read_text().splitlines()] == ["a", "b"]
    count = len(read_events(scenario.events))
    assert scenario.runner().run() == 0
    final = read_events(scenario.events)[count:]
    assert [row["stage"] for row in final if row["event"] == "start"] == [generation.key]
    assert next(row for row in final if row["event"] == "end")["resumed_rows"] == 2


@pytest.mark.parametrize("damage", ["missing_final", "result_incomplete"])
def test_completed_status_never_overrides_incomplete_training_artifacts(scenario, damage):
    assert scenario.runner().run() == 0
    before = len(read_events(scenario.events))
    stage = scenario.groups["qwen"][0]
    if damage == "missing_final":
        (stage.output_path / "final/model.safetensors").unlink()
    else:
        path = stage.output_path / "result.json"
        result = json.loads(path.read_text())
        result["complete"] = False
        path.write_text(json.dumps(result))
    assert scenario.runner().run() == 0
    starts = [row for row in read_events(scenario.events)[before:] if row["event"] == "start"]
    assert len(starts) == 1
    assert starts[0]["stage"] == stage.key
    assert Path(starts[0]["resume"]) == stage.output_path / "checkpoints/step_000002"


@pytest.mark.parametrize("change", ["config", "data"])
def test_changed_manifest_is_rejected_before_completed_runs_are_skipped(scenario, change):
    assert scenario.runner().run() == 0
    before = len(read_events(scenario.events))
    stage = scenario.groups["qwen"][0]
    if change == "config":
        stage.config["learning_rate"] *= 2
        stage.config_path.write_text(json.dumps(stage.config))
    else:
        Path(stage.config["train_file"]).write_text('{"id":"changed","prompt":"new","response":"data"}\n')
    assert scenario.runner().run() == 1
    assert len(read_events(scenario.events)) == before


def test_interruption_before_first_checkpoint_archives_partial_attempt_and_restarts(scenario):
    stage = scenario.groups["qwen"][0]
    stage.output_path.mkdir(parents=True)
    (stage.output_path / "manifest.json").write_text(json.dumps(scenario.manifest_builder(stage.config)))
    (stage.output_path / "metrics.jsonl").write_text('{"partial":')
    partial_checkpoint = stage.output_path / "checkpoints/step_000001"
    partial_checkpoint.mkdir(parents=True)
    (partial_checkpoint / "training.pt").write_bytes(b"torn uncommitted save")
    assert scenario.runner().run() == 0
    archived = list(stage.output_path.parent.glob(stage.output_path.name + ".attempt_*"))
    assert len(archived) == 1
    assert (archived[0] / "metrics.jsonl").read_text() == '{"partial":'
    assert (archived[0] / "checkpoints/step_000001/training.pt").read_bytes() == b"torn uncommitted save"
    start = next(row for row in read_events(scenario.events) if row["event"] == "start" and row["stage"] == stage.key)
    assert start["resume"] is None


def test_interruption_during_initial_manifest_write_archives_bytes_and_restarts(scenario):
    stage = scenario.groups["qwen"][0]
    stage.output_path.mkdir(parents=True)
    partials = {".manifest.json.123.tmp": b'{"config": {"method": "kd",',
                ".manifest.json.456.other.tmp": b'\x00partial write\n'}
    for name, payload in partials.items():
        (stage.output_path / name).write_bytes(payload)
    assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
    archived = list(stage.output_path.parent.glob(stage.output_path.name + ".attempt_*"))
    assert len(archived) == 1
    assert {path.name: path.read_bytes() for path in archived[0].iterdir()} == partials
    starts = [row for row in read_events(scenario.events) if row["event"] == "start"]
    assert len(starts) == 1
    assert starts[0]["resume"] is None
    assert json.loads((stage.output_path / "result.json").read_text())["complete"] is True


@pytest.mark.parametrize("unknown_entry", ["ordinary_file", "matching_directory", "mixed_entries"])
def test_nonempty_output_without_manifest_rejects_unrecognized_artifacts(scenario, unknown_entry):
    stage = scenario.groups["qwen"][0]
    stage.output_path.mkdir(parents=True)
    if unknown_entry == "matching_directory":
        directory = stage.output_path / ".manifest.json.123.tmp"
        directory.mkdir()
        (directory / "valuable.bin").write_bytes(b"must remain untouched")
    else:
        (stage.output_path / "unrecognized.bin").write_bytes(b"must remain untouched")
        if unknown_entry == "mixed_entries":
            (stage.output_path / ".manifest.json.123.tmp").write_bytes(b"partial manifest")
    before = {str(path.relative_to(stage.output_path)): path.read_bytes()
              for path in stage.output_path.rglob("*") if path.is_file()}
    assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 1
    assert read_events(scenario.events) == []
    assert not list(stage.output_path.parent.glob(stage.output_path.name + ".attempt_*"))
    assert {str(path.relative_to(stage.output_path)): path.read_bytes()
            for path in stage.output_path.rglob("*") if path.is_file()} == before


def test_latest_checkpoint_ignores_partial_later_saves_and_checks_committed_files(scenario):
    from baseline_common.orchestration import latest_checkpoint

    stage = scenario.groups["qwen"][0]
    assert latest_checkpoint(stage.output_path) is None
    assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
    assert latest_checkpoint(stage.output_path) == stage.output_path / "checkpoints/step_000002"
    partial = stage.output_path / "checkpoints/step_999999"
    partial.mkdir()
    (partial / "training.pt").write_bytes(b"partial")
    assert latest_checkpoint(stage.output_path) == stage.output_path / "checkpoints/step_000002"
    (stage.output_path / "checkpoints/step_000002/training.pt").unlink()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        latest_checkpoint(stage.output_path)


def test_completion_requires_expected_step_and_usable_export(scenario):
    from baseline_common.orchestration import is_training_complete

    stage = scenario.groups["qwen"][0]
    assert not is_training_complete(stage.output_path, expected_steps=2)
    assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
    assert is_training_complete(stage.output_path, expected_steps=2)
    assert not is_training_complete(stage.output_path, expected_steps=3)
    (stage.output_path / "final/model.safetensors").write_bytes(b"")
    assert not is_training_complete(stage.output_path, expected_steps=2)


def test_new_result_schema_requires_matching_final_export_commit_marker(scenario):
    from baseline_common.orchestration import is_training_complete

    stage = scenario.groups["qwen"][0]
    assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
    assert is_training_complete(stage.output_path, expected_steps=2)  # Legacy exports remain supported.
    result_path = stage.output_path / "result.json"
    result = json.loads(result_path.read_text())
    result["interrupted"] = False
    result_path.write_text(json.dumps(result))
    assert not is_training_complete(stage.output_path, expected_steps=2)
    marker = stage.output_path / "final/complete.json"
    marker.write_text('{"step": 1}')
    assert not is_training_complete(stage.output_path, expected_steps=2)
    marker.write_text('{"step": 2}')
    assert is_training_complete(stage.output_path, expected_steps=2)


@pytest.mark.parametrize("relative_path", [
    "final/model.safetensors",
    "checkpoints/step_000002/student/model.safetensors",
    "checkpoints/step_000002/training.pt",
    "checkpoints/step_000002/rng_rank0.pt",
])
def test_nonempty_truncated_artifacts_cannot_claim_completion(scenario, relative_path):
    from baseline_common.orchestration import is_training_complete, latest_checkpoint

    stage = scenario.groups["qwen"][0]
    assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
    damaged = stage.output_path / relative_path
    content = damaged.read_bytes()
    damaged.write_bytes(content[:len(content) // 2])
    assert damaged.stat().st_size > 0
    assert not is_training_complete(stage.output_path, expected_steps=2)
    if relative_path.startswith("checkpoints/"):
        with pytest.raises(ValueError, match="claims completion"):
            latest_checkpoint(stage.output_path)
        assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 1
        assert len([row for row in read_events(scenario.events) if row["event"] == "start"]) == 1
    else:
        # A damaged final export can be reconstructed from its intact checkpoint.
        assert latest_checkpoint(stage.output_path) == stage.output_path / "checkpoints/step_000002"
        assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
        assert is_training_complete(stage.output_path, expected_steps=2)


HARNESS = r'''
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from baseline_common.data import file_sha256
from baseline_common.orchestration import Runner, Stage

spec = json.loads(Path(sys.argv[2]).read_text())
groups = {}
for group, rows in spec["groups"].items():
    groups[group] = []
    for row in rows:
        for key in ("config_path", "output_path", "input_path"):
            if row.get(key) is not None:
                row[key] = Path(row[key])
        groups[group].append(Stage(**row))

def manifest_builder(config):
    return {"config": config, "world_size": 1, "vocabulary_alignment": None,
            "train_sha256": file_sha256(config["train_file"]),
            "validation_sha256": file_sha256(config["validation_file"])}

def command_builder(stage, resume):
    payload = dict(stage=stage.key, group=stage.group,
                   output=str(stage.output_path), config=stage.config,
                   manifest=manifest_builder(stage.config), events=spec["events"],
                   resume=str(resume) if resume else None, hold=spec["hold"])
    return [sys.executable, spec["worker"], json.dumps(payload)]

runner = Runner(groups=groups, gpu_groups=spec["gpu_groups"],
                state_dir=Path(spec["state_dir"]), lock_dir=Path(spec["lock_dir"]),
                command_builder=command_builder, manifest_builder=manifest_builder)
sys.exit(runner.run())
'''


def launch_harness(scenario, *, state_dir=None, hold=True):
    from dataclasses import asdict

    stage = scenario.groups["qwen"][0]
    spec = dict(groups={"qwen": [asdict(stage)]},
                gpu_groups={"qwen": scenario.gpu_groups["qwen"]},
                state_dir=str(state_dir or scenario.state_dir),
                lock_dir=str(scenario.lock_dir), worker=str(scenario.worker),
                events=str(scenario.events), hold=hold)
    harness = scenario.tmp_path / "runner_harness.py"
    harness.write_text(textwrap.dedent(HARNESS))
    spec_path = scenario.tmp_path / f"harness_{time.time_ns()}.json"
    spec_path.write_text(json.dumps(spec, default=str))
    return subprocess.Popen([sys.executable, str(harness), str(Path(__file__).resolve().parents[1]),
                             str(spec_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)


def stop_harness(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


@pytest.mark.parametrize("different_state_dir", [False, True])
def test_concurrent_orchestrators_cannot_claim_same_run_or_same_gpu_pair(scenario, different_state_dir):
    stage = scenario.groups["qwen"][0]
    first = launch_harness(scenario)
    second = None
    try:
        wait_for_file(scenario.events / (stage.key + ".holding"))
        second = launch_harness(scenario, state_dir=scenario.tmp_path / "other_state" if different_state_dir else None,
                                hold=False)
        output = second.communicate(timeout=15)[0]
        assert second.returncode == 1, output
        starts = [row for row in read_events(scenario.events) if row["event"] == "start"]
        assert len(starts) == 1
        assert first.poll() is None
    finally:
        if second is not None:
            stop_harness(second)
        stop_harness(first)


def test_sigterm_stops_children_and_next_invocation_resumes_committed_checkpoint(scenario):
    stage = scenario.groups["qwen"][0]
    first = launch_harness(scenario)
    try:
        holding = scenario.events / (stage.key + ".holding")
        wait_for_file(holding)
        worker_pid = int(holding.read_text())
        first.send_signal(signal.SIGTERM)
        output = first.communicate(timeout=15)[0]
        assert first.returncode == 130, output
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        assert scenario.runner(groups={"qwen": [stage]}, gpu_groups={"qwen": ("0", "1")}).run() == 0
        starts = [row for row in read_events(scenario.events) if row["event"] == "start"]
        assert len(starts) == 2
        assert Path(starts[-1]["resume"]) == stage.output_path / "checkpoints/step_000001"
    finally:
        stop_harness(first)


def test_orphaned_worker_keeps_gpu_lock_until_it_exits(scenario):
    stage = scenario.groups["qwen"][0]
    first = launch_harness(scenario)
    second = None
    worker_pid = None
    try:
        holding = scenario.events / (stage.key + ".holding")
        wait_for_file(holding)
        worker_pid = int(holding.read_text())
        first.kill()
        first.wait(timeout=10)
        os.kill(worker_pid, 0)  # The child still owns its inherited lock descriptor.
        second = launch_harness(scenario, state_dir=scenario.tmp_path / "other_state", hold=False)
        output = second.communicate(timeout=15)[0]
        assert second.returncode == 1, output
        assert len([row for row in read_events(scenario.events) if row["event"] == "start"]) == 1
    finally:
        if second is not None:
            stop_harness(second)
        stop_harness(first)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
