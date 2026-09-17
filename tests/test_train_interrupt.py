"""Real tiny-model interruption, safe commits and exact resume, entirely offline."""
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

from test_training import limited_cpu_threads, tiny_experiment, _json_lines, _state


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize("method", ["kd", "skd"])
def test_signal_mid_accumulation_commits_step_and_exactly_resumes(tiny_experiment, monkeypatch, signum, method):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    cfg = {**cfg, "method": method, "save_steps": 100, "eval_steps": 100}
    expected = training.train(cfg)
    resumed_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]).with_name("interrupted"))}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    actual_response_loss = training.response_loss
    teacher_instances = []
    calls = 0

    def interrupt_after_forward(student, teacher, *args, **kwargs):
        nonlocal calls
        loss = actual_response_loss(student, teacher, *args, **kwargs)
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        teacher_instances.append(teacher)
        calls += 1
        if calls == 1:
            os.kill(os.getpid(), signum)
        return loss

    with monkeypatch.context() as patch:
        patch.setattr(training, "response_loss", interrupt_after_forward)
        interrupted = training.train(resumed_cfg)
    assert calls == cfg["gradient_accumulation_steps"]
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    assert teacher_instances and all(all(p.grad is None for p in teacher.parameters()) for teacher in teacher_instances)
    result = json.loads((interrupted / "result.json").read_text())
    assert result["step"] == 1 and not result["complete"] and result["interrupted"]
    assert result["signal"] == signum
    checkpoint = interrupted / "checkpoints/step_000001"
    assert (checkpoint / "complete.json").is_file()
    assert (checkpoint / "training.pt").is_file()
    assert (checkpoint / "rng_rank0.pt").is_file()
    assert not (interrupted / "final").exists()
    assert not (interrupted / "validation.jsonl").exists()  # No extra evaluation changes the schedule.
    manifest = json.loads((interrupted / "manifest.json").read_text())
    assert manifest["config"] == resumed_cfg
    resumed = training.train(resumed_cfg, resume=str(checkpoint))
    assert _json_lines(resumed / "metrics.jsonl") == _json_lines(expected / "metrics.jsonl")
    assert _json_lines(resumed / "validation.jsonl") == _json_lines(expected / "validation.jsonl")
    assert json.loads((resumed / "result.json").read_text())["complete"]
    resumed_state = _state(resumed / "final")
    for key, value in _state(expected / "final").items():
        torch.testing.assert_close(resumed_state[key], value, rtol=0, atol=0)


def test_signal_during_model_load_stops_without_optimizer_update(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    actual_load = training.load_model
    loads = []

    def interrupt_load(*args, **kwargs):
        model = actual_load(*args, **kwargs)
        loads.append(model)
        os.kill(os.getpid(), signal.SIGTERM)
        return model

    monkeypatch.setattr(training, "load_model", interrupt_load)
    output = training.train(cfg)
    assert len(loads) == 1
    result = json.loads((output / "result.json").read_text())
    assert result["step"] == 0 and result["interrupted"] and not result["complete"]
    assert result["last_checkpoint"] is None
    assert (output / "manifest.json").is_file()
    assert not (output / "checkpoints").exists()
    assert not (output / "metrics.jsonl").exists()


def test_signal_during_checkpoint_finishes_commit_and_restores_handlers(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    actual_save = training.save_checkpoint
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt_save(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return actual_save(*args, **kwargs)

    monkeypatch.setattr(training, "save_checkpoint", interrupt_save)
    output = training.train(cfg)
    assert signal.getsignal(signal.SIGTERM) == previous
    result = json.loads((output / "result.json").read_text())
    assert result["step"] == 1 and result["interrupted"]
    checkpoint = Path(result["last_checkpoint"])
    assert json.loads((checkpoint / "complete.json").read_text()) == {"step": 1, "world_size": 1}
    assert torch.load(checkpoint / "training.pt", weights_only=False)["state"]["step"] == 1
    assert _state(checkpoint / "student")


def test_last_step_checkpoint_recovers_missing_final_export_without_new_updates(tiny_experiment):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    output = training.train(cfg)
    metrics = (output / "metrics.jsonl").read_bytes()
    validation = (output / "validation.jsonl").read_bytes()
    expected = _state(output / "final")
    (output / "final").rename(output / "old_final")
    (output / "result.json").rename(output / "old_result.json")
    training.train(cfg, resume=str(output / "checkpoints/step_000002"))
    assert (output / "metrics.jsonl").read_bytes() == metrics
    assert (output / "validation.jsonl").read_bytes() == validation
    assert json.loads((output / "result.json").read_text())["complete"]
    assert json.loads((output / "final/complete.json").read_text()) == {"step": 2, "world_size": 1}
    actual = _state(output / "final")
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("legacy", [False, True])
def test_interrupted_export_recovery_revokes_stale_completion_and_recovers_again(tiny_experiment, monkeypatch, legacy):
    training = importlib.import_module("baseline_common.train")
    from baseline_common.orchestration import is_training_complete
    cfg, _, _ = tiny_experiment
    output = training.train(cfg)
    checkpoint = output / "checkpoints/step_000002"
    final = output / "final"
    expected = _state(final)
    metrics, validation = (output / "metrics.jsonl").read_bytes(), (output / "validation.jsonl").read_bytes()
    if legacy:
        result = json.loads((output / "result.json").read_text())
        result.pop("interrupted")
        result.pop("signal")
        (output / "result.json").write_text(json.dumps(result))
        (final / "complete.json").unlink()
    (final / "model.safetensors").write_bytes(b"broken export requiring recovery")
    assert not is_training_complete(output, expected_steps=2)
    actual_load = training.load_model

    def inject_interrupted_export(path, *args, **kwargs):
        model = actual_load(path, *args, **kwargs)
        if Path(path) == checkpoint / "student":
            actual_save = model.save_pretrained

            def fail_during_export(destination, *save_args, **save_kwargs):
                if Path(destination) == final:
                    pending = json.loads((output / "result.json").read_text())
                    assert pending["complete"] is False and pending["export_pending"] is True
                    assert not (final / "complete.json").exists()
                    (final / "model.safetensors").write_bytes(b"interrupted a second time during export")
                    raise RuntimeError("simulated second export interruption")
                return actual_save(destination, *save_args, **save_kwargs)

            model.save_pretrained = fail_during_export
        return model

    with monkeypatch.context() as patch:
        patch.setattr(training, "load_model", inject_interrupted_export)
        with pytest.raises(RuntimeError, match="simulated second export interruption"):
            training.train(cfg, resume=str(checkpoint))
    pending = json.loads((output / "result.json").read_text())
    assert pending["complete"] is False and pending["export_pending"] is True
    assert pending["step"] == 2 and Path(pending["last_checkpoint"]) == checkpoint
    assert not is_training_complete(output, expected_steps=2)
    training.train(cfg, resume=str(checkpoint))
    assert is_training_complete(output, expected_steps=2)
    assert "export_pending" not in json.loads((output / "result.json").read_text())
    assert (output / "metrics.jsonl").read_bytes() == metrics
    assert (output / "validation.jsonl").read_bytes() == validation
    actual = _state(final)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def test_metadata_atomic_replacement_preserves_previous_on_failure(tmp_path, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    target = tmp_path / "result.json"
    training.write_json(target, {"step": 1})
    previous = target.read_bytes()

    def fail_replace(*_args):
        raise OSError("simulated write interruption")

    monkeypatch.setattr(training.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated write interruption"):
        training.write_json(target, {"step": 2})
    assert target.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


def test_artifact_sync_precedes_checkpoint_and_final_commit_markers(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    cfg = {**cfg, "save_steps": 100, "eval_steps": 100}
    actual_sync, actual_write = training.sync_artifacts, training.write_json
    synced, committed = set(), []

    def observe_sync(*paths):
        actual_sync(*paths)
        for item in paths:
            path = Path(item)
            if path.is_dir():
                synced.update(child for child in path.rglob("*") if child.is_file())
            else:
                synced.add(path)

    def observe_write(path, value):
        path = Path(path)
        if path.name == "complete.json":
            assert not path.exists()
            required = {child for child in path.parent.rglob("*") if child.is_file()}
            assert required and required <= synced
            committed.append(path)
        return actual_write(path, value)

    monkeypatch.setattr(training, "sync_artifacts", observe_sync)
    monkeypatch.setattr(training, "write_json", observe_write)
    output = training.train(cfg)
    assert committed == [output / "checkpoints/step_000002/complete.json", output / "final/complete.json"]
    assert output / "checkpoints/step_000002/training.pt" in synced
    assert output / "checkpoints/step_000002/rng_rank0.pt" in synced
    assert output / "checkpoints/step_000002/student/model.safetensors" in synced
    assert output / "final/model.safetensors" in synced


def test_artifact_sync_failure_never_publishes_checkpoint_marker(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    actual_sync = training.sync_artifacts

    def failed_sync(*paths):
        if any(Path(path).name == "training.pt" for path in paths):
            raise OSError("simulated storage flush failure")
        return actual_sync(*paths)

    monkeypatch.setattr(training, "sync_artifacts", failed_sync)
    with pytest.raises(OSError, match="simulated storage flush failure"):
        training.train(cfg)
    from baseline_common.config import run_directory
    output = run_directory(cfg)
    checkpoint = output / "checkpoints/step_000001"
    assert (checkpoint / "training.pt").is_file()
    assert (checkpoint / "student/model.safetensors").is_file()
    assert not (checkpoint / "complete.json").exists()
    assert not (output / "result.json").exists()


def test_sync_artifacts_flushes_only_requested_export_files(tmp_path, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    old = tmp_path / "step_000001"
    current = tmp_path / "step_000002/student"
    old.mkdir()
    current.mkdir(parents=True)
    (old / "old_weights").write_bytes(b"old")
    (current / "weights").write_bytes(b"weights")
    (current / "config").write_bytes(b"config")
    paths = []
    actual_fsync = os.fsync

    def observe_fsync(descriptor):
        paths.append(Path(os.readlink(f"/proc/self/fd/{descriptor}")))
        return actual_fsync(descriptor)

    monkeypatch.setattr(training.os, "fsync", observe_fsync)
    training.sync_artifacts(current)
    assert set(paths) == {current / "weights", current / "config", current, current.parent}
    assert paths.index(current / "weights") < paths.index(current)
    assert paths.index(current / "config") < paths.index(current)
    assert paths.index(current) < paths.index(current.parent)


def test_manifest_helpers_are_read_only_and_reject_changed_schedule(tiny_experiment):
    training = importlib.import_module("baseline_common.train")
    cfg, tokenizer, _ = tiny_experiment
    expected = training.build_run_manifest(cfg)
    assert expected == training.build_run_manifest(cfg, tokenizer=tokenizer)
    previous = json.loads(json.dumps(expected))
    training.validate_run_manifest(previous, expected)
    assert previous == expected
    changed = training.build_run_manifest({**cfg, "max_steps": 3})
    with pytest.raises(ValueError, match="Resume manifest differs"):
        training.validate_run_manifest(previous, changed)
    assert not Path(cfg["output_root"]).exists()


def test_cuda_rng_restore_preserves_used_pair_and_ignores_invisible_legacy_devices(monkeypatch):
    training = importlib.import_module("baseline_common.train")
    state = training.rng_state()
    saved = [torch.tensor([i], dtype=torch.uint8) for i in range(4)]
    state["cuda"] = saved
    restored = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda rng, device: restored.append((rng, device)))
    training.restore_rng(state, required_cuda_devices=("cuda:0", "cuda:1"))
    assert [device for _rng, device in restored] == [0, 1]
    assert restored[0][0] is saved[0] and restored[1][0] is saved[1]
    state["cuda"] = saved[:1]
    with pytest.raises(ValueError, match="no CUDA RNG state for required device cuda:1"):
        training.restore_rng(state, required_cuda_devices=("cuda:0", "cuda:1"))


def test_cli_signal_exits_75_and_separate_process_resumes(tiny_experiment, tmp_path):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    cfg = {**cfg, "save_steps": 100, "eval_steps": 100}
    expected = training.train(cfg)
    cfg = {**cfg, "output_root": str(tmp_path / "subprocess_results")}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(cfg))
    source = """
import importlib, os, signal, torch
torch.set_num_threads(2)
training = importlib.import_module('baseline_common.train')
original = training.response_loss
fired = False
def interrupt(*args, **kwargs):
    global fired
    loss = original(*args, **kwargs)
    if not fired:
        fired = True
        os.kill(os.getpid(), signal.SIGTERM)
    return loss
training.response_loss = interrupt
from scripts.train import main
raise SystemExit(main())
"""
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}
    interrupted = subprocess.run([sys.executable, "-c", source, "--config", str(config_file)],
                                 cwd=root, env=environment, text=True, capture_output=True, timeout=90)
    assert interrupted.returncode == 75, interrupted.stdout + interrupted.stderr
    from baseline_common.config import run_directory
    output = run_directory(cfg)
    checkpoint = output / "checkpoints/step_000001"
    assert (checkpoint / "complete.json").is_file()
    resumed = subprocess.run([sys.executable, "scripts/train.py", "--config", str(config_file),
                              "--resume", str(checkpoint)], cwd=root, env=environment,
                              text=True, capture_output=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert _json_lines(output / "metrics.jsonl") == _json_lines(expected / "metrics.jsonl")
    actual = _state(output / "final")
    for key, value in _state(expected / "final").items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
