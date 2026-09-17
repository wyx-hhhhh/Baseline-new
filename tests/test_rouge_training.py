"""Training-only defaults and opt-in ROUGE-L selection preserve exact updates."""
import copy
import importlib
import json
from pathlib import Path
import random

import pytest
import torch

from baseline_common.config import validate_config
from test_training import limited_cpu_threads, tiny_experiment, _json_lines, _state


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_default_training_performs_no_evaluation_and_resumes_exactly(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    cfg = copy.deepcopy(cfg)
    del cfg["eval_during_training"]
    cfg = validate_config(cfg)
    assert cfg["eval_during_training"] is False
    assert cfg["evaluation_metric"] == "rouge_l"

    def unexpected_evaluation(*_args, **_kwargs):
        raise AssertionError("Training-only execution must not evaluate or generate validation outputs")

    monkeypatch.setattr(training, "evaluate", unexpected_evaluation)
    complete = training.train(cfg)
    interrupted_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]).with_name("train_only_resumed"))}
    interrupted = training.train(interrupted_cfg, stop_after_steps=1)
    checkpoint = interrupted / "checkpoints/step_000001"
    # Enabling a new validation protocol midway is not an exact resume.
    with pytest.raises(ValueError, match="Resume manifest differs"):
        training.train({**interrupted_cfg, "eval_during_training": True}, resume=str(checkpoint))
    resumed = training.train(interrupted_cfg, resume=str(checkpoint))
    for output in (complete, resumed):
        assert not (output / "validation.jsonl").exists()
        assert not (output / "best_checkpoint.json").exists()
        result = json.loads((output / "result.json").read_text())
        assert result["complete"] and result["best_step"] is None
        assert (output / "final/complete.json").is_file()
    assert _json_lines(resumed / "metrics.jsonl") == _json_lines(complete / "metrics.jsonl")
    _assert_nested_equal(_state(resumed / "final"), _state(complete / "final"))


def test_rouge_selection_maximizes_score_and_keeps_earliest_tie_on_resume(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    cfg = {**cfg, "max_steps": 4, "eval_during_training": True}
    scores = iter([0.2, 0.8, 0.8, 0.5])
    monkeypatch.setattr(training, "evaluate", lambda *_args: {"rouge_l": next(scores), "examples": 1})
    paused = training.train(cfg, stop_after_steps=2)
    best_before = json.loads((paused / "best_checkpoint.json").read_text())
    assert best_before["step"] == 2 and best_before["rouge_l"] == 0.8
    output = training.train(cfg, resume=str(paused / "checkpoints/step_000002"))
    metrics = _json_lines(output / "validation.jsonl")
    assert [row["rouge_l"] for row in metrics] == [0.2, 0.8, 0.8, 0.5]
    assert all("reference_nll" not in row for row in metrics)
    best = json.loads((output / "best_checkpoint.json").read_text())
    assert best["step"] == 2 and best["rouge_l"] == 0.8
    assert best["path"] == "checkpoints/step_000002/student"
    assert (output / best["path"]).is_dir()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["selection_metric"] == "rouge_l"
    result = json.loads((output / "result.json").read_text())
    assert result["best_step"] == 2 and result["best_score"] == 0.8


def test_rouge_validation_leaves_optimizer_rng_and_training_trajectory_unchanged(tiny_experiment, monkeypatch):
    training = importlib.import_module("baseline_common.train")
    cfg, _, _ = tiny_experiment
    cfg = {**cfg, "method": "skd", "eval_during_training": False}
    baseline = training.train(cfg)
    actual_evaluate = training.evaluate
    calls = []

    def evaluate_and_consume_rng(student, *args, **kwargs):
        assert student.training
        before = {name: value.detach().clone() for name, value in student.named_parameters()}
        original_generate = student.generate

        def generate_and_consume_rng(*generation_args, **generation_kwargs):
            # Exercise the evaluator's RNG guard using a generation hook:
            # subsequent SKD sampling and dropout must follow the same stream.
            random.random()
            torch.rand(7)
            return original_generate(*generation_args, **generation_kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(student, "generate", generate_and_consume_rng)
            result = actual_evaluate(student, *args, **kwargs)
        assert student.training
        for name, value in student.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        assert 0 <= result["rouge_l"] <= 100 and "reference_nll" not in result
        calls.append(result)
        return result

    monkeypatch.setattr(training, "evaluate", evaluate_and_consume_rng)
    evaluated_cfg = {**cfg, "eval_during_training": True,
                     "output_root": str(Path(cfg["output_root"]).with_name("with_rouge"))}
    evaluated = training.train(evaluated_cfg)
    assert len(calls) == 2
    assert _json_lines(evaluated / "metrics.jsonl") == _json_lines(baseline / "metrics.jsonl")
    _assert_nested_equal(_state(evaluated / "final"), _state(baseline / "final"))
    checkpoint = "checkpoints/step_000002"
    left = torch.load(evaluated / checkpoint / "training.pt", weights_only=False)
    right = torch.load(baseline / checkpoint / "training.pt", weights_only=False)
    _assert_nested_equal(left["optimizer"], right["optimizer"])
    _assert_nested_equal(left["scheduler"], right["scheduler"])
    _assert_nested_equal(left["state"]["counters"], right["state"]["counters"])
    _assert_nested_equal(torch.load(evaluated / checkpoint / "rng_rank0.pt", weights_only=False),
                         torch.load(baseline / checkpoint / "rng_rank0.pt", weights_only=False))


@pytest.mark.parametrize("changes", [
    {"evaluation_metric": "reference_nll"}, {"evaluation_metric": "bleu"},
    {"eval_during_training": "false"}, {"eval_during_training": 0},
    {"eval_max_new_tokens": 0}, {"eval_max_new_tokens": True},
])
def test_config_rejects_incompatible_evaluation_protocol(changes):
    from test_config_models import config

    with pytest.raises(ValueError):
        validate_config(config(**changes))
