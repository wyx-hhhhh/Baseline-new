"""Tiny native HF inference exercises math durability independently of grading."""
import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, write_jsonl
from baseline_common.pair_progress import atomic_json, atomic_jsonl, read_progress_rows


def fake_grade(prediction, reference, benchmark):
    parsed = prediction.strip() if prediction.strip() in {"1", "2"} else None
    gold = reference.removeprefix("#### ")
    return {"correct": parsed == gold, "predicted_answer": parsed, "reference_answer": gold,
            "parse_error": None if parsed else "prediction could not be parsed",
            "extraction_failed": parsed is None, "reference_override": None}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    torch.set_num_threads(2)
    words = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>",
             "training", "development", "first", "second", "third", "heldout", "1", "2"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
        unk_token="<unk>", bos_token="<bos>", eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>"])
    tokenizer.chat_template = "{{ bos_token }} <user> {{ messages[0]['content'] }} <assistant>"
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=128, bos_token_id=2,
        eos_token_id=3, pad_token_id=0))
    model_path = tmp_path / "final"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    train = [{"id": "train", "prompt": "training", "response": "#### 1", "source_group": "training"}]
    dev = [{"id": "dev", "prompt": "development", "response": "#### 1", "source_group": "development"}]
    test = [{"id": word, "prompt": word + " heldout", "response": "#### 1",
             "source_group": word + "-group", "benchmark": "gsm8k"} for word in ("first", "second", "third")]
    for name, rows in (("train", train), ("dev", dev), ("test", test)):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    cfg = validate_config(dict(name="tiny_math_kd", pair="tiny", method="kd", dataset="metamathqa_fixed",
        teacher_model=str(tmp_path / "missing-teacher"), student_model=str(tmp_path / "missing-student"),
        train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "dev.jsonl"),
        dtype="float32", device="cpu", max_prompt_tokens=64, max_new_tokens=3,
        eval_max_new_tokens=1, eval_max_examples=1))
    cfg_path = tmp_path / "config.json"
    atomic_json(cfg_path, cfg)
    common = importlib.import_module("baseline_common.math_evaluation")
    module = importlib.import_module("scripts.evaluate_math")
    def validate(reference, benchmark):
        if reference not in {"#### 1", "#### 2"}:
            raise ValueError("unparseable reference")
        return reference.removeprefix("#### ")
    def definition(benchmark):
        if benchmark not in {"gsm8k", "math"}:
            raise ValueError("unknown benchmark")
        return {"name": "accuracy", "benchmark": benchmark, "range": [0, 100], "implementation": "test-only"}
    monkeypatch.setattr(common, "math_score", fake_grade)
    monkeypatch.setattr(common, "validate_reference", validate)
    monkeypatch.setattr(common, "metric_definition", definition)
    monkeypatch.setattr(module, "metric_definition", definition)
    output = tmp_path / "evaluation"
    def run(**kwargs):
        options = {"data_path": tmp_path / "test.jsonl", "benchmark": "gsm8k",
                   "max_prompt_tokens": 64, "max_new_tokens": 3}
        options.update(kwargs)
        return module.run_evaluation(options.pop("cfg", cfg), model_path=model_path,
                                     output=options.pop("output", output), **options)
    return dict(cfg=cfg, cfg_path=cfg_path, module=module, common=common,
                model=model_path, tokenizer=tokenizer, output=output, train=train, dev=dev,
                test=test, test_path=tmp_path / "test.jsonl", run=run,
                paths=module.evaluation_paths(output))


def forbid_load(*_args, **_kwargs):
    raise AssertionError("Model weights must not be loaded")


def interrupt_after_first(exp, monkeypatch):
    original = exp["module"].append_jsonl
    def interrupt(path, row):
        original(path, row)
        raise InterruptedError("crash after durable append")
    monkeypatch.setattr(exp["module"], "append_jsonl", interrupt)
    with pytest.raises(InterruptedError):
        exp["run"]()
    monkeypatch.setattr(exp["module"], "append_jsonl", original)


def test_native_inference_all_questions_and_cache_only_complete_verification(experiment, monkeypatch):
    exp = experiment
    assert exp["run"]()
    rows = read_progress_rows(exp["paths"]["predictions"])
    metrics = json.loads(exp["paths"]["metrics"].read_text())
    assert len(rows) == metrics["total"] == metrics["examples"] == 3
    assert metrics["accuracy_percent"] == metrics["pass_at_1"] == sum(row["correct"] for row in rows) / 3 * 100
    assert metrics["unparseable_predictions"] == sum(row["extraction_failed"] for row in rows)
    assert "rouge_l" not in metrics
    assert all(row["benchmark"] == "gsm8k" and row["reference"] == "#### 1" for row in rows)
    assert metrics["evaluation"]["decoding"] == "greedy"
    assert metrics["evaluation"]["samples_per_question"] == 1
    assert metrics["evaluation"]["prompt_truncation"] == "forbidden"
    before = {key: (file_sha256(path), path.stat().st_mtime_ns)
              for key, path in exp["paths"].items() if key in {"metrics", "manifest", "predictions"}}
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    assert exp["run"](resume=True)
    assert before == {key: (file_sha256(exp["paths"][key]), exp["paths"][key].stat().st_mtime_ns) for key in before}


def test_signal_stop_torn_tail_resume_and_fresh_run_match(experiment, monkeypatch, tmp_path):
    exp = experiment
    original = exp["module"].append_jsonl
    def interrupt(path, row):
        original(path, row)
        signal.raise_signal(signal.SIGTERM)
    monkeypatch.setattr(exp["module"], "append_jsonl", interrupt)
    assert exp["run"]() is False
    prefix = exp["paths"]["predictions"].read_bytes()
    assert len(read_progress_rows(exp["paths"]["predictions"])) == 1
    assert not exp["paths"]["manifest"].exists()
    with exp["paths"]["predictions"].open("ab") as stream:
        stream.write(b'{"id": "torn')
    monkeypatch.setattr(exp["module"], "append_jsonl", original)
    assert exp["run"](resume=True)
    assert exp["paths"]["predictions"].read_bytes().startswith(prefix)
    assert len(list(exp["output"].glob("predictions.jsonl.truncated-*.bak"))) == 1
    fresh = tmp_path / "fresh"
    assert exp["run"](output=fresh)
    assert (fresh / "predictions.jsonl").read_bytes() == exp["paths"]["predictions"].read_bytes()
    assert json.loads((fresh / "metrics.json").read_text()) == json.loads(exp["paths"]["metrics"].read_text())


@pytest.mark.parametrize("mutation", ["id", "prompt", "reference", "benchmark", "correct", "parsed", "token_ids", "prediction", "duplicate"])
def test_partial_corruption_rejected_before_weights(experiment, monkeypatch, mutation):
    exp = experiment
    interrupt_after_first(exp, monkeypatch)
    rows = read_progress_rows(exp["paths"]["predictions"])
    if mutation in {"id", "prompt", "reference", "benchmark", "prediction"}:
        rows[0][mutation] = "wrong"
    elif mutation == "correct":
        rows[0]["correct"] = int(rows[0]["correct"])
    elif mutation == "parsed":
        rows[0]["reference_answer"] = "999"
    elif mutation == "token_ids":
        rows[0]["response_ids"] = [99999]
    else:
        rows.append(rows[0])
    atomic_jsonl(exp["paths"]["predictions"], rows)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError):
        exp["run"](resume=True)


@pytest.mark.parametrize("mutation", ["dataset", "model", "runtime", "budget", "training", "development", "benchmark", "metric"])
def test_identity_change_never_mixes_results(experiment, monkeypatch, mutation):
    exp = experiment
    # Unlabeled input permits testing CLI benchmark identity independently of
    # the stricter benchmark metadata check in canonical prepared test rows.
    atomic_jsonl(exp["test_path"], [{k: v for k, v in row.items() if k != "benchmark"} for row in exp["test"]])
    interrupt_after_first(exp, monkeypatch)
    options = {}
    if mutation == "dataset":
        rows = read_progress_rows(exp["test_path"])
        rows[-1]["response"] = "#### 2"
        atomic_jsonl(exp["test_path"], rows)
    elif mutation == "model":
        path = exp["model"] / "config.json"
        atomic_json(path, {**json.loads(path.read_text()), "identity_changed": True})
    elif mutation == "runtime":
        monkeypatch.setattr(exp["module"], "package_version", lambda _: "changed")
    elif mutation in {"training", "development"}:
        path = exp["cfg"]["train_file" if mutation == "training" else "validation_file"]
        rows = read_progress_rows(path)
        atomic_jsonl(path, [{**row, "response": "#### 2"} for row in rows])
    elif mutation == "benchmark":
        options["benchmark"] = "math"
    elif mutation == "metric":
        monkeypatch.setattr(exp["module"], "metric_definition", lambda b: {"benchmark": b, "changed": True})
    else:
        options["max_new_tokens"] = 4
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="Evaluation inputs changed"):
        exp["run"](resume=True, **options)


@pytest.mark.parametrize("field", ["id", "prompt", "source_group"])
@pytest.mark.parametrize("source", ["train", "dev"])
def test_leakage_full_input_checked_before_pilot_limit(experiment, monkeypatch, field, source):
    exp = experiment
    records = [dict(row) for row in exp["test"]]
    records[-1][field] = exp[source][0][field]
    atomic_jsonl(exp["test_path"], records)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="leakage"):
        exp["run"](max_examples=1)


def test_invalid_gold_outside_limit_aborts_before_generation(experiment, monkeypatch):
    exp = experiment
    records = [dict(row) for row in exp["test"]]
    records[-1]["response"] = "invalid reference"
    atomic_jsonl(exp["test_path"], records)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="Invalid gsm8k reference.*third"):
        exp["run"](max_examples=1)
    assert not exp["paths"]["predictions"].exists()


@pytest.mark.parametrize("corruption", ["missing", "hash", "rehash_grade", "rehash_metrics"])
def test_complete_artifacts_require_hashes_and_valid_grades(experiment, monkeypatch, corruption):
    exp = experiment
    assert exp["run"]()
    paths = exp["paths"]
    if corruption == "missing":
        paths["metrics"].unlink()
    elif corruption == "hash":
        with paths["predictions"].open("a") as stream:
            stream.write("\n")
    else:
        key = "predictions" if corruption == "rehash_grade" else "metrics"
        if key == "predictions":
            rows = read_progress_rows(paths[key])
            rows[0]["correct"] = not rows[0]["correct"]
            atomic_jsonl(paths[key], rows)
        else:
            atomic_json(paths[key], {**json.loads(paths[key].read_text()), "accuracy_percent": 999})
        manifest = json.loads(paths["manifest"].read_text())
        manifest["files"][paths[key].name] = file_sha256(paths[key])
        atomic_json(paths["manifest"], manifest)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError):
        exp["run"](resume=True)


def test_publish_crash_requires_no_regeneration(experiment, monkeypatch):
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
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    assert exp["run"](resume=True)


def test_math_instructions_full_question_and_budget_are_explicit(experiment, monkeypatch):
    exp = experiment
    assert exp["common"].render_math_prompt_text("question").endswith(r"Put your final answer inside \boxed{}.")
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="increase --max-prompt-tokens"):
        exp["run"](max_prompt_tokens=5)


def test_math_run_requires_resume_and_lock(experiment):
    exp = experiment
    with exp["module"].evaluation_lock(exp["paths"]["lock"]):
        with pytest.raises(RuntimeError, match="Another evaluation"):
            exp["run"]()
    assert exp["run"]()
    with pytest.raises(FileExistsError, match="--resume"):
        exp["run"]()


def test_distillm2_pair_training_file_and_absent_teacher_are_supported(experiment):
    exp = experiment
    pairs = [{"id": row["id"], "prompt": row["prompt"], "source_group": row["source_group"],
              "prompt_ids": [2, 4, 6, 5], "chosen_ids": [12, 3], "rejected_ids": [13, 3],
              "provenance": {"teacher": "teacher", "student": "student", "tokenizer": "student"}}
             for row in exp["train"]]
    atomic_jsonl(exp["cfg"]["train_file"], pairs)
    assert exp["run"](cfg={**exp["cfg"], "method": "distillm2"})
    metrics = json.loads(exp["paths"]["metrics"].read_text())
    assert metrics["training_run"]["method"] == "distillm2"
    assert metrics["total"] == 3


def test_saved_vocabulary_policy_must_match_training_config(experiment, monkeypatch):
    exp = experiment
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="Saved vocabulary policy"):
        exp["run"](cfg={**exp["cfg"], "vocabulary_policy": "llama3_shared"})


def test_grade_aggregation_retains_unparseable_predictions_in_denominator(experiment):
    common = experiment["common"]
    rows = [{"benchmark": "math", "correct": correct, "extraction_failed": failed,
             "generated_tokens": 1} for correct, failed in [(True, False), (False, False), (False, True)]]
    metrics = common.summarize_predictions(rows, "math")
    assert metrics["accuracy_percent"] == pytest.approx(100 / 3)
    assert metrics["total"] == 3 and metrics["correct_count"] == 1
    assert metrics["unparseable_predictions"] == 1
    with pytest.raises(ValueError, match="one benchmark"):
        common.summarize_predictions([{**rows[0], "benchmark": "gsm8k"}, *rows[1:]], "math")


def test_parser_failures_and_verification_errors_reported_separately(experiment):
    rows = [{"benchmark": "math", "correct": False, "generated_tokens": 1,
             "parse_error": error, "extraction_failed": False}
            for error in ("invalid_math_expression", "verification_timeout", None)]
    metrics = experiment["common"].summarize_predictions(rows, "math")
    assert metrics["unparseable_predictions"] == 1
    assert metrics["extraction_failures"] == 0
    assert metrics["verification_errors"] == 1
    assert metrics["verification_timeouts"] == 1
    assert metrics["parsing_failures"] == 1
    assert metrics["parsing_timeouts"] == 0
    assert metrics["reference_override_count"] == 0
    assert metrics["total"] == 3 and metrics["accuracy_percent"] == 0


@pytest.mark.parametrize("benchmark", ["gsm8k", "math"])
def test_native_cli_with_real_pinned_grader(experiment, tmp_path, benchmark):
    from baseline_common.math_metrics import metric_definition
    try:
        metric_definition("gsm8k")
    except RuntimeError:
        pytest.skip("Run this integration test in the pinned .venv-math-eval environment")
    exp = experiment
    if benchmark == "math":
        atomic_jsonl(exp["test_path"], [{**row, "benchmark": "math", "response": r"Solution: \boxed{1}"}
                                      for row in exp["test"]])
    output = tmp_path / "actual-grader-cli"
    result = subprocess.run([sys.executable, str(Path(exp["module"].__file__)),
        "--config", str(exp["cfg_path"]), "--model", str(exp["model"]),
        "--data", str(exp["test_path"]), "--benchmark", benchmark, "--output", str(output),
        "--device", "cpu", "--max-prompt-tokens", "64", "--max-new-tokens", "3", "--resume"],
        text=True, capture_output=True, timeout=90, env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert result.returncode == 0, result.stderr
    rows = read_progress_rows(output / "predictions.jsonl")
    metrics = json.loads((output / "metrics.json").read_text())
    assert len(rows) == metrics["total"] == 3
    assert metrics["metric_definition"]["name"] == "math_accuracy"
    assert metrics["metric_definition"]["dependencies"]["math-verify"] == "0.9.0"
    assert metrics["accuracy_percent"] == 100 * sum(row["correct"] for row in rows) / 3
    assert all(row["reference_answer"] == "1" for row in rows)
    assert all(row["reference_override"] is None for row in rows)
    assert metrics["reference_override_count"] == 0


def test_reference_overrides_and_alternatives_persist_count_and_validate(experiment, monkeypatch):
    exp = experiment
    def grade(prediction, reference, benchmark):
        return {**fake_grade(prediction, reference, benchmark), "reference_override": "gold-annotation-hash",
                "reference_alternatives": ["1", "2"]}
    monkeypatch.setattr(exp["common"], "math_score", grade)
    interrupt_after_first(exp, monkeypatch)
    assert exp["run"](resume=True)
    metrics = json.loads(exp["paths"]["metrics"].read_text())
    assert metrics["reference_override_count"] == 3
    rows = read_progress_rows(exp["paths"]["predictions"])
    assert all(row["reference_override"] == "gold-annotation-hash" for row in rows)
    rows[0]["reference_alternatives"] = ["1"]
    atomic_jsonl(exp["paths"]["predictions"], rows)
    manifest = json.loads(exp["paths"]["manifest"].read_text())
    manifest["files"]["predictions.jsonl"] = file_sha256(exp["paths"]["predictions"])
    atomic_json(exp["paths"]["manifest"], manifest)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="reference_alternatives"):
        exp["run"](resume=True)
