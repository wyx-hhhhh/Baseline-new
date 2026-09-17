"""Real tiny CPU generation and durable multi-reference evaluation contracts."""
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
             "training", "development", "first", "second", "third", "heldout", "answer", "other"]
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
    train = [{"id": "train", "prompt": "training", "response": "answer", "source_group": "training"}]
    test = [{"id": word, "prompt": word + " heldout", "response": "first answer",
             "references": ["first answer", "second answer"], "source_group": word + "-group",
             "source_rows": [{"path": "original.jsonl", "line": index + 1, "subset": "."}],
             "topic": "test", "benchmark": "dolly"}
            for index, word in enumerate(("first", "second", "third"))]
    write_jsonl(tmp_path / "train.jsonl", train)
    write_jsonl(tmp_path / "test.jsonl", test)
    cfg = validate_config(dict(name="tiny_nl_kd", pair="tiny", method="kd", dataset="dolly",
        teacher_model=str(tmp_path / "missing-teacher"), student_model=str(tmp_path / "missing-student"),
        train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "absent-development.jsonl"),
        dtype="float32", device="cpu", max_prompt_tokens=64, max_new_tokens=3,
        eval_max_new_tokens=1, eval_max_examples=1))
    cfg_path = tmp_path / "config.json"
    atomic_json(cfg_path, cfg)
    common = importlib.import_module("baseline_common.nl_evaluation")
    module = importlib.import_module("scripts.evaluate_nl")
    output = tmp_path / "evaluation"
    def run(**kwargs):
        options = {"data_path": tmp_path / "test.jsonl", "benchmark": "dolly",
                   "max_prompt_tokens": 64, "max_new_tokens": 3}
        options.update(kwargs)
        return module.run_evaluation(options.pop("cfg", cfg), model_path=model_path,
                                     output=options.pop("output", output), **options)
    return dict(cfg=cfg, cfg_path=cfg_path, module=module, common=common,
                model=model_path, tokenizer=tokenizer, output=output, train=train,
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


def test_native_generation_all_examples_full_references_and_no_load_cache(experiment, monkeypatch):
    exp = experiment
    assert exp["run"]()
    rows = read_progress_rows(exp["paths"]["predictions"])
    metrics = json.loads(exp["paths"]["metrics"].read_text())
    assert len(rows) == metrics["examples"] == 3
    assert metrics["references"] == 6 and metrics["multiple_reference_examples"] == 3
    assert metrics["rouge_l"] == pytest.approx(sum(row["rouge_l"] for row in rows) / 3)
    assert all(row["rouge_l"] == max(row["reference_rouge_l"]) for row in rows)
    assert all(row["references"] == ["first answer", "second answer"] for row in rows)
    assert metrics["evaluation"]["decoding"] == "greedy"
    assert metrics["evaluation"]["prompt_truncation"] == "forbidden"
    assert metrics["metric_definition"]["truncate_references"] is False
    assert metrics["metric_definition"]["multiple_references"] == "maximum_fmeasure_over_all_references_per_example"
    before = {key: (file_sha256(path), path.stat().st_mtime_ns)
              for key, path in exp["paths"].items() if key in {"metrics", "manifest", "predictions"}}
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    assert exp["run"](resume=True)
    assert before == {key: (file_sha256(exp["paths"][key]), exp["paths"][key].stat().st_mtime_ns) for key in before}


def test_best_full_reference_then_macro_aggregation(experiment, monkeypatch):
    exp = experiment
    # The generation budget is only two tokens; the first full reference is
    # much longer. A matching second reference must still receive full credit.
    rows = [{**row, "response": "first " * 1000 + "answer",
             "references": ["first " * 1000 + "answer", "second answer"]} for row in exp["test"]]
    atomic_jsonl(exp["test_path"], rows)
    second_answer = exp["tokenizer"].encode("second answer", add_special_tokens=False)
    calls = []
    def generate(model, tokenizer, prompt_ids, max_new_tokens, generation):
        calls.append((list(prompt_ids), max_new_tokens, generation))
        return second_answer
    monkeypatch.setattr(exp["common"], "generate_response", generate)
    assert exp["run"](max_new_tokens=2)
    predictions = read_progress_rows(exp["paths"]["predictions"])
    assert all(row["rouge_l"] == 100 and row["best_reference_index"] == 1 for row in predictions)
    assert all(row["reference_rouge_l"][0] < 1 for row in predictions)
    assert len(calls) == 3 and all(call[2]["temperature"] == 0 for call in calls)
    mixed = [dict(row) for row in predictions]
    mixed[0].update(rouge_l=0, references=["wrong"])
    summary = exp["common"].summarize_predictions(mixed, "dolly")
    assert summary["rouge_l"] == pytest.approx(200 / 3)


def test_signal_torn_tail_resume_matches_uninterrupted(experiment, monkeypatch, tmp_path):
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


@pytest.mark.parametrize("mutation", ["id", "prompt", "reference", "references", "benchmark", "score",
                                    "reference_score", "best_index", "token_ids", "prediction", "duplicate",
                                    "prompt_ids", "generated_tokens", "source_rows", "topic", "extra_field"])
def test_partial_corruption_rejected_before_weights(experiment, monkeypatch, mutation):
    exp = experiment
    interrupt_after_first(exp, monkeypatch)
    rows = read_progress_rows(exp["paths"]["predictions"])
    if mutation in {"id", "prompt", "reference", "benchmark", "prediction", "topic"}:
        rows[0][mutation] = "wrong"
    elif mutation == "references":
        rows[0]["references"] = ["first answer"]
    elif mutation == "score":
        rows[0]["rouge_l"] = float("nan")
    elif mutation == "reference_score":
        rows[0]["reference_rouge_l"][0] = True
    elif mutation == "best_index":
        rows[0]["best_reference_index"] = True
    elif mutation == "token_ids":
        rows[0]["response_ids"] = [99999]
    elif mutation == "prompt_ids":
        rows[0]["prompt_ids"][0] = 1
    elif mutation == "generated_tokens":
        rows[0]["generated_tokens"] += 1
    elif mutation == "source_rows":
        rows[0]["source_rows"][0]["line"] = 999
    elif mutation == "extra_field":
        rows[0]["unexpected"] = True
    else:
        rows.append(rows[0])
    atomic_jsonl(exp["paths"]["predictions"], rows)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError):
        exp["run"](resume=True)


@pytest.mark.parametrize("mutation", ["dataset", "model", "runtime", "budget", "training", "metric", "config"])
def test_identity_changes_rejected_before_loading(experiment, monkeypatch, mutation):
    exp = experiment
    interrupt_after_first(exp, monkeypatch)
    options = {}
    if mutation == "dataset":
        rows = read_progress_rows(exp["test_path"])
        rows[-1]["references"].append("third answer")
        atomic_jsonl(exp["test_path"], rows)
    elif mutation == "model":
        path = exp["model"] / "config.json"
        atomic_json(path, {**json.loads(path.read_text()), "identity_changed": True})
    elif mutation == "runtime":
        monkeypatch.setattr(exp["module"], "_runtime_identity", lambda _: {"changed": True})
    elif mutation == "training":
        path = exp["cfg"]["train_file"]
        atomic_jsonl(path, [{**row, "response": "third answer"} for row in exp["train"]])
    elif mutation == "metric":
        original = exp["module"].metric_definition
        monkeypatch.setattr(exp["module"], "metric_definition", lambda tokenizer: {**original(tokenizer), "changed": True})
    elif mutation == "config":
        options["cfg"] = {**exp["cfg"], "seed": 999}
    else:
        options["max_new_tokens"] = 4
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="Evaluation inputs changed"):
        exp["run"](resume=True, **options)


@pytest.mark.parametrize("field", ["id", "prompt", "source_group"])
def test_complete_input_training_leakage_checked_before_limit(experiment, monkeypatch, field):
    exp = experiment
    records = [dict(row) for row in exp["test"]]
    records[-1][field] = exp["train"][0][field]
    atomic_jsonl(exp["test_path"], records)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="leakage"):
        exp["run"](max_examples=1)


@pytest.mark.parametrize("mutation", ["references", "response", "benchmark", "reference_language"])
def test_full_input_references_validated_before_limit(experiment, monkeypatch, mutation):
    exp = experiment
    records = [dict(row) for row in exp["test"]]
    if mutation == "references":
        records[-1]["references"] = []
    elif mutation == "reference_language":
        records[-1]["references"] = ["first answer", "你好"]
    else:
        records[-1][mutation] = "wrong"
    atomic_jsonl(exp["test_path"], records)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError):
        exp["run"](max_examples=1)


def test_complete_prompt_and_context_checked_before_limit_and_weights(experiment, monkeypatch):
    exp = experiment
    text = "first " * 80
    records = [dict(row) for row in exp["test"]]
    records[-1]["prompt"] = text
    atomic_jsonl(exp["test_path"], records)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="increase --max-prompt-tokens"):
        exp["run"](max_examples=1, max_prompt_tokens=16)
    ids = exp["common"].nl_prompt(exp["tokenizer"], text, 128)
    assert len(ids) == 83
    assert ids[2:-1] == exp["tokenizer"].encode(text, add_special_tokens=False)
    with pytest.raises(ValueError, match="exceeds model context"):
        exp["run"](max_examples=1, max_prompt_tokens=128, max_new_tokens=64)


@pytest.mark.parametrize("corruption", ["missing", "hash", "rehash_score", "rehash_metrics"])
def test_completed_artifacts_require_hashes_and_valid_scores(experiment, monkeypatch, corruption):
    exp = experiment
    assert exp["run"]()
    paths = exp["paths"]
    if corruption == "missing":
        paths["metrics"].unlink()
    elif corruption == "hash":
        with paths["predictions"].open("a") as stream:
            stream.write("\n")
    else:
        key = "predictions" if corruption == "rehash_score" else "metrics"
        if key == "predictions":
            rows = read_progress_rows(paths[key])
            rows[0]["rouge_l"] += 1
            atomic_jsonl(paths[key], rows)
        else:
            atomic_json(paths[key], {**json.loads(paths[key].read_text()), "rouge_l": 999})
        manifest = json.loads(paths["manifest"].read_text())
        manifest["files"][paths[key].name] = file_sha256(paths[key])
        atomic_json(paths["manifest"], manifest)
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError):
        exp["run"](resume=True)


def test_publish_crash_recovers_without_regeneration(experiment, monkeypatch):
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


def test_resume_flag_and_output_lock_are_required(experiment):
    exp = experiment
    with exp["module"].evaluation_lock(exp["paths"]["lock"]):
        with pytest.raises(RuntimeError, match="Another evaluation"):
            exp["run"]()
    assert exp["run"]()
    with pytest.raises(FileExistsError, match="--resume"):
        exp["run"]()


def test_distillm2_pairs_and_absent_teacher_or_development_are_supported(experiment):
    exp = experiment
    pairs = [{"id": row["id"], "prompt": row["prompt"], "source_group": row["source_group"],
              "prompt_ids": [2, 4, 6, 5], "chosen_ids": [12, 3], "rejected_ids": [13, 3],
              "provenance": {"teacher": "teacher", "student": "student", "tokenizer": "student"}}
             for row in exp["train"]]
    atomic_jsonl(exp["cfg"]["train_file"], pairs)
    assert exp["run"](cfg={**exp["cfg"], "method": "distillm2"})
    metrics = json.loads(exp["paths"]["metrics"].read_text())
    assert metrics["training_run"]["method"] == "distillm2" and metrics["examples"] == 3


def test_vocabulary_policy_must_match_saved_training_policy(experiment, monkeypatch):
    monkeypatch.setattr(experiment["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="Saved vocabulary policy"):
        experiment["run"](cfg={**experiment["cfg"], "vocabulary_policy": "llama3_shared"})


def test_unicode_reference_scoring_is_explicit(experiment):
    common = experiment["common"]
    with pytest.raises(ValueError, match="no English ROUGE tokens"):
        common.reference_scores("你好", ["你好"])
    assert common.reference_scores("你好", ["其他", "你好"], "unicode")["rouge_l"] == 100
    assert common.reference_scores("", ["first answer"])["rouge_l"] == 0


def test_full_unicode_input_and_symbol_only_references_remain_in_denominator(experiment, monkeypatch):
    exp = experiment
    references = [["first answer"], ["😀"], ["你好", "其他"]]
    atomic_jsonl(exp["test_path"], [{**row, "references": refs, "response": refs[0]}
                                   for row, refs in zip(exp["test"], references)])
    monkeypatch.setattr(exp["common"], "generate_response", lambda *args: [])
    assert exp["run"](rouge_tokenizer="unicode")
    metrics = json.loads(exp["paths"]["metrics"].read_text())
    assert metrics["examples"] == 3 and metrics["references"] == 4
    assert metrics["rouge_l"] == 0
    assert metrics["metric_definition"]["tokenizer"] == "unicode"
    assert exp["common"].reference_scores("😀", ["😀"], "unicode")["rouge_l"] == 0


def prepare_source_pool(exp, tmp_path):
    from baseline_common.nl_data import prepare_nl_pool
    source_root = tmp_path / "local-sources"
    source_path = source_root / "DollyEval/valid.jsonl"
    source_path.parent.mkdir(parents=True)
    write_jsonl(source_path, [{"instruction": row["prompt"], "output": row["references"], "topic": row["topic"]}
                              for row in exp["test"]])
    pool = tmp_path / "prepared-pool"
    prepare_nl_pool(source_root, pool, ["dolly"], exclude_files=[exp["cfg"]["train_file"]])
    return pool, source_path


def test_prepared_source_manifest_integration_and_explicit_path_binding(experiment, tmp_path, monkeypatch):
    exp = experiment
    pool, _ = prepare_source_pool(exp, tmp_path)
    options = {"data_path": pool / "dolly.jsonl", "data_manifest": pool / "manifest.json"}
    assert exp["run"](**options)
    identity = json.loads(exp["paths"]["manifest"].read_text())["identity"]
    provenance = identity["dataset"]["source_manifest"]
    assert provenance["sha256"] == file_sha256(pool / "manifest.json")
    assert provenance["benchmark"]["records"] == 3
    assert len(provenance["source_files"]) == len(provenance["exclusion_files"]) == 1
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    assert exp["run"](data_path=pool / "dolly.jsonl", resume=True)
    # A valid manifest for one path cannot attest to a separate JSONL copy.
    with pytest.raises(ValueError, match="path disagrees"):
        exp["run"](data_manifest=pool / "manifest.json", output=tmp_path / "wrong-path")


@pytest.mark.parametrize("mutation", ["source", "exclusion", "manifest"])
def test_original_source_and_exclusion_changes_rejected_on_resume(experiment, tmp_path, monkeypatch, mutation):
    exp = experiment
    pool, source_path = prepare_source_pool(exp, tmp_path)
    options = {"data_path": pool / "dolly.jsonl", "data_manifest": pool / "manifest.json"}
    assert exp["run"](**options)
    target = {"source": source_path, "exclusion": Path(exp["cfg"]["train_file"]),
              "manifest": pool / "manifest.json"}[mutation]
    target.write_text(target.read_text() + "\n")
    monkeypatch.setattr(exp["module"], "load_model", forbid_load)
    with pytest.raises(ValueError, match="changed"):
        exp["run"](**options, resume=True)


def test_source_changed_during_generation_never_publishes_complete_metrics(experiment, tmp_path, monkeypatch):
    exp = experiment
    pool, source_path = prepare_source_pool(exp, tmp_path)
    original = exp["module"].append_jsonl
    def alter_source(path, row):
        original(path, row)
        source_path.write_text(source_path.read_text() + "\n")
    monkeypatch.setattr(exp["module"], "append_jsonl", alter_source)
    with pytest.raises(ValueError, match="source changed"):
        exp["run"](data_path=pool / "dolly.jsonl", data_manifest=pool / "manifest.json")
    assert not exp["paths"]["metrics"].exists() and not exp["paths"]["manifest"].exists()


@pytest.mark.parametrize("benchmark", ["dolly", "selfinst", "super_natural", "unnatural", "vicuna"])
def test_native_cli_real_rouge_and_resume(experiment, tmp_path, benchmark):
    exp = experiment
    atomic_jsonl(exp["test_path"], [{**row, "benchmark": benchmark} for row in exp["test"]])
    output = tmp_path / "native-cli"
    command = [sys.executable, str(Path(exp["module"].__file__)),
        "--config", str(exp["cfg_path"]), "--model", str(exp["model"]),
        "--data", str(exp["test_path"]), "--benchmark", benchmark, "--output", str(output),
        "--device", "cpu", "--max-prompt-tokens", "64", "--max-new-tokens", "3", "--resume"]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    result = subprocess.run(command, text=True, capture_output=True, timeout=90, env=environment)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["examples"] == 3 and metrics["benchmark"] == benchmark
    assert metrics["metric_definition"]["implementation"] == "rouge-score"
    assert 0 <= metrics["rouge_l"] <= 100
    cached = subprocess.run(command, text=True, capture_output=True, timeout=90, env=environment)
    assert cached.returncode == 0, cached.stderr
    assert "already complete" in cached.stdout and "Loading student" not in cached.stdout
