"""Real Llama-3/3.2 tokenizers with tiny offline native HF checkpoints.

These tests cover the explicit shared-vocabulary policy through the public
training, pair generation, sampling, save/reload and resume entry points. The
weights are random: success demonstrates protocol execution, not model quality.
"""

import copy
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import load_records, write_jsonl
from baseline_common.models import load_model, load_tokenizer, render_prompt, verify_tokenizers


TEACHER = Path("/nas/Models/Meta-Llama-3-8B-Instruct")
STUDENT = Path("/nas/Models/Meta-Llama-3.2-1B-Instruct")
UNSUPPORTED = 128008  # Llama-3 reserved slot; Llama-3.2 end-of-message control.
SHARED_SPECIAL = {128000, 128001, 128006, 128007, 128009}
EXCLUDED = sorted(set(range(128000, 128256)) - SHARED_SPECIAL)


@pytest.fixture(scope="module", autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    for path in (TEACHER, STUDENT):
        if not (path / "tokenizer.json").is_file():
            pytest.skip("Actual local Llama-3 and Llama-3.2 tokenizers unavailable")
    root = tmp_path_factory.mktemp("llama_shared_checkpoints")
    for role, source, width, seed in [
        ("teacher", TEACHER, 16, 137), ("student", STUDENT, 8, 249),
    ]:
        tokenizer = load_tokenizer(source)
        torch.manual_seed(seed)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=128256, hidden_size=width, intermediate_size=width * 2,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=width // 2, max_position_embeddings=128,
            bos_token_id=128000, eos_token_id=[128001, 128008, 128009],
            pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=False,
        ))
        model.save_pretrained(root / role)
        tokenizer.save_pretrained(root / role)
    return root


@pytest.fixture
def experiment(tmp_path, checkpoints):
    rows = [
        {"id": "train-a", "prompt": "Write a number.", "response": "One."},
        {"id": "train-b", "prompt": "Name a color.", "response": "Blue."},
    ]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "validation.jsonl", [
        {"id": "validation-a", "prompt": "Name an animal.", "response": "Cat."},
    ])
    return validate_config(dict(
        name="llama_shared_fixture", pair="llama3_llama32", method="kd", dataset="fixture",
        teacher_model=str(checkpoints / "teacher"), student_model=str(checkpoints / "student"),
        train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "validation.jsonl"),
        output_root=str(tmp_path / "results"), vocabulary_policy="llama3_shared",
        device="cpu", teacher_device="cpu", dtype="float32", optimizer_offload=True,
        max_prompt_tokens=64, max_new_tokens=3, learning_rate=0.001,
        gradient_accumulation_steps=1, max_steps=1, gradient_checkpointing=True,
        save_steps=1, eval_steps=1, loss_chunk_size=2, acceptance_k=2, proposal_block_size=2,
        eval_during_training=True, eval_max_new_tokens=3,
        generation=dict(temperature=0.7, top_p=1.0, top_k=2),
        teacher_generation=dict(temperature=0.7, top_p=1.0, top_k=2),
    ))


def _tokenizer(cfg):
    return verify_tokenizers(cfg["teacher_model"], cfg["student_model"], policy="llama3_shared")


def _models(cfg, tokenizer):
    from baseline_common.models import bind_vocabulary

    student = load_model(cfg["student_model"], "float32", "cpu")
    teacher = load_model(cfg["teacher_model"], "float32", "cpu")
    bind_vocabulary(student, tokenizer)
    bind_vocabulary(teacher, tokenizer)
    teacher.eval().requires_grad_(False)
    return student, teacher


def _state(path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, local_files_only=True).state_dict()


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _generate_pairs(cfg, monkeypatch):
    from scripts.generate_pairs import main

    cfg = copy.deepcopy(cfg)
    input_file = cfg["train_file"]
    cfg["method"] = "distillm2"
    cfg["train_file"] = str(Path(input_file).with_name("pairs.jsonl"))
    config_file = Path(input_file).with_name("pair_config.json")
    config_file.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", [
        "generate_pairs.py", "--config", str(config_file), "--input", input_file,
        "--output", cfg["train_file"], "--device", "cpu",
    ])
    main()
    return cfg


@pytest.mark.parametrize("method", ["kd", "abkd", "skd", "distillm2"])
def test_actual_tokenizers_one_update_all_methods_and_policy_export(experiment, method, monkeypatch):
    from transformers import AutoConfig, GenerationConfig

    training = importlib.import_module("baseline_common.train")
    cfg = _generate_pairs(experiment, monkeypatch) if method == "distillm2" else {**experiment, "method": method}
    initial = _state(cfg["student_model"])
    tokenizer = _tokenizer(cfg)
    expected_alignment = tokenizer._baseline_vocab_alignment
    loaded_teachers = []
    original_loader = training.load_model

    def observe_loader(path, *args, **kwargs):
        model = original_loader(path, *args, **kwargs)
        if Path(path) == Path(cfg["teacher_model"]):
            loaded_teachers.append(model)

            def assert_frozen(module, _inputs):
                assert not module.training and not torch.is_grad_enabled()
                assert all(not p.requires_grad and p.grad is None for p in module.parameters())

            model.register_forward_pre_hook(assert_frozen)
        return model

    monkeypatch.setattr(training, "load_model", observe_loader)
    output = training.train(cfg)
    metrics = _lines(output / "metrics.jsonl")
    assert len(metrics) == 1 and metrics[0]["step"] == 1
    assert torch.isfinite(torch.tensor([metrics[0]["loss"], metrics[0]["grad_norm"]])).all()
    assert metrics[0]["grad_norm"] > 0
    result = json.loads((output / "result.json").read_text())
    assert result["complete"] and result["step"] == 1
    if method == "skd":
        assert result["counters"]["skd_generated_tokens"] > 0
    assert loaded_teachers and all(p.grad is None for model in loaded_teachers for p in model.parameters())
    trained = _state(output / "final")
    assert any(not torch.equal(initial[key], trained[key]) for key in initial)
    assert all(torch.isfinite(value).all() for value in trained.values())
    saved = AutoConfig.from_pretrained(output / "final", local_files_only=True)
    generation = GenerationConfig.from_pretrained(output / "final", local_files_only=True)
    assert saved.baseline_vocab_alignment == expected_alignment
    assert set(EXCLUDED) <= set(generation.suppress_tokens)
    assert set(generation.eos_token_id) == {128001, 128009}
    restored_tokenizer = load_tokenizer(output / "final")
    assert restored_tokenizer._baseline_vocab_alignment == expected_alignment
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["vocabulary_policy"] == "llama3_shared"
    assert manifest["vocabulary_alignment"] == expected_alignment
    assert (output / "checkpoints/step_000001/complete.json").is_file()


@pytest.mark.parametrize("method", ["kd", "abkd", "skd", "skew_forward", "skew_reverse"])
def test_excluded_logits_do_not_change_loss_or_gradients(experiment, method):
    from baseline_common.models import encode_reference
    from baseline_common.train import response_loss

    tokenizer = _tokenizer(experiment)
    student, teacher = _models(experiment, tokenizer)
    student.eval()
    record = encode_reference(tokenizer, "Write a number.", "One.", 64, 3)
    counters = {"teacher_training_tokens": 0, "response_tokens": 0}
    baseline = response_loss(student, teacher, record, experiment, method, "sum", counters)
    baseline.backward()
    expected_gradients = {name: value.grad.clone() for name, value in student.named_parameters()}
    assert student.get_output_embeddings().weight.grad[EXCLUDED].count_nonzero() == 0
    student.zero_grad(set_to_none=True)

    def corrupt_excluded(sign):
        def hook(_module, _inputs, output):
            logits = output.logits.clone()
            logits[..., EXCLUDED] = sign * 1e6
            output.logits = logits
            return output
        return hook

    teacher_handle = teacher.register_forward_hook(corrupt_excluded(-1))
    student_handle = student.register_forward_hook(corrupt_excluded(1))
    try:
        polluted = response_loss(student, teacher, record, experiment, method, "sum", counters)
        polluted.backward()
    finally:
        teacher_handle.remove()
        student_handle.remove()
    torch.testing.assert_close(polluted, baseline, rtol=0, atol=0)
    for name, parameter in student.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, expected_gradients[name], rtol=0, atol=0)


def _dominant_unsupported(allowed):
    def hook(_module, _inputs, output):
        output.logits = torch.full_like(output.logits, -1000)
        output.logits[..., allowed] = 100
        output.logits[..., UNSUPPORTED] = 1000
        return output
    return hook


@pytest.mark.parametrize("teacher_token", [100, 101])
def test_native_and_skd_generation_mask_dominant_unsupported_id(experiment, teacher_token, tmp_path):
    from transformers import AutoModelForCausalLM
    from baseline_common.models import stop_ids
    from baseline_common.sampling import SamplingConfig, skd_generate

    tokenizer = _tokenizer(experiment)
    student, teacher = _models(experiment, tokenizer)
    student.register_forward_hook(_dominant_unsupported(100))
    teacher.register_forward_hook(_dominant_unsupported(teacher_token))
    prompt = torch.tensor([render_prompt(tokenizer, "Continue.", 64)])
    generated = student.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                 max_new_tokens=3, do_sample=False)
    assert generated[0, prompt.shape[1]:].tolist() == [100] * 3
    result, stats = skd_generate(
        student, teacher, prompt, max_new_tokens=3, eos_token_ids=stop_ids(teacher, tokenizer),
        proposal_config=SamplingConfig(top_k=1, top_p=1), teacher_config=SamplingConfig(top_k=1, top_p=1),
        acceptance_k=1, block_size=2, return_stats=True,
    )
    assert result[0, prompt.shape[1]:].tolist() == [teacher_token] * 3
    assert stats["teacher_interventions"] == (0 if teacher_token == 100 else 3)
    exported = tmp_path / "exported"
    student.save_pretrained(exported)
    reloaded = AutoModelForCausalLM.from_pretrained(exported, local_files_only=True)
    assert reloaded.config.baseline_vocab_alignment == tokenizer._baseline_vocab_alignment
    reloaded.register_forward_hook(_dominant_unsupported(100))
    # Vanilla HF generation must retain suppression without a custom loader.
    restored = reloaded.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                                max_new_tokens=3, do_sample=False)
    assert restored[0, prompt.shape[1]:].tolist() == [100] * 3


def test_rouge_eval_ignores_excluded_logits_and_rejects_unsupported_training_targets(experiment):
    from baseline_common.models import encode_reference
    from baseline_common.train import evaluate, response_loss, sample_record

    tokenizer = _tokenizer(experiment)
    student, teacher = _models(experiment, tokenizer)
    rows = load_records(experiment["validation_file"])
    baseline = evaluate(student, tokenizer, rows, experiment)
    assert "rouge_l" in baseline and "reference_nll" not in baseline

    def corrupt(_module, _inputs, output):
        output.logits[..., EXCLUDED] = 1e6
        return output

    handle = student.register_forward_hook(corrupt)
    try:
        assert evaluate(student, tokenizer, rows, experiment) == baseline
    finally:
        handle.remove()
    with pytest.raises(ValueError):
        encode_reference(tokenizer, "Continue.", "<|eom_id|>", 64, 3)
    prompt = render_prompt(tokenizer, "Continue.", 64)
    paired = {"id": "bad", "prompt": "Continue.", "prompt_ids": prompt,
              "chosen_ids": [UNSUPPORTED], "rejected_ids": [100]}
    with pytest.raises(ValueError):
        sample_record(paired, {**experiment, "method": "distillm2"}, student, teacher, tokenizer, {})
    record = {"input_ids": prompt + [UNSUPPORTED], "labels": [-100] * len(prompt) + [UNSUPPORTED]}
    with pytest.raises(ValueError):
        response_loss(student, teacher, record, experiment, "kd", "sum",
                      {"teacher_training_tokens": 0, "response_tokens": 0})
    # Target validation must inspect labels themselves, even if input IDs are
    # valid (the KL objectives use target positions rather than label values).
    record["input_ids"][-1] = 100
    with pytest.raises(ValueError):
        response_loss(student, teacher, record, experiment, "kd", "sum",
                      {"teacher_training_tokens": 0, "response_tokens": 0})


def test_pair_provenance_rejects_wrong_or_missing_policy(experiment, monkeypatch):
    from baseline_common.train import check_data

    cfg = _generate_pairs(experiment, monkeypatch)
    pairs = load_records(cfg["train_file"], paired=True)
    validation = load_records(cfg["validation_file"])
    check_data(cfg, pairs, validation)
    assert all(row["provenance"]["vocabulary_policy"] == "llama3_shared" for row in pairs)
    expected_alignment = _tokenizer(cfg)._baseline_vocab_alignment
    assert all(row["provenance"]["vocabulary_alignment"] == expected_alignment for row in pairs)
    from baseline_common.train import sample_record
    altered_row = copy.deepcopy(pairs[0])
    altered_row["provenance"]["vocabulary_alignment"]["excluded_ids"].pop()
    with pytest.raises(ValueError, match="vocabulary alignment differs"):
        sample_record(altered_row, cfg, None, None, _tokenizer(cfg), {})
    for replacement in (None, "full"):
        altered = copy.deepcopy(pairs)
        for row in altered:
            if replacement is None:
                row["provenance"].pop("vocabulary_policy")
            else:
                row["provenance"]["vocabulary_policy"] = replacement
        with pytest.raises(ValueError, match="vocab|policy|protocol"):
            check_data(cfg, altered, validation)


def test_shared_policy_exact_resume_and_saved_checkpoint(experiment):
    from transformers import AutoConfig
    from baseline_common.train import train

    cfg = {**experiment, "max_steps": 2}
    uninterrupted = train(cfg)
    resumed_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]).with_name("resumed"))}
    paused = train(resumed_cfg, stop_after_steps=1)
    checkpoint = paused / "checkpoints/step_000001"
    alignment = AutoConfig.from_pretrained(checkpoint / "student", local_files_only=True).baseline_vocab_alignment
    assert alignment == _tokenizer(cfg)._baseline_vocab_alignment
    torch.manual_seed(982347)
    resumed = train(resumed_cfg, resume=str(checkpoint))
    expected, actual = _state(uninterrupted / "final"), _state(resumed / "final")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert _lines(resumed / "metrics.jsonl") == _lines(uninterrupted / "metrics.jsonl")
    assert _lines(resumed / "validation.jsonl") == _lines(uninterrupted / "validation.jsonl")
    assert json.loads((resumed / "result.json").read_text())["complete"]


def test_shared_chat_date_and_prompt_ids_survive_tokenizer_export(experiment, tmp_path, monkeypatch):
    from baseline_common.models import load_tokenizer

    tokenizer = _tokenizer(experiment)
    assert tokenizer._baseline_vocab_alignment["chat_date_string"] == "26 Jul 2024"
    # Exercise both an ordinary prompt and the repeated rendering used during
    # prompt truncation; each render must receive the same explicit date.
    original = tokenizer.apply_chat_template
    passed_dates = []

    def observe_date(*args, **kwargs):
        passed_dates.append(kwargs.get("date_string"))
        return original(*args, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", observe_date)
    prompts = ["Continue.", "A long prompt " * 100]
    prefixes = [render_prompt(tokenizer, text, 64) for text in prompts]
    assert len(passed_dates) > len(prompts)
    assert set(passed_dates) == {"26 Jul 2024"}
    for prefix in prefixes:
        assert "Today Date: 26 Jul 2024" in tokenizer.decode(prefix)
    exported = tmp_path / "dated_tokenizer"
    tokenizer.save_pretrained(exported)
    restored = load_tokenizer(exported)
    assert restored._baseline_vocab_alignment == tokenizer._baseline_vocab_alignment
    assert [render_prompt(restored, text, 64) for text in prompts] == prefixes
