"""Protocol validation and actual local Qwen tokenizer regression checks."""
import copy
import json
from pathlib import Path

import pytest

from baseline_common.config import validate_config
from baseline_common.models import encode_reference, load_tokenizer, render_prompt, verify_tokenizers


def config(**changes):
    values = dict(name="test", pair="qwen", method="kd", dataset="fixture",
                  teacher_model="/models/teacher", student_model="/models/student",
                  train_file="/data/train.jsonl", validation_file="/data/validation.jsonl")
    values.update(changes)
    return values


@pytest.mark.parametrize("changes", [
    {"typo": 1}, {"method": "gkd"}, {"dtype": "float16"},
    {"name": "../run"}, {"pair": ""}, {"dataset": ".."},
    {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"seed": 1.5},
    {"optimizer_offload": "false"}, {"gradient_checkpointing": 0},
    {"enable_thinking": True}, {"enable_thinking": 0},
    {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"weight_decay": -1}, {"max_grad_norm": float("inf")},
    {"warmup_ratio": 1}, {"alpha": float("inf")}, {"beta": float("nan")},
    {"alpha_1": 0}, {"alpha_2": 1},
    {"max_prompt_tokens": 0}, {"max_new_tokens": True},
    {"gradient_accumulation_steps": 0}, {"max_steps": 0},
    {"teacher_model": ""}, {"validation_file": "/data/./train.jsonl"},
    {"generation": None}, {"generation": {"temperature": 0.7}},
    {"generation": {"temperature": float("nan"), "top_p": 1, "top_k": 0}},
    {"generation": {"temperature": 1, "top_p": 0, "top_k": 0}},
    {"generation": {"temperature": 1, "top_p": 1, "top_k": True}},
    {"teacher_generation": {"temperature": -1, "top_p": 1, "top_k": 0}},
    {"method": "skd", "generation": {"temperature": 0, "top_p": 1, "top_k": 0}},
])
def test_config_rejects_invalid_protocol_values(changes):
    with pytest.raises(ValueError):
        validate_config(config(**changes))


def test_config_requires_fields_without_mutating_input_or_shared_defaults():
    source = config()
    before = copy.deepcopy(source)
    resolved = validate_config(source)
    assert source == before
    resolved["generation"]["top_k"] = 999
    assert validate_config(source)["generation"]["top_k"] == 20
    del source["teacher_model"]
    with pytest.raises(ValueError, match="Missing"):
        validate_config(source)


def test_tokenizer_alignment_rejects_same_size_relabelled_vocabulary(monkeypatch):
    class FakeTokenizer:
        def __init__(self, vocabulary):
            self.vocabulary = vocabulary
        def get_vocab(self):
            return self.vocabulary
    tokenizers = {"teacher": FakeTokenizer({"a": 0, "b": 1}),
                  "student": FakeTokenizer({"a": 1, "b": 0})}
    monkeypatch.setattr("baseline_common.models.load_tokenizer", tokenizers.__getitem__)
    with pytest.raises(ValueError, match="full token-to-ID maps differ"):
        verify_tokenizers("teacher", "student")


def test_tokenizer_alignment_rejects_same_vocab_different_normalization(monkeypatch):
    class Backend:
        def __init__(self, normalization):
            self.normalization = normalization
        def to_str(self):
            return json.dumps({"model": "same", "normalizer": self.normalization})
    class FakeTokenizer:
        def __init__(self, normalization):
            self.backend_tokenizer = Backend(normalization)
        def get_vocab(self):
            return {"a": 0, "b": 1}
    tokenizers = {"teacher": FakeTokenizer("lowercase"), "student": FakeTokenizer(None)}
    monkeypatch.setattr("baseline_common.models.load_tokenizer", tokenizers.__getitem__)
    with pytest.raises(ValueError, match="normalizer differs"):
        verify_tokenizers("teacher", "student")


@pytest.fixture(scope="module")
def actual_qwen_tokenizer():
    path = Path("/nas/Models/Qwen3-1.7B-Instruct")
    if not (path / "tokenizer.json").exists():
        pytest.skip("Actual local Qwen3 tokenizer unavailable")
    return load_tokenizer(path)


@pytest.mark.parametrize("prompt,budget", [("Calculate 2 + 2.", 128), ("Long question " * 400, 48)])
def test_actual_qwen_nonthinking_prefix_survives_truncation(actual_qwen_tokenizer, prompt, budget):
    tokenizer = actual_qwen_tokenizer
    ids = render_prompt(tokenizer, prompt, budget, enable_thinking=False)
    rendered = tokenizer.decode(ids, skip_special_tokens=False)
    assert len(ids) <= budget
    assert rendered.startswith("<|im_start|>user\n")
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # <think> and </think> are part of the saved prompt, never synthesized by
    # decoding/re-encoding the generated assistant completion.
    assert tokenizer.convert_tokens_to_ids("<think>") in ids
    assert tokenizer.convert_tokens_to_ids("</think>") in ids


def test_actual_qwen_preserves_supervised_eos_when_padding_shares_id(actual_qwen_tokenizer, monkeypatch):
    tokenizer = actual_qwen_tokenizer
    monkeypatch.setattr(tokenizer, "pad_token", tokenizer.eos_token)
    encoded = encode_reference(tokenizer, "Calculate 2 + 2.", "4", 64, 8, False)
    prefix = render_prompt(tokenizer, "Calculate 2 + 2.", 64, False)
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert encoded["input_ids"][:len(prefix)] == prefix
    assert encoded["labels"][:len(prefix)] == [-100] * len(prefix)
    assert encoded["input_ids"][-1] == encoded["labels"][-1] == tokenizer.eos_token_id
    # A prompt EOS is masked by boundary; the same numerical ID at the real
    # response end remains a target, even after conventional batch padding.
    assert tokenizer.eos_token_id in prefix
    padded_ids = encoded["input_ids"] + [tokenizer.pad_token_id] * 2
    padded_labels = encoded["labels"] + [-100] * 2
    assert padded_ids[-3:] == [tokenizer.eos_token_id] * 3
    assert padded_labels[-3:] == [tokenizer.eos_token_id, -100, -100]


def test_actual_qwen_budget_below_template_is_rejected(actual_qwen_tokenizer):
    with pytest.raises(ValueError, match="too small"):
        render_prompt(actual_qwen_tokenizer, "Question", 1, False)


def test_training_rejects_metamath_augmentation_leakage():
    from baseline_common.train import check_data
    train = [{"id": "a", "prompt": "original phrasing", "response": "a", "source_group": "same-original"}]
    validation = [{"id": "b", "prompt": "different paraphrase", "response": "b", "source_group": "same-original"}]
    with pytest.raises(ValueError, match="source_group"):
        check_data(validate_config(config()), train, validation)
