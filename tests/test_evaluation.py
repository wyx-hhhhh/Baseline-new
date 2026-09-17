"""Generated-text metric oracles, protocol boundaries and interruption safety."""

import copy
import importlib.metadata
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baseline_common.evaluation import evaluate_records, metric_definition, rouge_l_score


@pytest.mark.parametrize("prediction,reference,expected", [
    ("alpha gamma beta", "alpha beta gamma", 200 / 3),  # order matters
    ("alpha extra beta extra gamma", "alpha beta gamma", 75),  # noncontiguous LCS
    ("beta alpha", "alpha\nbeta", 50),  # rougeL, not sentence union rougeLsum
    ("CATS, running!", "cat run", 100),  # canonical English normalization/stemming
    ("wrong", "answer", 0),
    ("", "answer", 0),
    ("answer", "", 0),
    ("", "", 0),
    ("...", "!?", 0),
    ("你好", "English answer", 0),  # a wrong-script prediction is still an answer
])
def test_rouge_l_known_f1_values(prediction, reference, expected):
    assert rouge_l_score(prediction, reference) == pytest.approx(expected)


def test_multilingual_tokenization_is_explicit_and_does_not_silently_drop_answers():
    with pytest.raises(ValueError, match="eval_rouge_tokenizer='unicode'"):
        rouge_l_score("你好", "你好")
    assert rouge_l_score("你好", "你好", "unicode") == 100
    assert rouge_l_score("你好吗", "你好啊", "unicode") == pytest.approx(200 / 3)
    assert rouge_l_score("ПРИВЕТ мир", "привет мир", "unicode") == 100
    assert rouge_l_score("CAFÉ", "cafe\u0301", "unicode") == 100
    assert rouge_l_score("ｃａｔ", "cat", "unicode") == 100
    assert rouge_l_score("cats", "cat", "unicode") == 0  # no English stemming


def test_metric_definition_records_comparable_protocol():
    english, unicode = metric_definition(), metric_definition("unicode")
    assert english["variant"] == "rougeL"
    assert english["version"] == "0.1.2"
    assert english["statistic"] == "fmeasure"
    assert english["aggregation"] == "arithmetic_mean_over_examples"
    assert english["range"] == [0, 100] and english["higher_is_better"]
    assert english["use_stemmer"] and not english["truncate_references"]
    assert english["stemmer"] == {"implementation": "nltk.stem.porter.PorterStemmer",
                                   "version": importlib.metadata.version("nltk")}
    assert not unicode["use_stemmer"] and unicode["unicode_version"]
    assert "stemmer" not in unicode
    with pytest.raises(ValueError, match="tokenizer"):
        metric_definition("automatic")


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 1
    words = {20: "alpha", 21: "beta", 22: "gamma", 23: "delta"}

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            enable_thinking, **kwargs):
        assert tokenize and add_generation_prompt and enable_thinking is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        self.rendered.append(messages[0]["content"])
        return [2, 10 + len(messages[0]["content"]), 3]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(self.words[value] for value in ids if value not in {1, 4})


class FakeModel(torch.nn.Module):
    def __init__(self, sequences, *, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout()
        self.generation_config = SimpleNamespace(eos_token_id=[1, 4])
        self.sequences = sequences
        self.calls = []
        self.fail = fail

    def generate(self, *, input_ids, attention_mask, **kwargs):
        assert not self.training and not torch.is_grad_enabled()
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert not kwargs["do_sample"]
        assert kwargs["eos_token_id"] == [1, 4]
        assert kwargs["use_cache"]
        self.calls.append((input_ids.tolist(), kwargs))
        # A model implementation or callback can use these RNGs even if the
        # decoding algorithm itself is greedy.
        random.random()
        np.random.random()
        torch.rand(1)
        if self.fail:
            raise RuntimeError("generation failed")
        answer = self.sequences[len(self.calls) - 1][:kwargs["max_new_tokens"]]
        return torch.cat((input_ids, input_ids.new_tensor([answer])), dim=-1)


@pytest.fixture
def setup():
    rows = [
        {"id": "short", "prompt": "first", "response": "alpha"},
        {"id": "long", "prompt": "second", "response": "beta gamma delta beta gamma delta"},
    ]
    cfg = {"max_prompt_tokens": 32, "max_new_tokens": 4, "enable_thinking": False}
    return FakeModel([[20, 1], [1]]), FakeTokenizer(), rows, cfg


def test_generates_prompt_only_and_decodes_continuation_then_macro_averages(setup):
    model, tokenizer, rows, cfg = setup
    predictions = []
    metrics = evaluate_records(model, tokenizer, rows, cfg, on_prediction=predictions.append)
    assert metrics == {"rouge_l": 50, "examples": 2, "generated_tokens": 3}
    assert tokenizer.rendered == ["first", "second"]
    assert [record["prediction"] for record in predictions] == ["alpha", ""]
    assert predictions[0]["response_ids"] == [20, 1]
    assert predictions[0]["prompt_ids"] == [2, 15, 3]
    assert model.calls[0][0] == [[2, 15, 3]]
    assert all(call[1]["max_new_tokens"] == 4 for call in model.calls)
    assert model.training


def test_reference_remains_untruncated_and_eval_token_budget_is_independent(setup):
    model, tokenizer, rows, cfg = setup
    rows = [{**rows[0], "response": "alpha beta gamma delta"}]
    metrics = evaluate_records(model, tokenizer, rows, {**cfg, "eval_max_new_tokens": 1})
    assert metrics["rouge_l"] == 40  # one match against all four reference words
    assert metrics["generated_tokens"] == 1
    assert model.calls[0][1]["max_new_tokens"] == 1


def test_all_rows_by_default_and_explicit_training_subset(setup):
    model, tokenizer, rows, cfg = setup
    all_rows = evaluate_records(model, tokenizer, rows, {**cfg, "eval_max_examples": 1})
    assert all_rows["examples"] == 2
    one = evaluate_records(FakeModel([[20, 1]]), tokenizer, rows, cfg, max_examples=1)
    assert one == {"rouge_l": 100, "examples": 1, "generated_tokens": 2}


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_equal(before):
    after = _rng_snapshot()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and before[1][2:] == after[1][2:]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize("failure", [None, "generation", "callback"])
@pytest.mark.parametrize("training", [False, True])
def test_rng_and_all_submodule_modes_restored_even_after_error(setup, failure, training):
    model, tokenizer, rows, cfg = setup
    model.train(training)
    model.dropout.train(not training)
    before = _rng_snapshot()

    def callback(_record):
        random.random()
        np.random.random()
        torch.rand(1)
        if failure == "callback":
            raise RuntimeError("callback failed")

    model.fail = failure == "generation"
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            evaluate_records(model, tokenizer, rows, cfg, on_prediction=callback)
    else:
        evaluate_records(model, tokenizer, rows, cfg, on_prediction=callback)
    _assert_rng_equal(before)
    assert model.training is training
    assert model.dropout.training is not training


def test_cached_resume_generates_only_missing_answers_and_matches_full_evaluation(setup):
    model, tokenizer, rows, cfg = setup
    predictions = []
    expected = evaluate_records(model, tokenizer, rows, cfg, on_prediction=predictions.append)
    resumed_model, new_predictions = FakeModel([[1]]), []
    resumed = evaluate_records(resumed_model, tokenizer, rows, cfg,
        cached_predictions={predictions[0]["id"]: predictions[0]},
        on_prediction=new_predictions.append)
    assert resumed == expected
    assert len(resumed_model.calls) == 1
    assert [row["id"] for row in new_predictions] == ["long"]
    all_cached = evaluate_records(FakeModel([]), tokenizer, rows, cfg,
        cached_predictions={row["id"]: row for row in predictions})
    assert all_cached == expected


@pytest.mark.parametrize("key,value", [
    ("reference", "different"), ("prompt", "different"), ("id", "different"),
    ("prompt_ids", [9]), ("response_ids", [21]), ("response_ids", [-1]),
    ("response_ids", [20] * 5), ("prediction", "different"),
    ("rouge_l", float("nan")), ("rouge_l", 99),
    ("generated_tokens", True), ("generated_tokens", 1),
])
def test_resumption_rejects_stale_or_corrupt_cached_predictions(setup, key, value):
    model, tokenizer, rows, cfg = setup
    predictions = []
    evaluate_records(model, tokenizer, rows[:1], cfg, on_prediction=predictions.append)
    damaged = copy.deepcopy(predictions[0])
    damaged[key] = value
    with pytest.raises(ValueError, match="[Cc]ached"):
        evaluate_records(FakeModel([]), tokenizer, rows[:1], cfg,
                         cached_predictions={"short": damaged})


def test_invalid_dataset_and_nonenglish_reference_rejected_before_generation(setup):
    model, tokenizer, rows, cfg = setup
    with pytest.raises(ValueError, match="at least one"):
        evaluate_records(model, tokenizer, [], cfg)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_records(model, tokenizer, [rows[0], rows[0]], cfg)
    with pytest.raises(ValueError, match="eval_rouge_tokenizer"):
        evaluate_records(model, tokenizer, [{**rows[0], "response": "你好"}], cfg)
    with pytest.raises(ValueError, match="map IDs"):
        evaluate_records(model, tokenizer, rows, cfg, cached_predictions={"unknown": {}})
    assert not model.calls


def test_context_budget_rejected_before_generation_with_rng_and_modes_restored(setup):
    model, tokenizer, rows, cfg = setup
    model.config = SimpleNamespace(max_position_embeddings=6)
    model.dropout.eval()
    predictions = []
    before = _rng_snapshot()
    with pytest.raises(ValueError, match=r"budget \(3 \+ 4\).*context \(6\).*short"):
        evaluate_records(model, tokenizer, rows, cfg, on_prediction=predictions.append)
    assert not model.calls and not predictions
    assert model.training and not model.dropout.training
    _assert_rng_equal(before)
    # The actual rendered prompt is three tokens: its full configured maximum
    # need not fit when the available prompt is shorter. Exact fit is valid.
    model.config.max_position_embeddings = 7
    metrics = evaluate_records(model, tokenizer, rows, cfg)
    assert metrics["examples"] == 2


@pytest.mark.parametrize("architecture", ["Llama", "Qwen3"])
def test_native_tiny_model_generates_scores_and_restores_training_state(architecture):
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2,
                                  "prompt": 3, "answer": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    config = getattr(transformers, architecture + "Config")(
        vocab_size=8, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4,
        max_position_embeddings=32, eos_token_id=1, pad_token_id=0,
    )
    model = getattr(transformers, architecture + "ForCausalLM")(config).train()
    # Control emitted logits while exercising native autoregressive generation.
    def emit_answer(_module, _inputs, output):
        output.logits.fill_(-100)
        output.logits[..., 4] = 100
        return output
    model.register_forward_hook(emit_answer)
    before = _rng_snapshot()
    predictions = []
    metrics = evaluate_records(model, tokenizer,
        [{"id": "native", "prompt": "prompt", "response": "answer answer answer"}],
        {"max_prompt_tokens": 8, "max_new_tokens": 3}, on_prediction=predictions.append)
    assert metrics == {"rouge_l": 100, "examples": 1, "generated_tokens": 3}
    assert predictions[0]["prediction"] == "answer answer answer"
    assert predictions[0]["response_ids"] == [4, 4, 4]
    _assert_rng_equal(before)
    assert model.training
