"""Vocabulary alignment, unsafe mismatch rejection and conditional losses."""

import copy
import json
from pathlib import Path

import pytest
import torch

from baseline_common.losses import token_divergence
from baseline_common.models import load_tokenizer
from baseline_common.vocabulary import (
    allowed_token_mask, assert_supported_token_ids, mask_logits,
    project_logits, project_token_ids, support_indices, validate_vocabulary,
)


@pytest.fixture(scope="module")
def llama_pair():
    paths = [Path("/nas/Models") / model for model in
             ("Meta-Llama-3-8B-Instruct", "Meta-Llama-3.2-1B-Instruct")]
    if not all((p / "tokenizer.json").exists() for p in paths):
        pytest.skip("Actual local Llama 3 / 3.2 tokenizer pair unavailable")
    return ([load_tokenizer(p) for p in paths],
            [json.loads((p / "config.json").read_text()) for p in paths])


@pytest.fixture(scope="module")
def shared_alignment(llama_pair):
    tokenizers, configs = llama_pair
    return validate_vocabulary(*tokenizers, policy="llama3_shared",
                               teacher_config=configs[0], student_config=configs[1])


def test_actual_llama_pair_requires_explicit_shared_policy(llama_pair, shared_alignment):
    tokenizers, _ = llama_pair
    with pytest.raises(ValueError, match="full token-to-ID maps differ"):
        validate_vocabulary(*tokenizers)
    assert shared_alignment["support_size"] == 128005
    assert shared_alignment["shared_special_ids"] == [128000, 128001, 128006, 128007, 128009]
    assert len(shared_alignment["excluded_ids"]) == 251
    assert int(allowed_token_mask(shared_alignment).sum()) == 128005


@pytest.mark.parametrize("method", ["kd", "skd", "abkd", "skew_forward", "skew_reverse"])
def test_conditional_loss_ignores_excluded_logits_and_gradients(shared_alignment, method):
    generator = torch.Generator().manual_seed(812)
    student = torch.randn(1, 128256, generator=generator, requires_grad=True)
    teacher = torch.randn(1, 128256, generator=generator, requires_grad=True)
    excluded = shared_alignment["excluded_ids"]
    changed_student, changed_teacher = student.detach().clone(), teacher.detach().clone()
    # These values would dominate an unprojected softmax, and an -inf mask
    # would produce undefined products in several distribution objectives.
    changed_student[:, excluded] = float("nan")
    changed_teacher[:, excluded] = 10000
    original = token_divergence(project_logits(student, shared_alignment),
                                project_logits(teacher, shared_alignment), method).sum()
    changed = token_divergence(project_logits(changed_student, shared_alignment),
                               project_logits(changed_teacher, shared_alignment), method).sum()
    assert torch.isfinite(original)
    torch.testing.assert_close(original, changed)
    original.backward()
    assert torch.count_nonzero(student.grad[:, excluded]) == 0
    assert torch.count_nonzero(student.grad[:, :128000]) > 0
    assert teacher.grad is None


def test_projected_ce_preserves_target_semantics(shared_alignment):
    ids = torch.tensor([0, 128000, 128001, 128006, 128007, 128009, -100])
    projected = project_token_ids(ids, shared_alignment)
    assert projected.tolist() == [0, 128000, 128001, 128002, 128003, 128004, -100]
    selected = ids != -100
    assert torch.equal(support_indices(shared_alignment)[projected[selected]], ids[selected])
    logits = torch.zeros(6, 128256)
    logits[torch.arange(6), ids[selected]] = 30
    loss = torch.nn.functional.cross_entropy(project_logits(logits, shared_alignment), projected[selected])
    assert loss < 1e-6


def test_generation_keeps_original_ids_and_suppresses_unsupported_specials(shared_alignment):
    logits = torch.zeros(1, 128256)
    logits[:, 128008] = 10000
    logits[:, 128009] = 10
    masked = mask_logits(logits, shared_alignment)
    assert masked.argmax(-1).item() == 128009
    assert torch.isneginf(masked[:, shared_alignment["excluded_ids"]]).all()
    assert_supported_token_ids([128000, 123, 128006, 128007, 128009], shared_alignment)
    for ids in ([128008], [128004], [-1], [128256], [1.5]):
        with pytest.raises(ValueError):
            assert_supported_token_ids(ids, shared_alignment)
    with pytest.raises(ValueError, match="outside shared vocabulary"):
        project_token_ids(torch.tensor([128008]), shared_alignment)


class _Backend:
    def __init__(self, value):
        self.value = value

    def to_str(self):
        return json.dumps(self.value)


class _MutatedTokenizer:
    def __init__(self, tokenizer, mutate):
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        vocab = dict(tokenizer.get_vocab())
        mutate(backend, vocab)
        self.backend_tokenizer = _Backend(backend)
        self.vocab = vocab
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def get_vocab(self):
        return self.vocab


@pytest.mark.parametrize("mutation", ["lexical_vocab", "merges", "normalizer", "active_special", "special_flags", "added_unknown", "full_map"])
def test_shared_policy_rejects_unsafe_mismatches(llama_pair, mutation):
    tokenizers, configs = llama_pair

    def mutate(backend, vocab):
        if mutation == "lexical_vocab":
            keys = list(backend["model"]["vocab"])[:2]
            a, b = keys
            backend["model"]["vocab"][a], backend["model"]["vocab"][b] = backend["model"]["vocab"][b], backend["model"]["vocab"][a]
        elif mutation == "merges":
            backend["model"]["merges"] = backend["model"]["merges"][1:]
        elif mutation == "normalizer":
            backend["normalizer"] = {"type": "Lowercase"}
        elif mutation == "active_special":
            backend["added_tokens"][6]["content"] = "<|reserved_special_token_9000|>"
        elif mutation == "special_flags":
            backend["added_tokens"][6]["normalized"] = True
        elif mutation == "added_unknown":
            backend["added_tokens"][8]["content"] = "<|unrelated_active_token|>"
        elif mutation == "full_map":
            keys = list(backend["model"]["vocab"])[:2]
            vocab[keys[0]], vocab[keys[1]] = vocab[keys[1]], vocab[keys[0]]

    changed = _MutatedTokenizer(tokenizers[1], mutate)
    with pytest.raises(ValueError):
        validate_vocabulary(tokenizers[0], changed, policy="llama3_shared",
                            teacher_config=configs[0], student_config=configs[1])


@pytest.mark.parametrize("change", [{"model_type": "qwen2"}, {"vocab_size": 128255}, None])
def test_shared_policy_requires_matching_llama_metadata(llama_pair, change):
    tokenizers, configs = llama_pair
    student_config = copy.deepcopy(configs[1]) if change is not None else None
    if change is not None:
        student_config.update(change)
    with pytest.raises(ValueError):
        validate_vocabulary(*tokenizers, policy="llama3_shared",
                            teacher_config=configs[0], student_config=student_config)


def test_full_policy_is_identity_for_matching_actual_tokenizer(llama_pair):
    tokenizer = llama_pair[0][1]
    alignment = validate_vocabulary(tokenizer, tokenizer)
    logits = torch.randn(1, 128256)
    assert project_logits(logits, alignment) is logits
    assert mask_logits(logits, alignment) is logits
    assert alignment["excluded_ids"] == []


def test_head_shape_must_match_audited_policy(shared_alignment):
    for operation in (project_logits, mask_logits):
        with pytest.raises(ValueError, match="output vocabulary size"):
            operation(torch.zeros(1, 128000), shared_alignment)


def test_full_policy_preserves_padded_output_head_rows(llama_pair):
    tokenizer = llama_pair[0][1]
    alignment = validate_vocabulary(tokenizer, tokenizer,
                                    teacher_config={"vocab_size": 128512},
                                    student_config={"vocab_size": 128512})
    logits = torch.randn(1, 128512)
    assert project_logits(logits, alignment) is logits
    assert allowed_token_mask(alignment).all()
    with pytest.raises(ValueError, match="equal sizes"):
        validate_vocabulary(tokenizer, tokenizer,
                            teacher_config={"vocab_size": 128512},
                            student_config={"vocab_size": 128256})
