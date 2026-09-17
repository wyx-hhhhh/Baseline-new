"""Deterministic causal-model oracles for proposal acceptance and intervention."""

from types import SimpleNamespace

import pytest
import torch

from baseline_common.sampling import SamplingConfig, sample_token, sampling_logits, skd_generate


class TableLM(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.embedding = torch.nn.Embedding(len(table), len(table))
        self.embedding.weight = torch.nn.Parameter(torch.as_tensor(table, dtype=torch.float32))
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=0):
        assert attention_mask.eq(1).all()
        assert not use_cache
        self.calls.append((input_ids.tolist(), torch.is_grad_enabled(), self.training, logits_to_keep))
        logits = self.embedding(input_ids)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


def _table(next_ids, vocab=None):
    vocab = vocab or len(next_ids)
    table = torch.full((vocab, vocab), -20.0)
    for row, token in enumerate(next_ids):
        table[row, token] = 20
    return table


GREEDY_SUPPORT = SamplingConfig(temperature=0.7, top_p=1, top_k=1)


def _run(student, teacher, **kwargs):
    defaults = dict(
        max_new_tokens=5, eos_token_ids=[], proposal_config=GREEDY_SUPPORT,
        teacher_config=GREEDY_SUPPORT, acceptance_k=1, block_size=3, return_stats=True,
    )
    defaults.update(kwargs)
    return skd_generate(student, teacher, torch.tensor([0]), **defaults)


def test_accepts_actual_emitted_proposals_then_intervenes_at_first_rejection():
    student = TableLM(_table([1, 2, 3, 4, 5, 5]))
    teacher = TableLM(_table([1, 4, 3, 4, 5, 5]))
    sequence, stats = _run(student, teacher, eos_token_ids=[5])
    assert sequence.tolist() == [[0, 1, 4, 5]]
    assert stats["proposed_tokens"] == 4  # 1,2,3 then 5 after the correction.
    assert stats["accepted_tokens"] == 2
    assert stats["teacher_interventions"] == 1
    assert stats["teacher_forward_calls"] == 2
    assert stats["student_forward_tokens"] == 1 + 2 + 3 + 3
    assert stats["teacher_forward_tokens"] == 4 + 4
    assert teacher.calls[0][0] == [[0, 1, 2, 3]]
    assert teacher.calls[1][0] == [[0, 1, 4, 5]]


def test_fully_accepted_blocks_do_not_append_teacher_bonus_and_respect_budget():
    student = TableLM(_table([1, 2, 3, 4, 5, 5]))
    teacher = TableLM(_table([1, 2, 3, 4, 5, 5]))
    sequence, stats = _run(student, teacher, max_new_tokens=5, block_size=2)
    assert sequence.tolist() == [[0, 1, 2, 3, 4, 5]]
    assert stats["accepted_tokens"] == stats["generated_tokens"] == 5
    assert stats["teacher_interventions"] == 0
    assert stats["teacher_forward_calls"] == 3


def test_accepted_eos_is_retained_even_when_it_is_also_pad():
    student = TableLM(_table([1, 0, 2]))
    teacher = TableLM(_table([1, 0, 2]))
    sequence, stats = _run(student, teacher, eos_token_ids=[0, 2])
    assert sequence.tolist() == [[0, 1, 0]]
    assert stats["generated_tokens"] == 2
    assert stats["teacher_interventions"] == 0


def test_teacher_intervention_eos_stops_and_discards_remaining_proposals():
    student = TableLM(_table([1, 1, 2]))
    teacher = TableLM(_table([2, 1, 2]))
    sequence, stats = _run(student, teacher, eos_token_ids=[2])
    assert sequence.tolist() == [[0, 2]]
    assert stats["teacher_interventions"] == 1
    assert stats["generated_tokens"] == 1
    assert stats["proposed_tokens"] == 3


def test_acceptance_topk_is_independent_of_teacher_generation_topk():
    student = TableLM(_table([1, 1, 2]))
    teacher = TableLM([[5, 4, 1], [5, 4, 1], [5, 4, 1]])
    sequence, stats = _run(student, teacher, acceptance_k=2, max_new_tokens=1)
    assert sequence.tolist() == [[0, 1]]  # Teacher would sample 0 with its top-k=1.
    assert stats["accepted_tokens"] == 1
    assert stats["teacher_interventions"] == 0


def test_generation_disables_grads_and_restores_independent_modes():
    student = TableLM(_table([1, 2, 2])).train()
    teacher = TableLM(_table([1, 2, 2])).eval()
    result, _ = _run(student, teacher, max_new_tokens=2)
    assert student.training and not teacher.training
    assert not result.requires_grad
    assert all(not grad and not training for _, grad, training, _ in student.calls + teacher.calls)
    # Generation does not put subsequent student training in no_grad.
    student.embedding(torch.tensor([[1]])).sum().backward()
    assert student.embedding.weight.grad is not None


def test_exception_restores_model_modes():
    student = TableLM(_table([1, 2, 2])).train()
    teacher = TableLM(_table([1, 2, 3, 3])).eval()
    with pytest.raises(ValueError, match="vocabularies"):
        _run(student, teacher)
    assert student.training and not teacher.training


def test_sampling_filters_temperature_topk_and_nucleus_boundary_without_mutation():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    original = logits.clone()
    scores = sampling_logits(logits, SamplingConfig(temperature=2, top_p=0.6, top_k=3))
    # Within top three, probabilities are ~[.506,.307,.186]; retain the second
    # token crossing .6, then remove the rest.
    assert torch.isfinite(scores).tolist() == [[True, True, False, False]]
    torch.testing.assert_close(scores[:, :2], logits[:, :2] / 2)
    torch.testing.assert_close(logits, original)


def test_separate_sampling_parameters_affect_proposal_and_teacher_independently():
    student = TableLM([[10, 9, 8], [10, 9, 8], [10, 9, 8]])
    teacher = TableLM([[0, 1, 10], [0, 1, 10], [0, 1, 10]])
    sequence, stats = _run(student, teacher, max_new_tokens=1,
                           proposal_config=SamplingConfig(0.2, 0.1, 1),
                           teacher_config=SamplingConfig(3.0, 1, 1))
    assert teacher.calls[0][0] == [[0, 0]]  # Actual student proposal.
    assert sequence.tolist() == [[0, 2]]  # Independent teacher replacement.
    assert stats["teacher_interventions"] == 1


def test_sampling_rng_is_repeatable():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    config = SamplingConfig(temperature=2, top_p=1, top_k=0)
    one = torch.Generator().manual_seed(83)
    two = torch.Generator().manual_seed(83)
    assert [sample_token(logits, config, one).item() for _ in range(20)] == [
        sample_token(logits, config, two).item() for _ in range(20)
    ]


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": float("inf")}, {"top_p": 0}, {"top_k": -1}])
def test_invalid_sampling_configuration(kwargs):
    with pytest.raises(ValueError):
        SamplingConfig(**kwargs)


@pytest.mark.parametrize("architecture", ["Llama", "Qwen3"])
def test_tiny_native_model_forward_without_transformers_generation_patches(architecture):
    transformers = pytest.importorskip("transformers")
    config = getattr(transformers, architecture + "Config")(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, max_position_embeddings=32,
    )
    model_class = getattr(transformers, architecture + "ForCausalLM")
    student = model_class(config)
    teacher = model_class(config)
    output, stats = skd_generate(student, teacher, torch.tensor([1, 4, 7]), max_new_tokens=4,
                                 eos_token_ids=[], acceptance_k=25, block_size=2, return_stats=True)
    assert output.shape == (1, 7)
    assert output[0, :3].tolist() == [1, 4, 7]
    assert stats["generated_tokens"] == 4
