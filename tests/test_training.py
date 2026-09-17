"""End-to-end native HF checkpoint, training, resume and optimizer checks.

All artifacts are tiny random models created offline in pytest's temporary
directory. These tests establish execution/correctness, not model quality.
"""

import copy
import importlib
import json
from pathlib import Path

import pytest
import torch

from baseline_common.config import validate_config
from baseline_common.data import file_sha256, write_jsonl
from baseline_common.models import load_model, model_fingerprint, render_prompt
from baseline_common.optim import CPUAdamW


@pytest.fixture(scope="module", autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["Llama", "Qwen3"])
def tiny_experiment(tmp_path, request):
    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocabulary = {token: i for i, token in enumerate(
        ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>",
         "one", "two", "three", "four", "answer", "short", "long", "heldout", "Continue", "OK"]
    )}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
        bos_token="<bos>", eos_token="<eos>", additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = (
        "{{ bos_token }} {% for message in messages %}"
        "{{ '<user>' if message['role'] == 'user' else '<assistant>' }} {{ message['content'] }} "
        "{% if message['role'] == 'assistant' %}{{ eos_token }} {% endif %}"
        "{% endfor %}{% if add_generation_prompt %}<assistant> {% endif %}"
    )
    architecture = request.param
    model_class = getattr(transformers, architecture + "ForCausalLM")
    config_class = getattr(transformers, architecture + "Config")
    for role, seed, width, layers in [("teacher", 10, 32, 2), ("student", 20, 16, 1)]:
        torch.manual_seed(seed)
        config = config_class(
            vocab_size=32, hidden_size=width, intermediate_size=width * 2,
            num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=2,
            head_dim=width // 2, max_position_embeddings=64,
            bos_token_id=2, eos_token_id=3, pad_token_id=0, attention_dropout=0.1,
        )
        model = model_class(config)
        model.save_pretrained(tmp_path / role)
        tokenizer.save_pretrained(tmp_path / role)
    raw_rows = [
        {"id": "a", "prompt": "one", "response": "short"},
        {"id": "b", "prompt": "two", "response": "long answer"},
        {"id": "c", "prompt": "three", "response": "long long answer"},
    ]
    write_jsonl(tmp_path / "train.jsonl", raw_rows)
    write_jsonl(tmp_path / "validation.jsonl", [{"id": "d", "prompt": "heldout four", "response": "answer"}])
    cfg = validate_config(dict(
        name="tiny", pair=architecture.lower(), method="kd", dataset="fixture",
        teacher_model=str(tmp_path / "teacher"), student_model=str(tmp_path / "student"),
        train_file=str(tmp_path / "train.jsonl"), validation_file=str(tmp_path / "validation.jsonl"),
        output_root=str(tmp_path / "results"), device="cpu", teacher_device="cpu", dtype="float32",
        optimizer_offload=True, max_prompt_tokens=16, max_new_tokens=5, learning_rate=0.003,
        gradient_accumulation_steps=2, max_steps=2, gradient_checkpointing=True,
        save_steps=1, eval_steps=1, loss_chunk_size=2, acceptance_k=7, proposal_block_size=3,
        eval_during_training=True, eval_max_new_tokens=5,
        generation=dict(temperature=0.8, top_p=0.9, top_k=0),
        teacher_generation=dict(temperature=0.6, top_p=0.9, top_k=0),
    ))
    return cfg, tokenizer, raw_rows


def _paired_config(base, tokenizer, raw_rows):
    """Use actual native generation for both roles with producer provenance."""
    from scripts.generate_pairs import generate_role
    from baseline_common.data import join_pair_records

    cfg = copy.deepcopy(base)
    cfg["method"] = "distillm2"
    teacher = load_model(cfg["teacher_model"], "float32", "cpu")
    student = load_model(cfg["student_model"], "float32", "cpu")
    chosen = generate_role(teacher, tokenizer, raw_rows, cfg, cfg["teacher_generation"])
    rejected = generate_role(student, tokenizer, raw_rows, cfg, cfg["generation"])
    provenance = {
        "teacher": model_fingerprint(cfg["teacher_model"]),
        "student": model_fingerprint(cfg["student_model"]),
        "tokenizer": {"path": cfg["student_model"]},
        **{key: cfg[key] for key in ("enable_thinking", "max_prompt_tokens", "max_new_tokens", "generation", "teacher_generation")},
    }
    paired_path = Path(cfg["train_file"]).with_name("pairs.jsonl")
    write_jsonl(paired_path, join_pair_records(chosen, rejected, provenance=provenance))
    cfg["train_file"] = str(paired_path)
    return cfg


def _state(path):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(path, local_files_only=True).state_dict()


def _json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.mark.parametrize("method", ["kd", "abkd", "skd", "distillm2"])
def test_two_real_updates_frozen_teacher_save_reload_and_exact_resume(tiny_experiment, method, monkeypatch):
    import transformers

    training = importlib.import_module("baseline_common.train")
    cfg, tokenizer, raw_rows = tiny_experiment
    cfg = _paired_config(cfg, tokenizer, raw_rows) if method == "distillm2" else {**cfg, "method": method}
    initial = _state(cfg["student_model"])
    teacher_path = Path(cfg["teacher_model"])
    teacher_hashes = {p.name: file_sha256(p) for p in teacher_path.iterdir() if p.is_file()}
    loaded_teachers = []
    actual_load = training.load_model

    def observe_load(path, *args, **kwargs):
        model = actual_load(path, *args, **kwargs)
        if Path(path) == teacher_path:
            loaded_teachers.append(model)
            def assert_frozen(module, _inputs):
                assert not module.training
                assert not torch.is_grad_enabled()
                assert all(not p.requires_grad and p.grad is None for p in module.parameters())
            model.register_forward_pre_hook(assert_frozen)
        return model

    monkeypatch.setattr(training, "load_model", observe_load)
    complete = training.train(cfg)
    metrics = _json_lines(complete / "metrics.jsonl")
    assert [row["step"] for row in metrics] == [1, 2]
    assert all(torch.isfinite(torch.tensor([row["loss"], row["grad_norm"]])).all() for row in metrics)
    assert all(row["grad_norm"] > 0 for row in metrics)
    result = json.loads((complete / "result.json").read_text())
    assert result["complete"] and result["step"] == 2
    assert result["counters"]["unique_epoch_examples"] == 3
    if method == "skd":
        assert result["counters"]["skd_generated_tokens"] > 0
        assert result["counters"]["skd_teacher_forward_calls"] > 0
    assert (complete / "checkpoints/step_000002/complete.json").is_file()
    trained = _state(complete / "final")
    assert any(not torch.equal(value, trained[name]) for name, value in initial.items())
    assert all(torch.isfinite(value).all() for value in trained.values())
    reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(complete / "final", local_files_only=True)
    assert reloaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert reloaded_tokenizer.chat_template == tokenizer.chat_template
    assert {p.name: file_sha256(p) for p in teacher_path.iterdir() if p.is_file()} == teacher_hashes
    assert loaded_teachers and all(all(p.grad is None and not p.requires_grad for p in t.parameters()) for t in loaded_teachers)

    resumed_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]).with_name("resumed"))}
    paused = training.train(resumed_cfg, stop_after_steps=1)
    paused_result = json.loads((paused / "result.json").read_text())
    assert not paused_result["complete"]
    checkpoint = paused / "checkpoints/step_000001"
    assert (checkpoint / "rng_rank0.pt").is_file()
    # Deliberately disturb global RNGs; resumption must restore saved sampling
    # and dropout state, including all SKD stochastic proposals/interventions.
    torch.manual_seed(654321)
    import random
    random.seed(7654321)
    resumed = training.train(resumed_cfg, resume=str(checkpoint))
    resumed_state = _state(resumed / "final")
    for key in trained:
        torch.testing.assert_close(resumed_state[key], trained[key], rtol=0, atol=0)
    assert _json_lines(resumed / "metrics.jsonl") == metrics
    resumed_result = json.loads((resumed / "result.json").read_text())
    assert resumed_result["counters"] == result["counters"]
    assert _json_lines(resumed / "validation.jsonl") == _json_lines(complete / "validation.jsonl")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_adamw_matches_fp32_oracle_and_preserves_master_state_on_resume(dtype, tmp_path):
    initial = torch.tensor([0.121, -0.913, 1.001, 0.004]).to(dtype)
    parameter = torch.nn.Parameter(initial.clone())
    reference = torch.nn.Parameter(initial.float().clone())
    options = dict(lr=0.00013, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.07)
    offloaded = CPUAdamW([parameter], **options)
    oracle = torch.optim.AdamW([reference], **options)
    resumed = None
    for step in range(7):
        gradient = torch.tensor([0.13 + step / 10, -0.8, -0.031, 0.014]).to(dtype)
        parameter.grad = gradient.clone()
        reference.grad = gradient.float().clone()
        offloaded.step()
        oracle.step()
        torch.testing.assert_close(offloaded.masters[0], reference, rtol=0, atol=0)
        torch.testing.assert_close(parameter, reference.to(dtype), rtol=0, atol=0)
        if resumed is not None:
            resumed.parameters[0].grad = gradient.clone()
            resumed.step()
            torch.testing.assert_close(resumed.masters[0], reference, rtol=0, atol=0)
            torch.testing.assert_close(resumed.parameters[0], parameter, rtol=0, atol=0)
        if step == 2:
            torch.save(offloaded.state_dict(), tmp_path / "optimizer.pt")
            restored_parameter = torch.nn.Parameter(parameter.detach().clone())
            resumed = CPUAdamW([restored_parameter], **options)
            resumed.load_state_dict(torch.load(tmp_path / "optimizer.pt", weights_only=False))
            for state in resumed.optimizer.state.values():
                assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    if dtype == torch.bfloat16:
        # The FP32 master must retain sub-BF16 updates rather than round after
        # every step, or exact continuation progressively loses information.
        assert not torch.equal(offloaded.masters[0], parameter.detach().float())
    offloaded.zero_grad()
    assert parameter.grad is None and offloaded.masters[0].grad is None


def _ddp_worker(rank, config, port):
    import os
    import torch.distributed as distributed
    from baseline_common.train import train

    torch.set_num_threads(1)
    os.environ.update(WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    try:
        train(config)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


@pytest.mark.parametrize("method", ["kd", "abkd", "skd", "distillm2"])
def test_two_cpu_ranks_match_global_token_mean_with_unequal_lengths_and_partial_tail(tiny_experiment, method):
    """Two ranks must optimize the same token mean as one global batch.

    Three rows give a final step where rank 1 has no real example; the DDP
    synchronization dummy must neither change the objective nor token counts.
    """
    import socket
    import torch.multiprocessing as multiprocessing
    from baseline_common.config import run_directory
    from baseline_common.train import train

    cfg, tokenizer, raw_rows = tiny_experiment
    for role in ("teacher_model", "student_model"):
        path = Path(cfg[role]) / "config.json"
        model_config = json.loads(path.read_text())
        model_config["attention_dropout"] = 0
        path.write_text(json.dumps(model_config))
    cfg["gradient_checkpointing"] = True
    cfg = _paired_config(cfg, tokenizer, raw_rows) if method == "distillm2" else {**cfg, "method": method}
    single_output = train(cfg)
    distributed_cfg = {**cfg, "gradient_accumulation_steps": 1,
                       "output_root": str(Path(cfg["output_root"]).with_name("distributed"))}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
    multiprocessing.spawn(_ddp_worker, args=(distributed_cfg, port), nprocs=2, join=True)
    distributed_output = run_directory(distributed_cfg)
    expected = _state(single_output / "final")
    actual = _state(distributed_output / "final")
    # Reduction order differs between accumulated single-rank gradients and
    # all-reduced gradients; allow floating-point summation roundoff only.
    if method != "skd":
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=2e-6)
        for single, parallel in zip(_json_lines(single_output / "metrics.jsonl"), _json_lines(distributed_output / "metrics.jsonl")):
            assert parallel["loss"] == pytest.approx(single["loss"], rel=1e-5, abs=1e-6)
            assert parallel["grad_norm"] == pytest.approx(single["grad_norm"], rel=1e-5, abs=1e-6)
    else:
        # Rank-specific sampling RNGs define different legitimate SKD rollouts;
        # compare health and accounting, not equality to another trajectory.
        initial = _state(cfg["student_model"])
        assert any(not torch.equal(actual[name], initial[name]) for name in actual)
        assert all(torch.isfinite(value).all() for value in actual.values())
        distributed_metrics = _json_lines(distributed_output / "metrics.jsonl")
        assert [row["step"] for row in distributed_metrics] == [1, 2]
        assert all(torch.isfinite(torch.tensor([row["loss"], row["grad_norm"]])).all() and row["grad_norm"] > 0 for row in distributed_metrics)
    single_result = json.loads((single_output / "result.json").read_text())
    parallel_result = json.loads((distributed_output / "result.json").read_text())
    assert parallel_result["complete"]
    assert parallel_result["counters"]["unique_epoch_examples"] == 3
    if method == "skd":
        assert parallel_result["counters"]["response_tokens"] == parallel_result["counters"]["skd_generated_tokens"]
    else:
        assert parallel_result["counters"]["response_tokens"] == single_result["counters"]["response_tokens"]
    assert (distributed_output / "checkpoints/step_000002/rng_rank1.pt").is_file()


def test_resume_rejects_stale_completed_checkpoint_without_rewriting_logs(tiny_experiment):
    from baseline_common.train import train

    cfg, _, _ = tiny_experiment
    output = train(cfg)
    before = {name: (output / name).read_bytes() for name in
              ("metrics.jsonl", "validation.jsonl", "best_checkpoint.json", "result.json")}
    with pytest.raises(ValueError, match="latest checkpoint"):
        train(cfg, resume=str(output / "checkpoints/step_000001"))
    assert {name: (output / name).read_bytes() for name in before} == before
    assert not list(output.glob("*.before_resume_*"))


def test_resume_rejects_foreign_checkpoint_even_with_same_models_and_data(tiny_experiment):
    from baseline_common.train import train

    cfg, _, _ = tiny_experiment
    original = train(cfg, stop_after_steps=1)
    foreign_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]).with_name("other_run"))}
    other = train(foreign_cfg, stop_after_steps=1)
    before = {name: (original / name).read_bytes() for name in ("metrics.jsonl", "validation.jsonl", "result.json")}
    with pytest.raises(ValueError, match="belong to this run"):
        train(cfg, resume=str(other / "checkpoints/step_000001"))
    assert {name: (original / name).read_bytes() for name in before} == before


def test_crash_resume_preserves_uncommitted_artifacts_and_recovers_exact_skd_trajectory(tiny_experiment):
    from baseline_common.train import train

    cfg, _, _ = tiny_experiment
    cfg = {**cfg, "method": "skd"}
    uninterrupted = train(cfg)
    resumed_cfg = {**cfg, "output_root": str(Path(cfg["output_root"]).with_name("crashed"))}
    crashed = train(resumed_cfg, stop_after_steps=1)
    # Simulate a process killed after appending the next step's records but
    # before committing its checkpoint, including a partially written JSON row.
    corrupt_logs = {}
    for name in ("metrics.jsonl", "validation.jsonl"):
        log = crashed / name
        corrupted = log.read_bytes() + b'{"step": 2, "loss": -999}\n{"step": 3, "loss":'
        log.write_bytes(corrupted)
        corrupt_logs[name] = corrupted
    partial = crashed / "checkpoints/step_000002"
    partial.mkdir()
    (partial / "partial_model.bin").write_bytes(b"incomplete checkpoint must survive recovery")
    (crashed / "best_checkpoint.json").write_text(json.dumps({"step": 999, "rouge_l": 999,
                                      "path": "checkpoints/step_000999/student"}))
    checkpoint = crashed / "checkpoints/step_000001"
    resumed = train(resumed_cfg, resume=str(checkpoint))
    for name, original_bytes in corrupt_logs.items():
        archived = list(resumed.glob(name + ".before_resume_*"))
        assert len(archived) == 1 and archived[0].read_bytes() == original_bytes
        assert _json_lines(resumed / name) == _json_lines(uninterrupted / name)
    archived_checkpoints = list((resumed / "checkpoints").glob("step_000002.incomplete_*"))
    assert len(archived_checkpoints) == 1
    assert (archived_checkpoints[0] / "partial_model.bin").read_bytes() == b"incomplete checkpoint must survive recovery"
    assert (resumed / "checkpoints/step_000002/complete.json").is_file()
    expected, actual = _state(uninterrupted / "final"), _state(resumed / "final")
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    recovered_best = json.loads((resumed / "best_checkpoint.json").read_text())
    expected_best = json.loads((uninterrupted / "best_checkpoint.json").read_text())
    for field in ("step", "rouge_l", "path"):
        assert recovered_best[field] == expected_best[field]
    assert (resumed / recovered_best["path"]).is_dir()
