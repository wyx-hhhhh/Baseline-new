# Validation record (2026-09-16)

> Update: the user subsequently selected `/nas/Models/Meta-Llama-3-8B-Instruct`. Missing Llama-3.1 references below describe the earlier request. Current launch instructions and verification are in [llama3_launch.md](llama3_launch.md).

This record concerns the new four-method runner. The previous `check/validation_record.md` described an earlier investigation on a different machine and is not evidence about this runtime.

## Installed and checked environment

An isolated `~/Baseline/.venv` was created using Python **3.11.14**. Training pins are **torch 2.6.0+cu124, Transformers 4.51.3, Accelerate 1.6.0, datasets 3.5.0, tokenizers 0.21.1, NumPy 1.26.4, huggingface-hub 0.30.2, safetensors 0.5.3**. Both `uv pip check` and `python -m pip check` pass. The hashed Linux/Python 3.11 dependency lock resolves; a `--require-hashes` install dry-run against the installed environment would make no changes. This dry-run does not re-download and rehash every installed wheel.

Evidence: [strict pinned Qwen preflight](environment-pinned.json), [installed package inventory](environment-pinned-freeze.txt), [lock resolution log](evidence/dependency-lock.log), and [training lock](../requirements-training.lock). The preflight verified BF16 CUDA operations on all four RTX 3090 GPUs, complete Qwen shard membership and tensor headers, full tokenizer maps, generation EOS semantics and model-head sizes. The requested Llama teacher directory is empty; only its student artifacts could be examined.

## Automated correctness tests

**115 tests passed on the pinned stack:** the 71-test combined losses/sampling/data/training suite passed in 61.06 seconds, followed by the 44 new config/tokenizer tests in 4.06 seconds. These are CPU native-model tests, including two-process Gloo tests; they do not claim full-size CUDA/NCCL performance.

| Coverage | Evidence |
|---|---|
| KD direction, equal-gradient soft CE relation; AB divergence general case/limits vs independent float64 oracle; DistiLLM-2 detached reverse-mixture gradient and paired reduction | `tests/test_losses.py` |
| Full output vocabulary, first answer/EOS causal targets, masked NaNs, chunked/unchunked loss and gradient parity | `tests/test_losses.py` |
| Actual proposed-token acceptance, first/middle/last rejection, correction, EOS, full-vocabulary acceptance, generation budget, separate teacher/student sampling; tiny native Qwen3/Llama forward | `tests/test_sampling.py` |
| Raw adapters, duplicate/augmented-source grouping, reserved evaluation exclusion, strict pair-ID join, empty record handling, saved split hashes | `tests/test_data.py` |
| Both native Qwen3/Llama architectures × four methods: two optimizer updates, finite nonzero gradients, changed student, teacher frozen/no-grad, HF export/reload | `tests/test_training.py` |
| Exact uninterrupted vs checkpoint-resumed student weights/metrics/counters, including fresh SKD random generation; FP32 CPUAdam oracle and BF16 master-parameter preservation | `tests/test_training.py` |
| Two CPU ranks, all eight combinations, unequal response lengths and a rank with no real tail example; global objective matches single-rank reference for fixed trajectories; gradient checkpointing enabled | `tests/test_training.py` |
| Foreign/stale checkpoint rejection and crash recovery preserving incomplete logs/checkpoint bytes while replaying the committed step exactly | `tests/test_training.py` |
| Strict config schema/nonfinite rejection, same-size incompatible token maps/normalization rejection, actual local Qwen non-thinking template/truncation, real EOS when pad=EOS, cross-split source-group rejection | `tests/test_config_models.py` |

Reproduce after activating `.venv`:

```bash
python -m pytest -q tests
python -m pip check
python scripts/check_environment.py --pair qwen3 --strict-pins --output docs/environment-current.json
```

The test dependency is in `requirements-test.txt`. A Transformers deprecation warning about the server's existing `TRANSFORMERS_CACHE` environment setting does not affect results. Local-Qwen tests inspect tokenizers without loading the large weights.

## Dataset and CLI evidence

[Dataset verification](data-verification.json) records real local Dolly/GSM8K/MetaMathQA reads and preparation, including the 395,000-row MetaMathQA scan. Tiny native Llama teacher/student pair generation through the actual CLI saved exact IDs and sidecars with validated hashes. The native training tests additionally generate pairs from both tiny architectures.

All eight JSON experiment configs resolve through `scripts/train.py --dry-run`; the Qwen three-seed matrix prints 12 commands with the intended data/output paths. The persisted default Dolly corpus contains **14,246 train + 750 validation** records, with prompt-disjoint grouping and a source/split manifest at `/nas/Users/wyx/Baseline/data/dolly/manifest.json`. This is a new controlled split, not the historical MiniLLM split.

## Actual checkpoint GPU checks

All four real Qwen methods completed one short BF16 update using the supplied 8B/1.7B weights on separate RTX 3090 GPUs. They use synthetic examples and are infrastructure checks, not useful trained experimental models or convergence evidence.

| Method | Executed path | Loss | Peak student VRAM | Save verification |
|---|---|---:|---:|---|
| KD | Actual `scripts/train.py` production loop, validation, complete optimizer/RNG checkpoint, final export | 1.834280 | 7.587 GiB | Final/checkpoint probe identical and changed from source; saved FP32 masters/moments read successfully |
| ABKD | Production sample/loss/CPU optimizer helpers | 0.067976 | 7.587 GiB | Full HF reload, parameter probe, tokenizer and tied embeddings pass |
| SKD | Fresh proposal/acceptance sampler plus production loss/optimizer helpers | 0.107919 | 7.586 GiB | Full HF reload, parameter probe, tokenizer and tied embeddings pass |
| DistiLLM-2 | Native initial teacher/student pair generation plus both production skew-loss branches | 0.101308 | 8.166 GiB | Full HF reload, parameter probe, tokenizer and tied embeddings pass |

Teacher peak memory in the helper runs was about 15.28 GiB. All gradients were finite and nonzero; teacher eval/no-grad/frozen checks and unchanged teacher probes passed in the helper runs. The KD saved optimizer contains 310 FP32 master parameter tensors and FP32 moment states. These runs used a 64-prompt/8-response token **budget**, with actual responses shorter than that. The easy SKD example generated three accepted tokens and required no correction; rejection paths are established by the controlled sampler tests. The easy DistiLLM-2 chosen/rejected text happened to agree, but the model distributions and paired loss were nonidentical. Loss values across these different synthetic prompts/objectives are not a method ranking.

Evidence: [KD production-loop result](evidence/qwen-kd-smoke-result.json), [KD config](evidence/qwen-kd-smoke-config.json), [KD log](evidence/qwen-kd-smoke.log), [three-method helper results](evidence/qwen-method-smoke.json), and `scripts/smoke_large_models.py`. All processes exited successfully and released their GPUs.

Saved smoke models:

- KD: `/nas/Users/wyx/Baseline/smoke/runs/qwen3_8b_1p7b/synthetic_smoke/kd/seed_42/final`
- ABKD/SKD/DistiLLM-2: `/nas/Users/wyx/Baseline/smoke/qwen3_8b_1p7b/<method>/student`

NAS loading/checkpoint I/O took substantially longer than the short update. KD's `elapsed_seconds` is sampled inside the training loop before checkpoint serialization and excludes model loading; it is **not** total command wall time. Use external wall timing plus the recorded generation/forward counters for full experiment cost comparisons.

## Explicit limits

- The Llama-3.1 teacher is missing. Full-size Llama loading, token alignment and training cannot be tested until it is supplied at the requested exact path. Tiny native Llama architecture tests are not a substitute for those model artifacts.
- No complete corpus/three-seed training experiment or downstream benchmark result has been produced. The request's preparation, executable code and environment analysis are separate from those expensive runs.
- A short large-model smoke does not establish fit for the full 2,048-token budget. Profile the chosen sequence length and record peak memory before long experiments.
- CPU DDP correctness does not certify full-size NCCL resource behavior. The four-GPU profile is supplied with that limitation.
- SKD is a complete uncached implementation; its repeated-prefix compute differs from an optimized cache implementation. No throughput parity with the source paper is claimed.
- Local model fingerprints hash metadata and record weight sizes/mtimes; the audit checks safetensors headers and file lengths, not cryptographic hashes of every model tensor byte.

## Two concurrent GPU groups and automatic recovery

The new [parallel run guide](parallel_runs.md) supersedes the earlier manual matrix instructions for running Qwen and Llama together. The actual four-GPU tiny-model pipeline completed all eight method/model jobs and both pair producers, then skipped training on restart with no model reloads; see [GPU evidence](evidence/parallel-gpu-smoke.json). Model weights were small random fixtures; full NAS datasets/models were not launched. Additional tests cover signals, durable checkpoint commits, pair-generator row recovery, configuration/runtime identity, duplicate and orphan locks, corrupted artifact rejection, and repeated interruption during final export recovery. The complete regression log is [parallel-full-tests.log](evidence/parallel-full-tests.log).

## ROUGE-L and training/evaluation separation

The current natural-language evaluation protocol is generated-response ROUGE-L F1 (macro mean, 0–100, higher is better), with training without evaluation by default and a separate resumable evaluation command. Historical NLL smoke results above remain observations of the older protocol, not current metric outputs. The new full regression pass completed **331 tests** successfully: [test log](evidence/rouge-full-tests.log). Additional focused tests verify genuine LCS F1, full references, greedy generation, per-example recovery, leakage checks, highest-score checkpoint selection, RNG preservation, and evaluation of all eight models without teacher directories. All 750 prepared Dolly validation references are valid for the English metric: [readiness](evidence/rouge-dolly-readiness.json). [Current two commands](train_evaluate.md).

The final CUDA evaluation check passed for all eight tiny trained baselines, with teachers unavailable and no model reload on restart: [GPU evaluation evidence](evidence/rouge-gpu-evaluation.json). The uncommitted best-checkpoint guard also passed. These two added cases bring the current suite to 333 tests, verified by the full pass plus the final targeted pass.
