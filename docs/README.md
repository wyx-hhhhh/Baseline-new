# Four-baseline experiments on this server

For the Dolly workflow, run `bash /home/wyx/Baseline/scripts/run_all.sh` to train, then `bash /home/wyx/Baseline/scripts/evaluate_all.sh` for ROUGE-L on the five downloaded [natural-language benchmarks](natural_language_evaluation.md). The evaluation wrapper now automatically prepares DollyEval, SelfInst, Super-Natural, Unnatural and VicunaEval, excludes training overlaps, and preserves multiple references. Use `evaluate_all.sh --validation-only` for the original internal Dolly validation split. See [the two-command guide](train_evaluate.md). The same command resumes interrupted work and skips completed baselines. See [parallel execution and recovery](parallel_runs.md).

The separate [fixed MetaMathQA campaign](math_campaign.md) is ready: all four methods share 50,000 training examples and 5,000 development examples, then evaluate each final student on full GSM8K and MATH with final-answer accuracy. Use `scripts/run_math.sh` for training and `scripts/evaluate_math_all.sh` afterward. Both support resuming. No full math training has been launched.

The executable path is now `baseline_common/` plus `scripts/`, with eight JSON configurations in `configs/experiments/`. The four original repositories are preserved as source references. **Use these new entry points, not the old installers or launch scripts.** No installed Transformers files are replaced.

**Current verification:** 115 tests passed on the installed pinned stack; all four methods completed short real-Qwen GPU updates and saved smoke models. The Llama group now uses the requested available Llama-3 teacher with an explicit shared-token policy; see the updated Llama launch guide. See the validation record for the exact scope.

The current request supersedes the older five-method proposal in `check`: run **KD, ABKD, SKD and DistiLLM-2**, with **Qwen3-8B → Qwen3-1.7B** and **Llama-3-8B → Llama-3.2-1B**, using the exact local Instruct directories supplied by the user. GKD is outside this request.

- [Environment and installation](environment.md): current packages, GPU/memory checks, exact installation commands and the updated Llama teacher selection.
- [Datasets](datasets.md): original training datasets, what actually exists locally, preparation/exclusion recipes and missing artifacts.
- [Implementation protocol](implementation.md): algorithm definitions, source provenance, memory behavior and intentional differences from original recipes.
- [Validation record](validation.md): executed checks and limits of the evidence.

## Storage

```text
~/Baseline/
  baseline_common/          shared infrastructure and separate loss/sampling implementations
  configs/experiments/      4 methods × 2 exact local model pairs
  scripts/                 check, prepare, generate pairs, train, run matrix
  tests/                   numerical, native-model, resume and distributed checks
  docs/                    all new analysis, instructions and evidence
/nas/Users/wyx/Baseline/
  data/<dataset>/
    train.jsonl
    validation.jsonl       optional ROUGE-L validation when explicitly enabled
    manifest.json          source and split hashes, counts and exclusions
    <pair>/pairs.train.jsonl
    <pair>/pairs.train.{teacher,student}.jsonl
    <pair>/pairs.train.manifest.json
  runs/<pair>/<dataset>/<method>/seed_<seed>/
    manifest.json          resolved settings, model metadata identity, runtime, data hashes
    metrics.jsonl
    validation.jsonl       optional ROUGE-L validation when explicitly enabled
    best_checkpoint.json   optional highest held-out ROUGE-L checkpoint
    checkpoints/step_XXXXXX/
      student/             HF model + tokenizer
      training.pt          optimizer, scheduler, counters and exact next step
      rng_rank<N>.pt       Python/PyTorch/CUDA random state per rank
      complete.json        written after all ranks finish saving
    final/                 final HF model + tokenizer
    result.json
```

Raw checkpoints and datasets under `/nas/Models` and `/nas/Datasets` are read-only inputs. An existing run cannot be accidentally overwritten. Seeds 42/43/44 can be run separately; one seed is the default.

## Environment and preflight

From `~/Baseline`, activate a compatible environment, then:

```bash
source .venv/bin/activate
python scripts/check_environment.py --pair qwen3 --output docs/environment-current.json
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_kd.json --dry-run
```

`--dry-run` resolves configuration only. The environment checker verifies local model files and actual CUDA access. At the user's updated request, the Llama teacher is `/nas/Models/Meta-Llama-3-8B-Instruct`. Its ordinary vocabulary matches the Llama-3.2 student, but its reserved/control tokens differ. The Llama configs explicitly select `vocabulary_policy=llama3_shared`: loss, validation and generation use the 128,000 ordinary tokens plus five identically mapped chat tokens. See [the Llama launch guide](llama3_launch.md).

The default memory profile uses student `cuda:0`, teacher `cuda:1`, BF16 model weights, gradient checkpointing, and FP32 CPU master weights/Adam state. Microbatch is one prompt; 32 prompts accumulate per optimizer step. This is full fine-tuning. CPU offload is synchronous and can be slower, but avoids requiring the optimizer state to fit the student's 24 GB GPU. Start with the short pilot below before using the 1,024-prompt + 1,024-response token budget.

## Prepare training data

Dolly is the default runnable instruction dataset. MetaMathQA is the provided mathematics alternative. These are controlled comparisons on shared data, not exact reproductions of every paper's data recipe.

```bash
# First-time setup only; the default Dolly output already exists and can be reused.
# Use a new --output-dir when preparing another split.
python scripts/prepare_data.py --dataset dolly
# Alternative mathematical corpus; see datasets.md for held-out exclusions:
python scripts/prepare_data.py --dataset metamathqa
```

Preparation groups duplicate/augmented questions before splitting, removes exact duplicate rows, reports empty records, and writes canonical text JSONL plus hashes. The already prepared Dolly artifact has 14,246 training and 750 held-out records. It uses a new split of the canonical 15k source; do not evaluate that run on the overlapping legacy `dolly/valid.jsonl` as if it were unseen. To reserve an external evaluation source, prepare a *new* directory with `--exclude-source` before training.

## Short pilot and full training

A KD pilot with the actual Qwen pair:

```bash
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_kd.json \
  --dataset dolly_pilot --output-root /nas/Users/wyx/Baseline/pilots \
  --max-steps 2 --gradient-accumulation-steps 1 \
  --max-prompt-tokens 128 --max-new-tokens 32
```

`--max-steps` defines the complete schedule budget, so use the same value when resuming. The data paths in the config stay Dolly; `--dataset` here names the separate pilot output. Replace `kd` with `abkd` or `skd` for those pilots. SKD generates fresh current-student proposals, so generation time is part of training cost.

DistiLLM-2 needs pairs first. Freeze the raw split, generate teacher/student responses once, then train from exact saved token IDs:

```bash
python scripts/generate_pairs.py \
  --config configs/experiments/qwen3_8b_1p7b_distillm2.json \
  --input /nas/Users/wyx/Baseline/data/dolly/train.jsonl \
  --output /nas/Users/wyx/Baseline/data/dolly/qwen3_8b_1p7b/pairs.train.jsonl
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_distillm2.json
```

Pair generation loads each model sequentially and records checkpoint identities, source/validation hashes, sampler settings, exact prompt/response token IDs and costs. Pilot generation can use `--max-examples`, `--max-prompt-tokens` and `--max-new-tokens`; use the same token limits in the corresponding training command. The trainer rejects mismatched producer checkpoints or tokenization/generation settings. Pair files are immutable; `--resume` verifies and reuses matching completed pairs or continues per-example progress. Choose a new output name when changing generation inputs.

For the ordinary KD/ABKD/SKD full runs:

```bash
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_kd.json
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_abkd.json
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_skd.json
```

Use `llama3_8b_llama32_1b_*.json` for the Llama group. Its teacher is the supplied original Llama-3-8B-Instruct, and `check_environment.py --pair llama3` checks the shared-token contract.

Preview or execute a matrix sequentially:

```bash
python scripts/run_matrix.py --pair qwen3_8b_1p7b --seeds 42 43 44
python scripts/run_matrix.py --pair qwen3_8b_1p7b --seeds 42 43 44 --execute
```

DistiLLM-2 uses the same fixed initial-model pair artifact across training seeds by default; the producer seed is recorded separately. Complete pair generation before executing the matrix. `--dataset metamathqa` changes matrix data paths, and pairs must first be generated from that corpus with `--validation-file /nas/Users/wyx/Baseline/data/metamathqa/validation.jsonl`.

## Resume and distributed use

```bash
python scripts/train.py --config configs/experiments/qwen3_8b_1p7b_kd.json \
  --resume /nas/Users/wyx/Baseline/runs/qwen3_8b_1p7b/dolly/kd/seed_42/checkpoints/step_000100
```

Use the latest completed checkpoint and the exact original effective settings, runtime, dataset hashes, model identities and world size. Foreign-run/stale checkpoints are rejected. Uncommitted logs and incomplete checkpoint directories after a crash are backed up before replaying the last completed step.

For two student DDP ranks on four GPUs (students 0/1, teachers 2/3), retaining 32 global prompts per update:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/experiments/qwen3_8b_1p7b_kd.json \
  --teacher-device paired --gradient-accumulation-steps 16 \
  --output-root /nas/Users/wyx/Baseline/ddp
```

This profile replicates the teacher and shards examples; it does not shard either model. It requires two GPUs per rank. CPU distributed correctness tests do not certify the full-size NCCL profile; check the validation record for actually executed GPU evidence.

Natural-language evaluation now uses generated-response ROUGE-L F1. Default training performs no validation; the separate evaluation command scores all held-out records. Optional training validation uses the same metric and selects the highest score. Keep evaluation-only datasets out of training. A changed dataset, sequence budget or method parameter should use a distinct output root/dataset identifier so an existing run is not mixed with it.
