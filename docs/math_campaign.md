# Fixed MetaMathQA campaign

The campaign is prepared and has not been launched. Run it after the current
Dolly jobs release their GPUs. The launchers never schedule an automatic
transition from Dolly. Their shared GPU locks reject conflicting baseline jobs.

All four methods use the same fixed 50,000 MetaMathQA training examples. Each
method starts a fresh student from the original Instruct checkpoint, trains
once, and evaluates that saved student separately on GSM8K and MATH. The default
seed is 42: eight training runs and sixteen benchmark evaluations in total.

## Commands

To launch each model group in its own terminal:

Qwen3, physical GPUs 0 and 1:

```bash
bash /home/wyx/Baseline/scripts/run_math.sh --group qwen --qwen-gpus 0,1
```

Llama3, physical GPUs 2 and 3:

```bash
bash /home/wyx/Baseline/scripts/run_math.sh --group llama --llama-gpus 2,3
```

Each group runs **KD → ABKD → SKD → DistiLLM-2** sequentially. DistiLLM-2's
teacher/student response pairs are generated automatically before its training
stage. Both groups may run concurrently. Within each group the student occupies
the first GPU and the frozen teacher the second; these are not two data-parallel
student replicas.

After both training groups finish, evaluate all saved students:

```bash
bash /home/wyx/Baseline/scripts/evaluate_math_all.sh
```

Alternatively, the complete workflow needs just two commands. The first starts
both training groups with the same GPU assignments, and the second runs both
evaluation queues after training has finished:

```bash
bash /home/wyx/Baseline/scripts/run_math.sh --qwen-gpus 0,1 --llama-gpus 2,3
bash /home/wyx/Baseline/scripts/evaluate_math_all.sh --qwen-gpus 0,1 --llama-gpus 2,3
```

Rerun the same command after an interruption. Completed training is skipped;
interrupted training resumes its optimizer, scheduler and RNG state from the
latest complete checkpoint. Pair generation resumes its saved records.
Evaluation reuses verified saved predictions and continues missing questions.
Use Ctrl-C once for a graceful save; an abrupt process or machine failure may
require replaying training since the latest completed checkpoint. Persistent
jobs should run inside your usual terminal multiplexer.

Both launchers accept `--dry-run` to print the plan without model loading or
training. The evaluator also accepts `--group qwen` or `--group llama` when only
that group's four students have finished. Subset evaluations have separate
summary filenames. `--max-steps` is a training pilot option; use a separate
`--output-root` for a pilot. `--limit` is an evaluation pilot option and writes
separate subset outputs. The default commands use the full training/test sets.

## Frozen data and common settings

The authoritative campaign is [configs/math_campaign.json](../configs/math_campaign.json).
Its data lives at `/nas/Users/wyx/Baseline/data/metamathqa_50k_v1/`.

| Split | Records | File relative to the pool |
|---|---:|---|
| MetaMathQA training | 50,000 | `train.jsonl` |
| MetaMathQA development | 5,000 | `validation.jsonl` |
| GSM8K full test | 1,319 | `tests/gsm8k.jsonl` |
| Competition MATH full test | 5,000 | `tests/math.jsonl` |

The selected IDs, source hashes and output hashes are fixed in `manifest.json`.
Preparation removes empty records and exact duplicates, checks normalized
queries and linked original-question families against both benchmark test sets,
and keeps development families out of training. The prepared pool has zero
overlap under these checks; this is not a claim to detect every unlinked semantic
paraphrase. Details and reproducible preparation commands are in the
[data preparation record](evidence/math_pool_preparation_notes.md).

The six KD/ABKD/SKD configurations read the identical training file. Both
DistiLLM-2 pair generators use all of that same file's IDs; generated responses
are method-specific artifacts derived from the common prompt pool. No method
trains on either benchmark's test split. Development data is reserved but is not
used to select a checkpoint in this training-only campaign; evaluation uses the
final student after the fixed training budget.

| Control | Common setting |
|---|---|
| Qwen teacher → student | `/nas/Models/Qwen3-8B-Instruct` → `/nas/Models/Qwen3-1.7B-Instruct` |
| Llama teacher → student | `/nas/Models/Meta-Llama-3-8B-Instruct` → `/nas/Models/Meta-Llama-3.2-1B-Instruct` |
| Initialization | Original Instruct checkpoints for every method; no additional SFT stage |
| Adaptation | Full student fine-tuning; frozen teacher |
| Precision and memory | BF16, FP32 CPU AdamW state/master weights, gradient checkpointing |
| Optimizer | LR 1e-5, weight decay 0.01, gradient clip 1.0 |
| Training budget | One epoch, microbatch 1, accumulation 32: 1,563 optimizer updates per method |
| Schedule | Cosine decay, 3% warmup |
| Token budgets | 2,048 prompt tokens and 2,048 response tokens |
| Checkpointing | Every 100 optimizer steps, completion, and graceful interruption |
| Training-time evaluation | Disabled |
| Evaluation decoding | One greedy response per question; up to 2,048 new tokens |

The [token audit](evidence/math_token_budget_audit.json) scanned every selected
training/development reference and every test question with both real
tokenizers. All fit these budgets. The evaluator refuses to truncate a test
question; generated responses can still reach their fixed output limit.

Method-specific controls preserve the existing implementations: KD/ABKD use
reference prefixes, ABKD uses alpha 0.1 and beta 0.8, SKD uses acceptance-k 25
and proposal blocks of five tokens, and DistiLLM-2 uses one-time response pairs
and fixed skew coefficients 0.1/0.1. Sampling settings are shared within each
model family. Qwen uses non-thinking mode; Llama retains the explicit shared
vocabulary policy. These controls define a common local comparison. The
[literature audit](math_baseline_literature.md) explains differences from the
papers, including DistiLLM-2's original extra SFT and adaptive/refreshed recipe.

## Evaluation and artifacts

Math evaluation reports final-answer accuracy/pass@1 as a percentage separately
for each benchmark. GSM8K uses exact numeric equivalence; MATH uses the pinned
symbolic verifier. Invalid predictions remain in the denominator. Sixteen MATH
gold annotations have explicit, hashed interpretations for multiple answers,
alternatives or unusual box notation. All 5,000 original questions are retained.
The complete [grading protocol](math_grading.md) documents these decisions;
this metric is a controlled comparison under that protocol.

The campaign pins both the grading source and its full gold-audit artifact.
Each evaluation also records code, dependency, model, input and generation
identity, and verifies those identities before resuming. Evaluation reads the
completed run's recorded training configuration. Natural-language evaluation
continues to use ROUGE-L through the existing Dolly entry points.

Math outputs have their own root, `/nas/Users/wyx/Baseline/math/`:

```text
runs/<pair>/metamathqa_50k_v1/<method>/seed_42/
  manifest.json
  checkpoints/step_XXXXXX/
  final/
  result.json
pairs/<pair>/pairs.train.jsonl
orchestration/metamathqa_50k_v1/          training logs and status
evaluation_orchestration/metamathqa_50k_v1/
evaluations/<pair>/metamathqa_50k_v1/<method>/seed_42/<gsm8k|math>/
  predictions.jsonl
  metrics.json
  manifest.json
evaluations/metamathqa_50k_v1/summary.csv
evaluations/metamathqa_50k_v1/summary.json
```

The full summary has sixteen rows, retaining a separate GSM8K and MATH row for
every saved student. Repeating a completed evaluation verifies and reuses its
artifacts without loading the student again.

## Environment and verification

Training uses the existing `.venv`. Evaluation uses the prepared isolated
`.venv-math-eval`, with the training stack plus Math-Verify 0.9.0,
latex2sympy2_extended 1.11.0, ANTLR runtime 4.13.2 and SymPy 1.13.1. The full
hashed [evaluation lockfile](../requirements-math-eval.lock) and
[installed inventory](environment-math-eval-freeze.txt) record the environment.
The live training environment was not changed to install these dependencies.

Verification includes all 6,319 benchmark gold answers, adversarial grader
cases, real tiny CPU training for all eight campaign runs, both pair producers,
all sixteen evaluation stages, interruption/restart and completed-run skipping.
See [campaign test results](evidence/math-campaign-tests.txt),
[data/evaluation tests](evidence/math-data-evaluation-tests.txt),
[Dolly regression tests](evidence/math-dolly-regression.txt), and the
[independent review](evidence/math_campaign_review.md).
Full-size math training has not been launched; these CPU checks do not measure
its GPU memory requirements or throughput at the new token budget.
