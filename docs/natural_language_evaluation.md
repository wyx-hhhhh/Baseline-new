# Evaluate the five local natural-language benchmarks

`scripts/evaluate_all.sh` now evaluates saved baseline students against the
downloaded datasets under `/nas/Datasets`. It prepares the evaluation inputs
automatically and reports **ROUGE-L F1 separately for every benchmark**. No new
training or dataset download is required. Run evaluation after the corresponding
training jobs release their GPUs; existing GPU locks prevent conflicting jobs.

## Commands

Evaluate all four methods for both model families:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh
```

The Qwen queue uses physical GPU 0 for student inference within the 0,1 group.
The Llama queue uses physical GPU 2 within the 2,3 group. Evaluation loads only
the saved student. Each family evaluates KD, ABKD, SKD, then DistiLLM-2; each
saved student is evaluated on all five benchmarks before advancing.

To launch the groups separately in two terminals:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh --group qwen --qwen-gpus 0,1
bash /home/wyx/Baseline/scripts/evaluate_all.sh --group llama --llama-gpus 2,3
```

Rerun exactly the same command to resume. Each completed response is saved
durably. Verified complete evaluations reuse their results without loading the
model. Ctrl-C once saves the current response before stopping.

To inspect the full queue without loading models or using a GPU:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh --dry-run
```

`--dry-run` also prepares or verifies the local CPU data cache. `--prepare-only`
prepares the inputs without checking for trained models. The full default queue
requires eight completed student exports. At implementation time KD and ABKD
had completed for both families; SKD and DistiLLM-2 were not yet complete. Once
their GPU group is free, completed methods can be selected explicitly:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh --methods kd abkd
```

## Inputs and overlap handling

| Benchmark | Exact local input | Raw rows | Evaluation rows |
|---|---|---:|---:|
| DollyEval | `DollyEval/valid.jsonl` | 500 | 75 |
| SelfInst | `SelfInst/valid.jsonl` | 242 | 242 |
| Super-Natural | `Super-Natural/{0_2,3_6,6_10,11_}/valid.jsonl` | 8,354 | 7,623 |
| Unnatural | `Unnatural/{0_2,3_5,6_10,11_}/valid.jsonl` | 64,809 | 64,809 |
| VicunaEval | `VicunaEval/valid.jsonl` | 80 | 80 |

Counts reflect the installed sources and current canonical Dolly training pool.
All listed length bins are included. Super-Natural's `3_6` and `6_10` JSONL
files are byte-identical; merging their 731 exact duplicate examples avoids
double-counting. Only exact prompt/reference tuples are merged. Different
references for the same prompt remain separate source examples; the retained
record lists every merged source file, row and bin.

**425 of the 500 DollyEval records overlap the current Dolly training prompts.**
These are excluded, leaving 75 records. Report the resulting score as
**DollyEval, training-overlap-filtered, n=75**. It is not a score on the full
downloaded 500-row split. The other four benchmarks have no detected overlap
with this training pool. Overlap checking uses Unicode/whitespace-normalized,
case-folded prompts; it does not detect every possible semantic paraphrase.

The shared exclusion source defaults to
`/nas/Users/wyx/Baseline/data/dolly/train.jsonl`. Both model groups use the same
prepared membership. Every selected model's actual recorded training file is
also checked before inference, including DistiLLM-2's paired prompt records.
A custom training corpus with extra overlaps requires adding its canonical file
with repeatable `--exclude-training-data PATH`; it is never silently scored as
held-out. All reference and overlap checks run before a requested pilot limit.

The adapters build the raw user turn from `instruction` and optional `input`,
then apply the saved model's chat template once. They preserve complete output
references, including every answer in Super-Natural's output lists. The existing
Alpaca-style `prompt`, `.txt` copies and DollyEval `raw.jsonl` training file are
not additional evaluation examples.

## Metric and generation

Each example receives one greedy generated continuation. Its ROUGE-L F1 is the
**maximum over all its complete reference answers**. The benchmark score is
the arithmetic mean of these example scores, multiplied by 100. Single-reference
examples use ordinary ROUGE-L F1. Scores are not ROUGE-Lsum and no teacher or
LLM judge is loaded. All five datasets default to the same **Unicode ROUGE-L**
tokenizer: NFKC/case-folded alphanumeric words, individual CJK/kana/Hangul
characters, and no English stemming, using the existing pinned
`rouge-score==0.1.2` scorer. Unnatural includes 15 non-English references that
the old English-only validator rejects, so Unicode mode allows all rows to be
evaluated under one common tokenization rule. Report this tokenizer with the
scores; they are not interchangeable with English/Porter-stemmed ROUGE-L.

References consisting entirely of punctuation or emoji can produce no ROUGE
tokens and score zero even for an identical completion. They remain in the
denominator. English mode remains an explicit option for compatible benchmark
subsets; full Unnatural requires Unicode mode under the strict reference checks.

The defaults allow 1,024 prompt tokens and 1,024 generated response tokens.
Qwen non-thinking mode and the saved Llama vocabulary/fixed-date policy are
retained. Evaluation refuses to truncate a prompt. The
[full local token audit](evidence/nl_token_budget_audit.json) checked all 73,985
raw rows with both real tokenizers: the longest prompts were 524 Qwen tokens
and 355 Llama tokens, so all inputs fit. References are never truncated;
generated responses may reach the configured output budget.

There are **40 separate evaluations** in a complete default run: two model
families × four methods × five datasets. Each model generates 72,829 responses,
including the full 64,809-row Unnatural set. Results stay separate by benchmark;
the script does not create an average across differently sized datasets.

## Storage and provenance

The automatic data cache is under
`/home/wyx/Baseline/artifacts/nl_eval_data/dolly/<request-id>/`.
Each benchmark has a canonical JSONL file. The accompanying `manifest.json`
records source and training-exclusion hashes, raw/deduplicated/excluded/scored
counts, excluded records and selected IDs. Raw `/nas/Datasets` files remain
unchanged. Existing complete caches are verified before reuse; changed or
partial artifacts require a new `--prepared-data-dir`.

Models are read from the existing run layout:

```text
/nas/Users/wyx/Baseline/runs/<pair>/dolly/<method>/seed_42/final/
```

Results use a separate natural-language evaluation tree:

```text
/nas/Users/wyx/Baseline/evaluations/nl/dolly/<protocol-id>/
  <pair>/<method>/seed_42/<benchmark>/
    predictions.jsonl
    progress.json
    metrics.json
    manifest.json
  summary.csv
  summary.json
/nas/Users/wyx/Baseline/evaluation_orchestration/nl/dolly/<protocol-id>/
  configs/
  logs/<qwen|llama>/
  qwen.status.json
  llama.status.json
```

Predictions record generated text/token IDs, every reference, per-reference
ROUGE-L and the best-reference index. Completion manifests bind model metadata
and weight size/mtime, source data, training configuration, evaluation settings,
runtime and relevant code hashes. Resuming revalidates cached text and scores.
Different generation settings/data pools have separate protocol directories;
group/method/seed/benchmark selections and pilot limits have separate summaries.

## Options and prior validation workflow

- `--benchmarks DollyEval SelfInst Super-Natural Unnatural VicunaEval` selects
  downloaded benchmarks; canonical names `dolly selfinst super_natural unnatural
  vicuna` are also accepted.
- `--limit 10` runs an explicit ten-example pilot **per model and per benchmark**,
  with separate output paths. Omit it for complete evaluation.
- `--dataset` / `--training-dataset`, `--output-root`, and `--data-root` identify
  the trained campaign and its canonical training data.
- `--source-root` overrides `/nas/Datasets`; `--evaluation-root` changes the
  result root; `--prepared-data-dir` selects another prepared-data cache.
- `--seeds`, `--methods`, `--group`, `--dtype`, `--max-prompt-tokens`,
  `--max-new-tokens` and `--rouge-tokenizer` are available. Changing generation
  settings creates a distinct evaluation protocol. The current defaults use
  final students; the original training manifest is preserved.

The original evaluation on the 750-row internally held-out Dolly validation
split is still available; put `--validation-only` first:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh --validation-only
```

This dispatches to the existing `scripts/run_evaluation.py`, with its original
options and result paths. Math evaluation remains
`scripts/evaluate_math_all.sh` with its separate correctness grader.

Implementation files are `baseline_common/nl_data.py`,
`baseline_common/nl_evaluation.py`, `scripts/evaluate_nl.py` and
`scripts/run_nl_evaluation.py`. The standalone evaluator accepts a saved config,
model and canonical benchmark directly; `--help` lists its arguments. CPU tests
cover adapters, multiple references, leakage checks, native generation,
interruption/restart and campaign planning. Full GPU evaluations were not
launched as part of this update.

Verification passed **102 new CPU tests**: [23 adapter cases](evidence/nl-data-tests.txt),
[54 evaluator cases](evidence/nl-evaluator-tests.txt), and
[25 campaign cases](evidence/nl-campaign-tests.txt). The
[existing evaluation regression suite](evidence/nl-shared-evaluation-regression.txt)
also passed 104 tests. A [read-only production preflight](evidence/nl_production_preflight.json)
verified the four completed KD/ABKD exports against all five prepared datasets,
confirmed six recorded Dolly training configurations still match their source
files, and checked the prior math campaign's code pins. The
[reference audit](evidence/nl_reference_audit.json) validates every retained
reference in Unicode mode and identifies all 74 examples with no reference
tokens (one SelfInst and 73 Unnatural).
