# Independent math campaign review

Review performed on 2026-09-17 UTC. This review inspected the current source,
read the published NAS data, ran both shell launchers with `--dry-run`, and ran
seven focused CPU evaluation tests. It launched no GPU work or full training.

## Verified scope

The frozen [campaign](../../configs/math_campaign.json) selects eight training
configurations under `configs/math/`: KD, ABKD, SKD and DistiLLM-2 for each of
Qwen3 and Llama3. Both successful dry plans retain the order
KD → ABKD → SKD → DistiLLM-2, with a pair-generation stage before DistiLLM-2.
Qwen uses physical GPUs 0,1 and Llama uses physical GPUs 2,3. Every training
configuration assigns the student to logical `cuda:0` and the teacher to
logical `cuda:1`; the shared runner masks the physical devices per group.

All six non-paired configurations read the same `train.jsonl`. Both paired
producers read that same file; their outputs feed their respective DistiLLM-2
training stage. No method uses a different prompt subset by default. The pair
writer checks record IDs/counts, input and model provenance, output hashes and
teacher/student sidecars when resuming.

The defaults initialize all four Qwen students from
`/nas/Models/Qwen3-1.7B-Instruct`, with
`/nas/Models/Qwen3-8B-Instruct` as teacher. All four Llama students start from
`/nas/Models/Meta-Llama-3.2-1B-Instruct`, with
`/nas/Models/Meta-Llama-3-8B-Instruct` as teacher. None initializes from Dolly
outputs. All use one epoch, no explicit step cap, and the seed 42 default.

Both full benchmarks are evaluated independently for every final student:
16 evaluation stages, two distinct destinations per student, and no `--limit`
argument in the default evaluator command. Final-answer `math_accuracy` is
selected throughout math execution. The inherited `eval_rouge_tokenizer`
configuration field is inactive: math configurations disable in-training
evaluation, and the separate math inference/aggregation path computes no ROUGE.

## Published pool verification

Called `validate_math_pool` on the real pool, recomputed all hashes/counts, and
independently compared normalized query/original-question keys and source
families. The pinned manifest SHA256 is:

`01c2e37cb1e3d33facf4336400e611f489b64e736b9ad83271f59d1a5fea3bcb`

Pool directory: `/nas/Users/wyx/Baseline/data/metamathqa_50k_v1`.

| Artifact | Records | Actual SHA256 |
|---|---:|---|
| `train.jsonl` | 50,000 | `1135c79fdc52b469b4036c6ab58528528dc3f7f1744eae1c904f996e84718acd` |
| `validation.jsonl` | 5,000 | `b1c6ffae13c47f95c36b7676081b1ee43318f9fa22f5ba97f180d2323b4c6513` |
| `tests/gsm8k.jsonl` | 1,319 | `4c4e33e816639a02939d3d189041ddecbd6ee3f750845274104c9ebf44229288` |
| `tests/math.jsonl` | 5,000 | `8051037ea52aa75fd53f5659120045be20e73b22cae6093b837ec356649ab1fd` |

All four train/development versus benchmark overlap counts were zero. Train
and development share zero `source_group` values. Training contains 11,732
source families; development contains 199. Source processing recorded 395,000
rows, four empty-text exclusions, 8,956 exact duplicates removed and 386,040
eligible rows. The connected-component policy excludes all linked families
touching normalized test queries before sampling. This is text/link exclusion,
not evidence that unlinked semantic paraphrases are impossible.

## Evaluation recovery and denominator

Read [math_evaluation.py](../../baseline_common/math_evaluation.py) and
[evaluate_math.py](../../scripts/evaluate_math.py). Cached rows are revalidated
against ordered source IDs, original questions/references, token IDs and freshly
computed grades. New rows append to the existing durable list. Completion
requires exactly one row per selected test question, and the final denominator
is the length of that complete list, including extraction/parsing failures.
Both generated and resumed rows contribute once. An invalid gold answer aborts
preflight rather than silently reducing the denominator.

Executed:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 .venv-math-eval/bin/python -m pytest -q tests/test_math_evaluate.py -k 'grade_aggregation_retains_unparseable or signal_stop_torn_tail_resume_and_fresh_run_match or complete_artifacts_require_hashes_and_valid_grades or invalid_gold_outside_limit'
```

Result: **7 passed, 33 deselected** in 4.33 seconds. This covers actual tiny CPU
generation across interruption/torn-tail recovery, equality to a fresh run,
invalid-gold preflight, unparseable-answer inclusion and completed-artifact
hash/grade validation. It does not establish large-model throughput or memory
requirements.

## Dolly isolation evidence

For each of the six existing Dolly KD/ABKD/SKD run manifests, the saved
`config` dictionary exactly matches the corresponding current resolved
`configs/experiments/*.json` dictionary. No DistiLLM-2 Dolly run manifest exists
yet to make that historical comparison. Current Dolly configs still select
`dataset: dolly` and `evaluation_metric: rouge_l`; math paths use the separate
`/nas/Users/wyx/Baseline/math` root. This checkout is not a Git repository, so
this evidence is a comparison with recorded runs, not a claim based on Git
history.

## Findings resolved on recheck

1. **Evaluation configuration provenance — resolved.** The current
   `build_math_groups` reads and validates `run/manifest.json`, verifies the
   recorded experiment identity, model paths, data paths/hashes and common
   controls, then derives completion steps from the recorded configuration.
   It passes that recorded configuration into both benchmark stages. A pilot
   can therefore be evaluated without restating its training step cap. The
   native CPU campaign test explicitly verifies this behavior and rejects
   mutated recorded pair, method, dataset, seed, initialization and learning
   rate. Source inspection confirms the fix remains in the current builder.
2. **Subset summary collision — resolved.** The current CLI appends the
   requested group, nondefault methods, nondefault seeds and pilot limit to
   summary filenames. Invoked the actual `main()` with its workers and summary
   writer replaced only in process memory, confirming six unique output names:
   `summary`, `summary_qwen`, `summary_llama`, `summary_methods_kd`,
   `summary_seeds_7` and `summary_first_10`. No workers or output files were
   created by this check. Separate group evaluations preserve each other's
   summaries and the default complete-campaign summary.

## Frozen grader and audit recheck

Read the latest `check_grader` implementation. It runs when building evaluation
stages and before each evaluation child. It requires the grader source SHA256
and gold-audit artifact SHA256 pinned in `configs/math_campaign.json`.
`check_grader` passed for the real campaign. Replacing either expected hash
with zeros in an in-memory copy raised the intended `ValueError`.

Independently verified these exact links against the current files:

| Link | Verified value |
|---|---|
| Campaign, grader and both audit metric implementations | `metamath-final-v2` |
| Campaign pin, actual grader source and audit's grader hash | `49c624abcbcbcd8e54412c822aeb7e647fda6d38d2e9785477dd6d3ebe33db17` |
| Campaign pin and actual gold-audit JSON | `d43b8416b8caf2364d7b2d6cafede0c91604518bd307d41fa1b553242a9e9d3d` |
| Audit producer source and audit's producer hash | `14a085677674183863b0534539d8a8d9a21efdd06e4b692060a5b2476dbe6b88` |

The audit's complete metric definitions equal the current runtime definitions,
including dependency versions. Both raw benchmark hashes and both canonical
test hashes still match the audited values. The artifact records all 1,319
GSM8K and 5,000 MATH references validated and correctly self-scored, all 6,319
unanchored rationales rejected, zero omitted rows, and original questions and
references preserved in order. It has no failures. Its producer was inspected:
it directly compares source/canonical questions and references, checks every
row, and verifies the 16 fixed MATH interpretation hashes and row positions.
This recheck did not rewrite or rerun that artifact.

The parent's `math-campaign-tests.txt` records 15 passing tests, including the
actual CPU campaign and recorded-config regression. At this recheck its test
file and grader hashes matched current source; its runner hash predates the
latest guard change. The current runner reviewed here has SHA256
`30ea420ea118b239f6483556ddd190a00c242163996b604fca0529103db0a814`.
The later guard and summary checks above were performed directly against this
source. Final test logs maintained by the parent may supersede that snapshot.

No unresolved concrete bug was identified in this bounded review. No GPU work,
production model generation, full training or NAS writes were performed.

## Final test evidence received after review

The final [campaign test log](math-campaign-tests.txt) supersedes the 15-test
snapshot above: **25 passed** in 154.86 seconds. It binds the current runner
SHA256 `30ea420ea118b239f6483556ddd190a00c242163996b604fca0529103db0a814`
and includes the full tiny CPU campaign plus ten additional missing/changed
grader/audit pin checks, including rechecking the audit between evaluation
children. The separate [data/evaluation run](math-data-evaluation-tests.txt)
passed **170 tests** against the same frozen grader.
