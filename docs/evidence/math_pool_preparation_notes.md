# Fixed MetaMathQA pool preparation

`scripts/prepare_math_data.py` prepares one shared pool for KD, ABKD, SKD and
DistiLLM-2. The default pool contains exactly 50,000 training rows and 5,000
validation rows, using seed 42. Both model groups consume these same files and
record IDs. Pair generation must use the frozen training file as its input.

The authoritative artifact is
`/nas/Users/wyx/Baseline/data/metamathqa_50k_v1/manifest.json` after preparation.
The manifest contains source hashes and observed source counts, all selected
record IDs, output hashes, filtering counts and group-selection counts.

The published policy-2 pool contains 50,000 training rows spanning **11,732
source-question families**, and 5,000 validation rows spanning **199 families**.
Its source scan read 395,000 MetaMathQA rows, removed four empty-text records and
8,956 exact duplicate prompt/response pairs, and found zero normalized or linked
test overlaps. There were 386,040 usable rows in 13,928 independent families
before quota selection. The reserved validation families contained 5,023 rows;
their 23 unselected rows do not enter training.

## Input and output

| Role | Default source | Output relative to pool |
| --- | --- | --- |
| Shared MetaMathQA source | `/nas/Datasets/MetaMathQA` (`train` saved Arrow split) | `train.jsonl`, `validation.jsonl` |
| Full GSM8K test | `/nas/Datasets/GSM8K/main/test-00000-of-00001.parquet` | `tests/gsm8k.jsonl` |
| Full competition MATH test | `/nas/Datasets/hendrycks_competition_math/data/test/0000.parquet` | `tests/math.jsonl` |

The MATH source is the competition mathematics dataset, not the unrelated
coding dataset stored under `/nas/Datasets/MATH`. Preparation requires exactly
1,319 GSM8K and 5,000 MATH test rows. Test records are not sampled or deduplicated;
an empty prompt/reference or an unexpected row count aborts preparation.
References retain their complete GSM8K `####` and MATH boxed solutions, including
MATH type and difficulty metadata. Separate mathematical-answer evaluation uses
these canonical test files.

## Leakage prevention and exact quotas

1. Normalize question text with Unicode NFKC, whitespace collapse and casefold.
2. Remove training records with empty prompt/response text. Deduplicate exact
   prompt/response pairs, merging their original-question links. Stable training
   IDs hash the complete, untruncated prompt/response text.
3. Build connected components over both normalized queries and original
   questions in one shared namespace. A query matching another record's original
   question therefore joins the same component. Identical prompts with different
   solutions also belong to the same component.
4. Remove every component touching either benchmark's test questions before
   sampling. Counts distinguish direct overlaps from additional family members
   excluded through a transitive connection. Overlap counts are counts of unique
   usable prompt/response records after exact deduplication.
5. Rank families deterministically using the seed and family hashes. Reserve
   whole families for validation while retaining enough independent records for
   training. Globally rank all rows outside those reserved families by a seeded
   record-ID hash and select exactly 50,000 training rows. Separately hash-rank
   every row in the reserved validation families and select exactly 5,000
   validation rows. Unused validation-family rows stay unused; they never move
   into training.

The training sample is uniform per row conditional on the validation-family
reservation, implemented by deterministic hash ranking. It covers a broader set
of source questions than filling whole training families until reaching the row
quota. Reordering MetaMathQA source rows leaves selected data unchanged.
Preparation fails if available records or independent families cannot satisfy
the requested quotas. No token truncation is performed by data preparation;
training and generation token limits are configured separately.

This exclusion detects normalized text matches and linked augmentation
families. It does not establish the absence of every semantic paraphrase of a
test question when that paraphrase has no matching original-question link.

## Publication and reuse

```bash
cd /home/wyx/Baseline
.venv/bin/python scripts/prepare_math_data.py --resume
```

The command exclusively creates its output directory and writes the manifest
last as a completion marker. It never overwrites an existing directory.
`--resume` reuses a complete pool only after verifying request parameters, source
file hashes, output hashes, selected IDs and counts, and train/validation/test
isolation. A changed source, modified artifact, incompatible seed or size, or
partial output is rejected; use a new output directory for a different pool.
The final manifest records `selection_policy_version: 2`. Earlier draft pools
without that policy version are rejected rather than silently reused.

`baseline_common.math_data.validate_math_pool(path)` validates published artifacts
without rereading the raw source. Preparation additionally verifies source hashes
before and after reading them, and on every resume.

## Targeted verification

`tests/test_math_data.py` exercises exact quotas, family isolation, row-order
invariance, seed changes, raw-query and original-question exclusions, transitive
family exclusions, duplicate metadata links, complete test preservation, source
and artifact tampering, partial-output rejection, insufficient independent data,
saved Arrow/parquet split selection, broad sampling across training families,
and rejection of an earlier selection policy.

The preparation change passed this command on the prepared `.venv`:

```bash
.venv/bin/python -m pytest -q tests/test_math_data.py tests/test_data.py
```

Result: **22 passed**. This fixture validation does not substitute for the real
pool's source counts and hashes; those are recorded by its published manifest.
