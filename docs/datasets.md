# Training datasets and preparation

The new experiments compare SKD, ABKD, DistiLLM-2, and forward KL KD on the **same prepared training prompts and held-out validation split**, separately for each requested model pair. Dolly is the default instruction-following corpus. MetaMathQA is an optional separate math experiment; GSM8K training data is another supported option. Changing the dataset changes the experiment and should use a different output directory and configuration. These are controlled adaptations to the supplied instruct checkpoints, not reproductions of each paper's original model/data recipe.

## What the original repositories actually load

| Baseline | Training data evidenced in this checkout | Existing local data and missing reproduction artifacts |
|---|---|---|
| SKD | `speculative_kd/speculative_kd/train/ddp_skd.py`, lines 330–420, loads custom `data/gsm_1k_train.json`, `gsm_100_train.json`, `Math_CoT_train.json`, `math_train_1k.json`, `summ_1k_train.json`, `summ_100_train.json`, `mt_1k_train.json`, and `mt_100_train.json`, with matching validation files. The GSM SFT config points at a custom `gsm_1k` prefix. The summarization SFT config **named `sft_config_samsum.yaml` actually loads `knkarthick/dialogsum`**. Math and translation SFT configs use custom `Math_CoT` and `trans_1k` files. | GSM8K is available for a new split, but the exact released/custom 100/1,000-example subsets and split seeds are absent. DialogSum and the custom summarization/translation/math artifacts are absent. The later paper audit identifies **UltraInteract** as SKD's intended math-instruction corpus; the checkout still lacks the exact extraction/row manifest for `Math_CoT`. Translation corpus provenance remains unresolved. Do not relabel MetaMathQA as SKD's original math data. See [the literature audit](math_baseline_literature.md). |
| ABKD, language-model branch | `abkd/distillation_llm/scripts/gpt2/ab/train_0.1B_1.5B.sh` and `scripts/openllama2/ab/train_3B_7B_teacher_lora.sh` use `processed_data/dolly/full/...`. The Qwen script `scripts/qwen/ab/train_3B_7B_teacher.sh` uses **`processed_data/metamath/pseudo/qwen/`**. `tools/process_data_metamath.py` maps `query,response` into instruction/response data, and the processing shell script points at the author's local MetaMath directory. | Both Dolly and MetaMathQA are available. The author's exact teacher-generated MetaMath pseudo-responses/checkpoint are not present; using original MetaMathQA responses in this shared runner is an explicit protocol change. |
| DistiLLM-2 | `distillm-2/training_configs/qwen2.5-1.5b-sft.yaml` uses **`HuggingFaceH4/ultrachat_200k`, splits `train_sft`/`test_sft`**, for optional SFT. The README describes UltraChat generation, but **the actual `generate/generate_vllm.py` default `--data_dir ultrachat` loads `UCLA-AGI/SPIN_iter1`** (or `SPIN_iterN` through `--iter`), using the first message in each `generated` conversation as the prompt. Distillation training configs point to a user-generated paired dataset. Other configs cover code and vision tasks. | Neither UltraChat-200k nor SPIN iteration datasets is in the supplied local inventory. To reproduce this code's prompt recipe, prepare the relevant SPIN iteration data; prepare UltraChat-200k too if reproducing its SFT stage. They are **not required** for the new Dolly/MetaMath controlled comparison, which generates teacher/student responses locally. |
| KD | `distillm/scripts/gpt2/kd/kd_base.sh` and `scripts/openllama2/kd/kd_3B_7B_teacher_lora.sh` use `processed_data/dolly/full/...`; the parallel ABKD KD scripts do the same. KD itself does not prescribe a dataset. The shared runner implements teacher-to-student forward KL on reference response prefixes. | Dolly is available. No extra training corpus is needed for this baseline in the common experiment. |

The older ABKD/DistiLLM READMEs also mention **OpenWebText** for an optional plain-text/pretraining path. It is absent from the provided inventory and is not needed by this runner. The DistiLLM-2 README names AlpacaEval, Evol-Instruct, and UltraFeedback as evaluation corpora; these are separate from training. Do not substitute evaluation examples as training prompts.

The legacy DistiLLM-2 `generate/reformat.py` defines its `test` split as the first 500 **training** rows and silently ignores failed prompt joins. The new workflow does neither: split before generation, strictly join by record ID, and retain a separate raw validation file.

## Local sources inspected

All paths below are under `/nas/Datasets`; the inspection reads metadata/schema/counts without printing raw examples.

| Dataset | Local representation | Use in the common runner |
|---|---|---|
| `databricks-dolly-15k` | `databricks-dolly-15k.jsonl`; `instruction,context,response,category` | Default source. Context is included in the user prompt. |
| `dolly` | `raw.jsonl`, `valid.jsonl`; MiniLLM-style `instruction,input,output,prompt` | Alternative source matching the older repositories more closely. The old rendered `prompt` is rebuilt from instruction/input, then the actual model's chat template is applied once. Keep `valid.jsonl` held out. |
| `MetaMathQA` | HF `save_to_disk` DatasetDict: `dataset_dict.json`, `train/state.json`, `train/data-00000-of-00001.arrow`; `query,response,original_question,type` | Supported directly. Read only state-listed Arrow files, not extra `cache-*.arrow` copies. Group augmentations sharing `original_question` in one split. |
| `GSM8K` | `main/train-00000-of-00001.parquet`, `main/test-00000-of-00001.parquet`; equivalent JSON files under `json/`; also a separate `socratic` configuration | Supported; default selects **main/train only**. Main/test remains an evaluation set. Do not combine main and socratic copies accidentally. |
| `hendrycks_competition_math` | `data/train/0000.parquet` (7,500 rows), `data/test/0000.parquet` (5,000 rows); verified `problem,level,type,solution` | Generic `problem,solution` adapter can consume the training source. Reserve the test set. |
| `MATH` | `train.jsonl`, `test.jsonl`; inspected schema is **`id,question,solutions,input_output,difficulty,url,starter_code`** | Despite the directory name, these files have a coding-task schema, not competition-math `problem,solution`. Do not use them as a MATH benchmark or silently interpret `solutions` as math answers. Use the verified `hendrycks_competition_math` directory for competition math. |
| `Math10K` | `math_10k.json` and `data/train-00000-of-00001.parquet` | A possible additional instruction corpus if its selected file matches a supported schema. Not an original required artifact established by the four launchers above. |
| `DeepMath-103K`, `Infinity-Instruct`, `OpenMathReasoning` | Local parquet shards and derived math JSON files; some directories contain multiple domains or subsets. OpenMathReasoning currently has **34 of 144** named CoT shards. | Optional new training tasks. OpenMathReasoning is an incomplete download, not a verified full corpus. Select the intended training subset explicitly and verify its response/conversation schema; no implicit joining of every shard/directory. These are not required by the default experiment. |
| `AIME-2024`, `AIME-2025`, `AMC23` | Competition benchmark data; some upstream benchmark splits are misleadingly named `train` | Keep as evaluation benchmarks. A file called `train` is not evidence it is appropriate for training. |
| `PIQA`, `WinoGrande`, `BoolQ`, `HellaSwag`, `MathQA`, `SVAMP` | Local task-specific JSON/JSONL files | Available evaluation/task data, but multiple-choice and structured schemas need explicit task formatting and metrics before use. Not automatically converted into instruction responses. |
| `SIQA`, `GPQA` | SIQA has an unextracted `raw/socialiqa-train-dev.zip`; GPQA has CSV variants (`gpqa_main`, `gpqa_diamond`, `gpqa_experts`, `gpqa_extended`) | Present but need archive extraction/CSV conversion and task-specific formatting before this JSONL runner can use them. |
| `PRM800K` | `phase1_train/test.jsonl`, `phase2_train/test.jsonl`, derived `all_train/test` files | Process-supervision data needs a separate step/label adapter. It is not a drop-in response corpus and is not used by these configs. |

The generic reader accepts local JSON arrays, JSONL, JSON objects with `instances`/`data`/explicit split arrays, parquet shards, and HF `save_to_disk` primitive-text Arrow data. Supported row mappings are `instruction` plus optional `input`/`context` and `output`/`response`; `query,response`; `question,answer`; `problem,solution`; `dialogue,summary`; canonical `prompt,response`; and exactly one user/assistant `messages` exchange. Multi-turn/system conversations are rejected until an explicit adapter preserves their semantics. No remote dataset is fetched.

## Prepare immutable splits

Run from `~/Baseline` with the experiment Python environment described in the runbook. Defaults write beneath `/nas/Users/wyx/Baseline/data`; override `--output-dir` for a local pilot. These commands create data only and do not start training.

```bash
# Recreate the default corpus in a fresh directory, with a new held-out split.
python scripts/prepare_data.py --dataset dolly --output-dir ./data/dolly-new

# Alternative protocol: also exclude the older MiniLLM validation prompts.
python scripts/prepare_data.py --dataset dolly \
  --output-dir /nas/Users/wyx/Baseline/data/dolly-with-exclusions \
  --exclude-source /nas/Datasets/dolly/valid.jsonl

# Optional math run, with known downstream evaluation questions excluded.
python scripts/prepare_data.py --dataset metamathqa \
  --exclude-source /nas/Datasets/GSM8K/main/test-00000-of-00001.parquet \
  --exclude-source /nas/Datasets/hendrycks_competition_math/data/test/0000.parquet

# GSM8K main training subset; never select the test shard for training.
python scripts/prepare_data.py --dataset gsm8k \
  --exclude-source /nas/Datasets/GSM8K/main/test-00000-of-00001.parquet

# Small workspace pilot; same deterministic algorithm, explicitly smaller corpus.
python scripts/prepare_data.py --dataset dolly --max-records 64 \
  --output-dir ./data/dolly-pilot
```

Each output contains `train.jsonl`, `validation.jsonl`, and `manifest.json`. Rows have `id,prompt,response`, with an optional hashed `source_group` for MetaMath augmentations. The default validation target is 5%, seed 42; connected duplicate/augmentation groups can make the exact percentage differ. Splitting is deterministic under source reordering. Unicode normalization, case folding, and collapsed whitespace define prompt equality; exact duplicate rows and empty prompt/response rows are removed and counted, conflicting IDs fail. Empty queries were observed in the actual local MetaMathQA corpus. Unsupported schemas and non-text values fail instead of silently disappearing. Evaluation-source exclusions also inspect MetaMath `original_question`. The manifest records source/exclusion file SHA-256 values, split hashes and counts, selection limits, seed, and normalization policy.

Output files are never silently overwritten. Use a new versioned directory when changing the source, split seed, or exclusion list, and update all compared configs together. The reader will not glob training and test shards together. An ambiguous directory with multiple formats fails and requires an explicit file/subdirectory.

The default corpus has already been prepared at `/nas/Users/wyx/Baseline/data/dolly`: 15,011 source rows, 15 exact duplicates removed, **14,246 training / 750 validation** rows. It uses the new common held-out split and does not reproduce MiniLLM's original 500-example validation split. Legacy MiniLLM validation prompts may therefore occur in its training set; do not evaluate that experiment on the legacy validation file. For that separate benchmark, use the alternative exclusion protocol above and update every compared config. The independent adapter verification used exclusions and 64-record pilots, so its counts/hashes in `data-verification.json` intentionally describe different artifacts.

Exact normalized matching and original-question grouping prevent those forms of contamination; they do not certify absence of semantic paraphrases or contamination inherited by pretrained checkpoints. Exclude every downstream evaluation source that matters to the experiment and retain that evidence in the manifest. None of the test/competition corpora is automatically mixed into training.

## Generate DistiLLM-2 pairs

Generate from the prepared **training** split. Keep each config's `validation_file` pointing to the separate canonical validation file. The DistiLLM-2 config's `train_file` should point to the resulting paired JSONL.

```bash
python scripts/generate_pairs.py \
  --config configs/experiments/qwen3_8b_1p7b_distillm2.json \
  --input /nas/Users/wyx/Baseline/data/dolly/train.jsonl \
  --output /nas/Users/wyx/Baseline/data/dolly/qwen3_8b_1p7b/pairs.train.jsonl
```

Repeat with `configs/experiments/llama3_8b_llama32_1b_distillm2.json` for the updated Llama-3 teacher / Llama-3.2 student pair, and use the `train_file` output location in that config. `--device cuda:0` selects a generation device for both roles; `--validation-file`, `--seed`, `--max-prompt-tokens`, and `--max-new-tokens` override those settings for a different prepared corpus or pilot. Use the same effective settings during training, since provenance checks reject mismatches. `--max-examples N` is only for a documented limited pilot. The tool loads and releases the teacher and student sequentially. It performs no gradient updates.

The chosen continuation is generated by the teacher and the rejected continuation by the initial student; these names are algorithm roles, not human quality labels. They are joined strictly by unique ID and identical prompt token IDs. Every pair contains `prompt_ids,chosen_ids,rejected_ids`, readable `chosen,rejected` strings, and teacher/student/tokenizer identities plus source/config hashes, seed, and decoding settings. Generation retains the exact native returned IDs, including stop tokens. Decoded text is never re-encoded for training. Qwen uses the shared chat renderer with `enable_thinking=False`, preserving its non-thinking assistant prefix in `prompt_ids`. Real token 65535 and IDs above 65535 remain ordinary integers.

Sidecars `*.teacher.jsonl`, `*.student.jsonl`, and `*.manifest.json` allow auditing the join and file hashes. The manifest includes the effective config and separate hashes of the input config and effective settings. Per-role model load time, generation time, prompt-token count, generated-token count, and device are stored in both the manifest and paired provenance (`offline_generation`); these offline costs must be included when comparing DistiLLM-2 with the online methods. Prompt/generated counts are not full attention-operation or FLOP counts. Generation rejects overlaps with configured validation prompts and refuses existing outputs. Checkpoint identity means hashed config/tokenizer metadata and weight-file size/mtime as specified by the model loader; it is not a full cryptographic hash of every weight byte.

## Resuming pair generation

`generate_pairs.py --resume` now validates and reuses complete artifacts or continues durable per-example teacher/student progress. The parallel launcher invokes this automatically. Every completed response is appended and synced before proceeding; an interrupted final JSONL tail is backed up before recovery. Settings affecting generation and raw source/model identities must match; unrelated training-only settings can differ. See [parallel runs](parallel_runs.md).
