# Start the updated Llama experiments

The user selected **`/nas/Models/Meta-Llama-3-8B-Instruct`** as teacher. The student remains **`/nas/Models/Meta-Llama-3.2-1B-Instruct`**. All four Llama configs now begin `llama3_8b_llama32_1b_`; the previous `llama31_...` configs were replaced. Qwen configs are unchanged.

For simultaneous Qwen and Llama groups with automatic recovery, use [the one-command parallel launcher](parallel_runs.md). The individual commands below remain available for one-off runs.

## Environment to activate

The installed, verified training environment is the project virtual environment **`/home/wyx/Baseline/.venv`**. It does not require a Conda environment switch. Activate it from your current shell:

```bash
cd /home/wyx/Baseline
source .venv/bin/activate
python -c 'import sys, torch, transformers; print(sys.executable); print(torch.__version__, transformers.__version__)'
```

Expected interpreter: `/home/wyx/Baseline/.venv/bin/python`; PyTorch `2.6.0+cu124`; Transformers `4.51.3`; Python `3.11.14`. The shared Conda environments (`base`, `RAGAny`, `vllm`, etc.) have different package sets. The command above selects the tested project environment regardless of which Conda environment was previously active.

## Start KD now

```bash
python scripts/check_environment.py --pair llama3 --strict-pins
python scripts/train.py --config configs/experiments/llama3_8b_llama32_1b_kd.json
```

This starts full student fine-tuning on the prepared Dolly training split: 14,246 train / 750 held-out records. Defaults: student GPU 0, teacher GPU 1, BF16 model weights, FP32 CPU optimizer/master state, one prompt per microbatch, accumulation 32, one epoch, at most 1,024 prompt + 1,024 response tokens. Training performs no validation by default. The separate `evaluate_all.sh` command evaluates all held-out responses with ROUGE-L. No full training job is launched merely by changing these configs or running the environment check.

Models/checkpoints are written under:

```text
/nas/Users/wyx/Baseline/runs/llama3_8b_llama32_1b/dolly/kd/seed_42/
```

The other methods have their corresponding method subdirectories. An existing nonempty run requires explicit resume or a new output root; the trainer does not overwrite previous experiments.

For a separate two-step pilot before the full run:

```bash
python scripts/train.py --config configs/experiments/llama3_8b_llama32_1b_kd.json \
  --output-root /nas/Users/wyx/Baseline/pilots_llama3 \
  --max-steps 2 --gradient-accumulation-steps 1 \
  --max-prompt-tokens 128 --max-new-tokens 32
```

The pilot writes to a separate root, so the ordinary full-run command remains available afterward.

## Start ABKD or SKD

```bash
python scripts/train.py --config configs/experiments/llama3_8b_llama32_1b_abkd.json
python scripts/train.py --config configs/experiments/llama3_8b_llama32_1b_skd.json
```

Run those commands sequentially with this two-GPU placement. To execute all three sequentially in one command:

```bash
python scripts/run_matrix.py --pair llama3_8b_llama32_1b --methods kd abkd skd --execute
```

## Prepare and start DistiLLM-2

DistiLLM-2 requires teacher/student response pairs produced by the exact selected models and vocabulary policy:

```bash
python scripts/generate_pairs.py \
  --config configs/experiments/llama3_8b_llama32_1b_distillm2.json \
  --input /nas/Users/wyx/Baseline/data/dolly/train.jsonl \
  --output /nas/Users/wyx/Baseline/data/dolly/llama3_8b_llama32_1b/pairs.train.jsonl
python scripts/train.py --config configs/experiments/llama3_8b_llama32_1b_distillm2.json
```

Generate the paired corpus once before launching its training. Pair generation is an offline preprocessing step and can take substantial time. After pairs exist, `python scripts/run_matrix.py --pair llama3_8b_llama32_1b --execute` runs all four methods sequentially. Avoid rerunning that matrix over already completed output directories.

## Cross-version tokenizer handling

The two checkpoints have identical ordinary BPE vocabulary/merges but 249 special-token slots have different meanings/names. Merely changing the teacher path would fail the old exact-equality check. Each new Llama config therefore explicitly sets `vocabulary_policy: llama3_shared`.

The adapter validates and retains all **128,000 ordinary tokens plus five common active chat tokens**. The other 251 reserved/control slots are excluded from both models' normalized training distributions, validation, native generation and SKD proposal/acceptance/correction. The original model heads are preserved. This is conditional distillation on the shared vocabulary; it is a declared protocol change from using all output slots. Qwen retains full-vocabulary distillation.

The student chat template renders both roles. Its dynamic date is fixed to the template's own fallback, `26 Jul 2024`, so saved prompt IDs remain identical across days. Pair provenance and exported model/tokenizer configuration preserve the shared policy and suppression settings. Previously generated pairs under another teacher/policy must not be reused.

## Verification scope

The actual Llama teacher and student config/tokenizer/weight-header checks and BF16 CUDA preflight pass: [preflight report](environment-llama3.json), [tokenizer audit](evidence/llama3-tokenizer-audit.json). The 115 existing tests plus 36 new vocabulary/integration tests pass (151 total). The new native tiny-Llama tests cover the shared support, four losses/sampling paths, generation/export/resume and pair provenance. These checks do not claim that a full-size Llama training experiment has already completed. The large Qwen smoke evidence in the earlier validation record applies to Qwen only.

All 14,246 prepared training records and 750 validation records were additionally rendered/tokenized with the new policy; every encoded token is supported. Evidence: [Dolly readiness](evidence/llama3-dolly-readiness.json). No training job has been launched for this updated pair.
