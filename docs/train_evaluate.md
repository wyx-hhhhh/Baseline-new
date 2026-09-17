# Train, then evaluate with ROUGE-L

Train the four baselines, then evaluate the saved students:

```bash
bash /home/wyx/Baseline/scripts/run_all.sh
bash /home/wyx/Baseline/scripts/evaluate_all.sh
```

Training retains Qwen student/teacher on GPUs 0/1 and Llama student/teacher on
GPUs 2/3. Each group runs KD → ABKD → SKD → DistiLLM-2, with automatic pair
generation before DistiLLM-2. Default training saves models and checkpoints;
in-training evaluation remains disabled. Evaluation uses the final students.
Both commands resume matching interrupted work.

The evaluation command now uses the five downloaded natural-language datasets
under `/nas/Datasets`: **DollyEval, SelfInst, Super-Natural, Unnatural and
VicunaEval**. It automatically prepares the inputs, preserves multiple reference
answers, removes exact duplicate examples and excludes training-prompt overlap.
The [natural-language evaluation guide](natural_language_evaluation.md) records
exact counts, tokenization, commands, output locations and recovery behavior.
The full default queue has forty evaluations with separate ROUGE-L scores.

DollyEval contains 425 records that overlap the current Dolly training pool;
its reported held-out result therefore uses **75 records**. Super-Natural's
731 repeated rows are merged, leaving 7,623. SelfInst has 242, Unnatural 64,809,
and VicunaEval 80 examples. All five use Unicode ROUGE-L F1 by default, because
Unnatural contains non-English references. For multiple references, each
example receives its best reference F1; scores are averaged over examples and
scaled to 0–100. No prompt or reference is truncated.

Run only one completed model family, or select completed methods:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh --group qwen --qwen-gpus 0,1
bash /home/wyx/Baseline/scripts/evaluate_all.sh --group llama --llama-gpus 2,3
```

`--methods kd abkd` selects those methods. `--benchmarks` selects datasets;
`--limit` is an explicit per-model/per-benchmark pilot. `--dry-run` prepares or
verifies the CPU data cache and prints the queue without loading models.
All selected training runs must be complete before normal evaluation starts.

The prior internal Dolly validation workflow is still available:

```bash
bash /home/wyx/Baseline/scripts/evaluate_all.sh --validation-only
```

Put `--validation-only` first. This dispatches to the original
`scripts/run_evaluation.py`, scores the existing 750-row validation split with
its original English/Porter ROUGE-L defaults, and keeps its original output
paths. Its `--checkpoint best` option applies only to runs that explicitly
selected a committed checkpoint using in-training ROUGE-L. The default
training-only workflow uses final exports.

The math workflow uses separate [MetaMathQA campaign commands](math_campaign.md)
and final-answer accuracy. This evaluation update leaves training and math
implementations unchanged and launches no full GPU evaluation.
