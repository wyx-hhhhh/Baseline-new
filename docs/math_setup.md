# Math experiment setup

The fixed MetaMathQA campaign is now prepared. Use the
[math campaign guide](math_campaign.md) for launch commands, recovery, settings
and output paths. Existing Dolly commands continue their Dolly queues.

All four methods (KD, ABKD, SKD, DistiLLM-2) share exactly 50,000 fixed MetaMathQA
training examples and a separate 5,000-example development split, seed 42.
Both benchmark test sets are excluded from the pool using normalized question
and linked-family checks. Every student trains once and is tested separately
on all 1,319 GSM8K and all 5,000 competition-MATH questions.

The original Qwen3-8B → Qwen3-1.7B and Llama-3-8B → Llama-3.2-1B Instruct
checkpoints initialize every method. Each baseline receives one epoch, LR
1e-5, microbatch 1 with accumulation 32, and 2,048-token prompt/response budgets.
No additional math SFT stage is included. This is the user's common comparison;
the [original baseline protocols](math_baseline_literature.md) differ.

Math final-answer accuracy and resumable evaluation are implemented in
`baseline_common/math_metrics.py`, `baseline_common/math_evaluation.py` and
`scripts/evaluate_math.py`. The [grading guide](math_grading.md) records the
pinned extraction/equivalence rules and sixteen explicit MATH gold-annotation
interpretations. Natural-language evaluation continues to use ROUGE-L.

The authoritative pool is
`/nas/Users/wyx/Baseline/data/metamathqa_50k_v1`; outputs use
`/nas/Users/wyx/Baseline/math`. Use the dedicated `run_math.sh` and
`evaluate_math_all.sh` launchers described in the guide. Data, code and CPU
recovery checks are complete; full math training is left for the user's launch.
