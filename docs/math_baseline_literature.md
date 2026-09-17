# Original baseline setups for GSM8K and MATH

Research checked 2026-09-16 against the papers, appendices, final conference papers and the four official source checkouts. This is a literature/source audit. It does not modify the current training jobs or convert the existing Dolly campaign into a math campaign.

**Finding:** there is no single shared original math recipe for SKD, ABKD, DistiLLM-2 and ordinary KD. DistiLLM-2 explicitly trains on MetaMathQA and evaluates the same math student on GSM8K and MATH. SKD instead reports both a GSM8K-specific task and a separate UltraInteract-based generalization task. ABKD's final paper reports math results, while several training details have to be recovered from nonmatching repository examples.

Implementation update: the [fixed MetaMathQA campaign](math_campaign.md) now implements the user's common Qwen3/Llama3 comparison: 50,000 shared training examples, 5,000 development examples, eight independent distillation runs and sixteen separate full-test evaluations. It starts from the original Instruct weights with no added SFT. The paper/source findings below remain the original literature audit, distinct from this common local protocol.

## Train/evaluate relationship

| Baseline/experiment | Original training setup | Evaluation | Evidence boundary |
|---|---|---|---|
| SKD: GSM8K task | Teacher task-SFT on 7,000 GSM8K examples; student KD on a sampled 1,000 (100-example ablation also exists); 473 dev examples | Official GSM8K test, 1,319 examples | A task-specific GSM8K experiment |
| SKD: math instruction | Sample 11,000 UltraInteract examples: 10,000 training and 1,000 dev; student KD with 1,000 or 10,000 prompts | Held-out MATH, **GSM-plus**, SVAMP, ASDiv | GSM-plus must not be relabelled GSM8K. Exact evaluation row membership/counts are not supplied |
| ABKD: math | Repository preparation samples 55,000 MetaMathQA rows → 50,000 train / 5,000 validation. Active ABKD training reads a separate `pseudo` dataset | Final paper Table 12 reports GSM8K, MATH and three additional math benchmarks with CoT prompting | The final paper does not specify the math training corpus/hyperparameters. The 50k/5k recipe writes `full`, so it does not prove the size/content of the `pseudo` dataset |
| DistiLLM-2: math | Student first SFT on the entire MetaMathQA for one epoch; KD on 50,000 randomly chosen MetaMathQA examples for two epochs; refresh response pairs per epoch | Same student evaluated on GSM8K and MATH, pass@1 | This directly supports “one math training track, two test benchmarks” |
| Ordinary KD | Dataset/initialization are supplied by the comparison protocol. SKD includes supervised KD on its task data; ABKD includes a KD row in its math comparison | Corresponding task test sets | KD is an objective, not a separate fixed math-data recipe. DistiLLM-2 Table 3 does not report an ordinary-KD math row |

Sources: [SKD §4.1 and §5.2](https://arxiv.org/html/2410.11325v2#S4.S1), [ABKD final Table 12, p.43](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25dz/wang25dz.pdf#page=43), [ABKD preprocessing script](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/tools/process_data_metamath.sh), [DistiLLM-2 §4.2 and Appendix C](https://arxiv.org/html/2503.07067v2#S4.SS2).

## Models, stages and optimization

| Item | SKD paper/release | ABKD math release example | DistiLLM-2 math paper |
|---|---|---|---|
| Teacher/student | Task-SFT Gemma-7B-it → Gemma-2B-it; task-SFT Qwen2-7B-Instruct → Qwen2-0.5B-Instruct | Qwen2.5-Math-7B-Instruct → **general** Qwen2.5-1.5B-Instruct | Qwen2-Math and Qwen2.5-Math 7B-Instruct → corresponding 1.5B **base** students |
| Initial student SFT | Optional comparison: instruction-tuned students also used directly | Active AB script directly selects the Instruct student; separate SFT scripts are not called by it | Required in main math experiments: entire MetaMathQA, one epoch |
| KD adaptation | All student parameters in release | Full student in active AB script; LoRA commented out | LoRA rank 16, attention and MLP linear layers |
| Learning rate | 1e-5 | 1e-5 | 5e-5 |
| KD batch | Paper lists batch 8, accumulation 1 | Microbatch 4 × accumulation 4 × GPU count; 256 at the script's default 16 ranks | Effective batch 128 |
| KD duration | 375 steps for 1k prompts; 7,500 printed for 10k math, with a discrepancy below | Four epochs in script; not established as final-paper math hyperparameter | Two epochs |
| Lengths | GSM: 256 prompt + 512 response; math: 1,024 + 1,024 | Max prompt 512, **total** sequence 2,048 | Evaluation max length 1,024; exact math training/rollout lengths not separately established by the paper |
| Decoding/metric | Greedy final-answer accuracy with mathematical-equivalence grading | Evaluation script: greedy, up to 2,048 new tokens, boxed-answer extraction/equivalence | Greedy, pass@1; exact extraction/subset details incompletely specified |
| Compute | Release describes 8×A100 80GB | Paper math hardware unspecified; script defaults are not hardware evidence | 4×A100 80GB training; single A100 evaluation |

SKD uses acceptance K=25 and a five-token proposal block. The paper's math/GSM sampler uses student temperature/top-p .5/.5 and teacher .2/1. ABKD's active script fixes alpha=.1, beta=.8; its outer alpha/beta loops are a sweep, not a schedule within training. DistiLLM-2's paper includes gradual reverse-loss weighting and adaptive skew coefficients; the latter remain fixed in the first epoch.

Evidence: [SKD Appendix D, pp.20–21](https://arxiv.org/pdf/2410.11325v2#page=20), [SKD released training/setup](https://github.com/google-research/google-research/tree/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd), [ABKD math launcher](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/ab/train_3B_7B_teacher.sh), [DistiLLM-2 final paper Appendix C, p.18](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ko25a/ko25a.pdf#page=18).

## Important paper/code discrepancies

**SKD.** The paper identifies UltraInteract as the math source, resolving the intended corpus behind the code's `Math_CoT` task. Exact custom-file extraction and selected row IDs remain unavailable. The paper's 7,500 updates on 10k examples at batch 8 imply six passes; the release says three epochs. The release's math evaluator reads `Math_CoT_vali.json`, not the MATH test set, and uses a hardcoded Gemma tokenizer. Generic KD YAML is a summarization example and must not be applied unchanged to math. See [detailed SKD audit](evidence/skd_math_research.md).

**ABKD.** The final ICML PDF contains Table 12, which is absent from the inspected arXiv version; reading only arXiv misses its math evidence. Table 12 identifies models and CoT evaluation, but not a full math recipe. Do not reuse Appendix I.1.3's Dolly hyperparameters as math settings. The checked-in KD script selects an SFT 0.5B student/SFT 7B teacher, `full` MetaMath, LR5e-6, two epochs and total length512; the AB script selects a different student, `pseudo` inputs, LR1e-5, four epochs and total length2,048. Those scripts cannot establish a controlled paper-matched KD-versus-ABKD comparison without reconciliation. The pseudo-data generation/membership manifest is missing. See [detailed ABKD audit](evidence/abkd_math_research.md).

**DistiLLM-2.** Its math paper really uses MetaMathQA; the README's UltraChat description concerns instruction-following. The released generator's default `ultrachat` branch loads `UCLA-AGI/SPIN_iter1`, and there is no dedicated released math configuration/grader. The generic Qwen example uses one epoch, effective batch32 and adapter code enabling DoRA, versus the paper's math settings above. `update_alpha=False` and `gradual_beta=False` in the release disable paper components. The paper's Appendix C.1 also links the wrong DeepMind math dataset while citing MATH; the README points to `hendrycks/competition_math`. Exact benchmark subsets/grading should therefore be frozen explicitly. See [detailed DistiLLM-2 audit](evidence/distillm2_math_research.md).

**Ordinary KD.** For a controlled local comparison, KD should share the data, teacher, initial student, optimizer budget and decoding/evaluation protocol with the other methods. Its reference trajectory policy must be stated. The sample KD filename `kd_3B_7B_teacher.sh` is not proof of its active model: the current script selects a 0.5B checkpoint. The DistiLLM-2 math table compares GKD, DistiLLM and DistiLLM-2, not all four methods requested here.

## Consequences for the user's experiment

The clean common comparison is a deliberately standardized math track: use the same fixed MetaMathQA prompt pool for all four methods and evaluate each trained student on both GSM8K and MATH. A 50,000-example KD pool is supported by DistiLLM-2's paper and ABKD's available preprocessing example. MetaMathQA's authors state that its examples derive from **training** questions in GSM8K and MATH, with `original_question` available for checking provenance: [dataset card](https://huggingface.co/datasets/meta-math/MetaMathQA/blob/main/README.md).

This recommendation is an adaptation for Qwen3-8B→1.7B and Llama-3-8B→Llama-3.2-1B, not an exact replication of those older model families. Use identical initial student/teacher checkpoints across methods; if a math-SFT initialization is added, create it once per model pair and reuse it. Do not initialize each method from whichever Dolly-distilled checkpoint happens to finish before it. Decide consistently whether to keep the current full-finetune memory profile or use an all-method LoRA track, and hold the prompt/update budget constant rather than copying incompatible per-paper defaults.

This common design has **eight distillation runs and sixteen benchmark evaluations per seed**, plus any shared SFT preparation. If exact SKD reproduction is also required, add its separate GSM8K-task and UltraInteract-math tracks. Training separately on the two benchmark training splits is a research-design option, not an inherent requirement imposed by evaluating two test sets.

For the selected evaluation protocol, report final-answer accuracy/pass@1 with the same prompt, answer extractor, generation budget and benchmark split across all methods. ROUGE-L measures textual overlap and remains the metric for the natural-language track, not math correctness. Keep official test questions out of training and keep augmentation families together when making development splits. The exact final MATH test subset must be named: the reviewed papers label MATH but do not uniformly establish MATH-500 versus full-test row membership.

On this server, use the verified `hendrycks_competition_math` files for the MATH benchmark; the separately named local `MATH` directory was inspected as coding-task data. At the time of this literature audit, the shared runner had no math answer-accuracy evaluator. The subsequent [math campaign implementation](math_campaign.md) adds the frozen pool and resumable evaluator with a [documented grading protocol](math_grading.md). It retains one-time DistiLLM-2 pairs and fixed skew coefficients, which differ from the paper's refreshed/adaptive setup.

## Reproducibility sources

Local original source commits inspected:

| Checkout | Commit |
|---|---|
| SKD / Google Research | `8025cf1351da4d3cb8c289afe2d11247623fbdfb` |
| ABKD | `027ec2621aad678b94587bb36cab9a16b8a05fed` |
| DistiLLM-2 | `fe4cf9bfbb4f83219dd1d69219800164b59685fa` |
| DistiLLM / ordinary KD examples | `d47e77ff9d27721783b32213be38c1204230cc0a` |

Primary PDFs/text snapshots and supporting artifacts are under `docs/evidence/math_sources/`. The three method-specific audit notes link exact sections/pages and flag unverified values. No training implementation, model checkpoint, optimizer state or running process was changed by this research.
