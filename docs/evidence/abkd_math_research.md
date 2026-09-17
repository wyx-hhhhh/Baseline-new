# ABKD math experiments: primary-source audit

Inspected 2026-09-16. This is research only; no training process, configuration, or model was changed.

## Published result versus released implementation

The [ICML proceedings paper](https://proceedings.mlr.press/v267/wang25dz.html), [PDF page 43, Table 12](https://raw.githubusercontent.com/mlresearch/v267/main/assets/wang25dz/wang25dz.pdf#page=43), evaluates **Qwen2.5-Math-7B-Instruct → Qwen2.5-1.5B-Instruct** with chain-of-thought prompting. The student is the general Instruct model, not Qwen2.5-Math-1.5B. Its five benchmarks are GSM8K, MATH, GaoKao 2023 English, OlympiadBench, and College Math. The [official conference slides, page 22](https://icml.cc/media/icml-2025/Slides/43650.pdf#page=22), describe the metric as pass@1.

| Published model/method | GSM8K | MATH | Five-benchmark average |
|---|---:|---:|---:|
| Teacher | 95.5 | 82.8 | 64.3 |
| Original student | 73.3 | 54.9 | 44.7 |
| SeqKD | 75.8 | 57.3 | 45.9 |
| KD | 75.9 | 58.1 | 46.3 |
| ABKD | 77.4 | 58.6 | 47.4 |

**The published math table does not specify the training corpus, split, epoch count, optimizer, seeds, GPU count, or MATH subset size.** Appendix I.1.3, page 37, explicitly describes Dolly experiments; its 20 epochs, five evaluation seeds, and alpha=.2/beta=.7 must not be presented as math settings. The arXiv v2/v3 HTML omits the final proceedings' math table.

## What the official repository actually establishes

The clean local checkout `abkd/` is official repository [ghwang-s/abkd](https://github.com/ghwang-s/abkd), commit `027ec2621aad678b94587bb36cab9a16b8a05fed` (2025-08-09). The following are **checked-in script defaults**, not proof of the complete configuration that produced Table 12.

| Item | Verified behavior | Authoritative repository source |
|---|---|---|
| Math corpus preparation | Shuffle `MetaMathQA-395K.json` using seed 42; keep 55,000 rows; first 5,000 validation, remaining 50,000 training | [preparation launcher](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/tools/process_data_metamath.sh), [processor lines 88–101](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/tools/process_data_metamath.py#L88) |
| Filtering | No source-type filtering, deduplication, grouped-question splitting, or test-overlap exclusions in this processor. Prompts longer than 256 tokens are truncated in the published launcher | Same processor, lines 26–76 and 88–101 |
| ABKD data | `processed_data/metamath/pseudo/qwen/`, whereas the preparation launcher writes `metamath/full/qwen/`. A complete Qwen pseudo-response generation/preparation recipe is not present | [ABKD launcher](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/ab/train_3B_7B_teacher.sh#L22) |
| ABKD models | Original Qwen2.5-1.5B-Instruct student; Qwen2.5-Math-7B-Instruct teacher. No preliminary local SFT checkpoint in the active ABKD model paths | ABKD launcher, lines 22–26 |
| ABKD optimizer/budget | AdamW; LR 1e-5; 4 epochs; cosine; zero warmup; weight decay .01; gradient clip 1.0 | ABKD launcher lines 30–69; [trainer lines 80–119](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/finetune.py#L80) |
| ABKD batch | Per-rank batch 4 × accumulation 4 × GPU process count. Default launcher count 16 gives effective batch 256; actual published math run GPU count is unspecified | ABKD launcher lines 7, 30–33, 58–60 |
| ABKD tokens | Total training sequence cap 2,048; prompt cap 512 | ABKD launcher lines 35, 68–69 |
| ABKD objective | Pure response-token alpha-beta distillation, alpha=.1, beta=.8, kd-ratio=1.0; cross-entropy coefficient becomes zero | ABKD launcher lines 9–12, 66, 132–144; trainer lines 175–207 and 355–358 |
| Alpha/beta behavior | Shell start/end variables describe a sweep over separate runs. They are not an annealing schedule within a training run. The defaults select one fixed pair | ABKD launcher lines 132–151 |
| Student-generated outputs | Although `--student-gen` is passed, actual generation branches require `mixed` or `adaptive` in method type. Plain `ab` and `fkl` do not take these branches | Trainer lines 320–343 |
| Full versus LoRA | ABKD and KD launchers have `--peft lora` commented out: full student optimization, fixed teacher under no-grad | ABKD launcher line 80; trainer lines 175–179; `utils.py` lines 125–175 |
| Seed/validation | ABKD launcher seed 10; validation capped at 1,000 despite the separate full-data preprocessing allocating 5,000; save/evaluate each epoch | ABKD launcher lines 37, 55, 74–75; trainer lines 610–614 |
| Precision caveat | The ABKD script references `ds_config_bf16.json`, absent in this commit. Model-loading functions initially request FP16. Therefore a working BF16 math configuration cannot be established from this launcher alone | ABKD launcher line 90; trainer lines 50–61; `utils.py` lines 125–138 |

The **50K/5K counts belong to the full MetaMath preparation script**. They do not establish the size, origin, correctness filtering, or generation settings of ABKD's `pseudo` data. Calling the ABKD math experiment a verified 50K-teacher-response experiment would overstate the evidence.

## Ordinary KD and SFT scripts are not a matching comparison recipe

The [Qwen KD launcher](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/kd/kd_3B_7B_teacher.sh) has a misleading 3B filename: its active student path is an SFT **Qwen2.5-0.5B** result, and its teacher an SFT 7B result. It selects `metamath/full/qwen`, LR 5e-6, 2 epochs, batch 8 × accumulation 2 × processes, total length 512/prompt 256, seed 10, and forward KL only. Hence this script cannot independently reproduce the Table 12 KD row or constitute a controlled comparison with the ABKD launcher.

The separate Qwen [teacher SFT](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/sft/sft_7B_lora.sh) and [student SFT](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/sft/sft_1.5B_lora.sh) scripts select locally named `qwen2.5-math-7b` / `qwen2.5-math-1.5b`, enable LoRA (argument defaults r=16, alpha=64, dropout=.1), and use LR 5e-5, 2 epochs, total length 512, prompt 256, seed 10, and default effective batch 128 on eight processes. Their model names alone do not establish base versus Instruct provenance. They are not prerequisites encoded by the active ABKD launcher; do not combine these scripts into a claimed paper training sequence.

## Math evaluation and shared-training inference

The [released math evaluator](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/eval/eval_math.py) uses a system instruction for step-by-step reasoning and a boxed final answer, the model chat template, greedy decoding (`temperature=0`, `top_p=1`), and **2,048 generated tokens**. It accepts `question`/`answer` or `problem`/`solution`; defaults to GSM8K's test JSONL. The [grading helper](https://github.com/ghwang-s/abkd/blob/027ec2621aad678b94587bb36cab9a16b8a05fed/distillation_llm/scripts/qwen/eval/util.py) extracts final answers and tests normalized/symbolic equivalence; invalid answers count as incorrect. This is answer accuracy, not ROUGE-L. The training-time evaluator in `finetune.py` still calls ROUGE; it should not be conflated with the standalone math result.

The table gives a multi-benchmark evaluation of each method, and the repository offers one MetaMath training track plus an evaluator taking arbitrary test files. This supports **one shared math-trained model evaluated on both GSM8K and MATH**, rather than requiring separately trained benchmark models. It is an inference from the released structure: no per-benchmark checkpoint manifest proves the exact Table 12 checkpoint identity. The publication labels the benchmark **MATH**, without a verified sample count or explicit MATH-500 selection; report that limitation rather than asserting either subset.

## Evidence files

`math_sources/abkd_icml2025.pdf` is the proceedings PDF, `abkd_icml2025.txt` its layout-preserving text extraction, and `abkd_icml2025_p43.png` a rendered/visually checked Table 12 page. `math_sources/abkd_provenance.json` records hashes and source locations. No model or dataset preparation was executed.
