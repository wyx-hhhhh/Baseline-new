# DistiLLM-2 math experiment evidence

Checked 2026-09-16. Research only; no training processes, GPU allocations, or executable training code changed.

## Published setup

The ICML proceedings and arXiv v2 both report a dedicated mathematical-reasoning experiment. They do **not** merely evaluate UltraChat-trained students out of domain. Their design is one shared math training pipeline, evaluated on both GSM8K and MATH.

| Item | Evidence in paper |
|---|---|
| Student initialization | Qwen2-Math-1.5B or Qwen2.5-Math-1.5B base student, then SFT on the entire MetaMathQA corpus for one epoch (§4.2, p. 7; §C.2, p. 18). |
| Teacher | Corresponding Qwen2-Math-7B-Instruct or Qwen2.5-Math-7B-Instruct (§4.2, p. 7). |
| Distillation data | 50,000 randomly selected MetaMathQA samples (§4.2, p. 7). The precise selected IDs and sampling seed are not provided in the inspected math release. |
| Pair refresh | Algorithm 1 (p. 3) generates teacher and current-student responses before each epoch, then trains on that round’s pairs. This is epoch-level batched generation, not fresh generation every optimizer step (§2.2, p. 2). |
| Math KD budget | Two epochs; LoRA rank 16; all attention and MLP linear layers; learning rate 5e-5; effective batch 128; four A100 80GB GPUs (§C.2, Table 11, p. 18). |
| Loss controls | Initial skew coefficient 0.1; no adaptive skew update during first epoch; beta floor 0.5 (Table 11). No extra language-model loss over pretraining text (§C.2). |
| Math evaluation | GSM8K and MATH pass@1, greedy decoding with maximum length 1,024; one A100 80GB GPU (§C.3, p. 18; Table 3, p. 7). |
| Math comparators | Teacher, SFT student, GKD, DistiLLM, DistiLLM-2. Table 3 does not contain ordinary KD, ABKD, or Speculative KD. Ordinary KD and Speculative KD appear in the instruction-following comparison, Table 2. |

Published Table 3, percentages (visually checked in proceedings PDF p. 7):

| Method | Qwen2 Math GSM8K | Qwen2 Math MATH | Qwen2.5 Math GSM8K | Qwen2.5 Math MATH |
|---|---:|---:|---:|---:|
| Teacher | 83.93 | 41.28 | 89.31 | 44.82 |
| SFT student | 74.53 | 25.56 | 77.33 | 27.14 |
| GKD | 75.44 | 34.16 | 80.21 | 40.54 |
| DistiLLM | 75.59 | 34.54 | 81.05 | 41.14 |
| DistiLLM-2 | 76.27 | 35.58 | 81.20 | 42.94 |

The paper does not give a reproducible math answer-extraction/grading implementation, exact test subset IDs/counts, a math-specific prompt template, or math training-generation temperature/top-p settings in the inspected setup sections. The greedy/max-length statement is for **evaluation**. The generic released generator’s sampling defaults must not be presented as proven math experimental settings.

## Official release audit

Local checkout: `distillm-2`, commit `fe4cf9bfbb4f83219dd1d69219800164b59685fa` (2025-06-27). Remote default branch is `master`, not `main`. The current official copies of the three files below were downloaded and compared byte-for-byte with the local checkout; all match.

| File | Actual behavior and consequence |
|---|---|
| `generate/generate_vllm.py`, lines 16–30, 63–67, 83–85 | Default `data_dir=ultrachat` actually gets prompts from the `generated` conversation field of `UCLA-AGI/SPIN_iter1`, train split; defaults temperature 0.8, top-p 0.95, 1,024 generated tokens, seed 42. It generates one completion per prompt; no explicit top-k setting. This is an instruction-data example, not the paper’s MetaMathQA pipeline. |
| `training_configs/qwen2.5-1.5b-distillm2.yaml` | Generic example points to an SFT student and Qwen2.5-7B-Instruct teacher; rank 16, alpha 128, dropout 0.05; LR5e-5; batch per device 1 × accumulation 8 × four processes = effective batch32; one epoch; total max_length1024, max_prompt_length512; cosine schedule, warmup0.1, AdamW, BF16. Thus the literal example differs from paper Table11 (batch128 and math two epochs). No released math configuration. |
| `src/alignment/model_utils.py`, lines 101–118 | `LoraConfig` sets `use_dora=True`. Paper calls its tuning LoRA; literal released adapter implementation uses DoRA. |
| `training_configs/qwen2.5-1.5b-sft.yaml` | Despite filename, model is `Qwen/Qwen2-1.5B`; data is UltraChat200k train_sft/test_sft, one epoch, max sequence2048. This example cannot establish the paper’s math SFT settings beyond what the paper states. |
| `generate/reformat.py`, lines 11–44 | Aligns teacher and student generations by prompt, stores teacher as chosen/student as rejected. Its “test” split selects the first500 rows from the training file; it is not a held-out GSM8K or MATH test benchmark. |
| `eval/`, `generate/`, `training_configs/` | Released eval scripts implement generic LLM-judge comparisons; no dedicated GSM8K/MATH grading pipeline or ready math config exists in this checkout. A generic `--data_dir` path alone does not supply required math fields/templates/grading. |

Official SPIN metadata reports **49,792 train + 500 test** examples, saved at `math_sources/distillm2_spin_iter1_metadata.json`; this explains the approximately50k default prompt set in the instruction-generation example. It is not the math subset.

The README’s evaluation section records a MATH availability warning and names `hendrycks/competition_math`. That is a release/access limitation, not evidence that the paper lacked math experiments. Appendix C.1 p.17 contains an apparent dataset-description/link error: its MATH entry cites Hendrycks but links to `deepmind/math_dataset` and describes that unrelated benchmark. The README makes competition-MATH the stronger identification, but exact released evaluation details are still absent. No claim about present universal dataset availability follows from this historical README warning.

## Implications for this Baseline workspace

Reproducing the published math setting requires a common math SFT starting checkpoint, selected math prompts, and epoch-level pair refresh for DistiLLM-2. Running its released UltraChat example or this workspace’s Dolly configuration would be a different experiment. A fair new comparison with Qwen3 and Llama3 can use the same math corpus and evaluate each distilled checkpoint on both benchmarks, but that model choice and any changed budget must be labelled an adaptation.

## Sources and saved artifacts

- [ICML proceedings landing page](https://proceedings.mlr.press/v267/ko25a.html), with [official PDF](https://raw.githubusercontent.com/mlresearch/v267/main/assets/ko25a/ko25a.pdf), pp. 2–3, 6–7, 17–19. Saved as `math_sources/distillm2_icml2025.pdf` and its extracted `.txt`.
- [arXiv v2](https://arxiv.org/html/2503.07067v2), [PDF](https://arxiv.org/pdf/2503.07067v2), §4.2/Table3 and AppendixC.2–C.3/Table11. Saved as `math_sources/distillm2_arxiv_v2.pdf` and `.txt`; relevant math setup agrees with proceedings.
- [Official README](https://github.com/jongwooko/distillm-2/blob/master/README.md).
- [Official generator](https://github.com/jongwooko/distillm-2/blob/master/generate/generate_vllm.py), saved as `math_sources/distillm2_generate_vllm.py`.
- [Official KD config](https://github.com/jongwooko/distillm-2/blob/master/training_configs/qwen2.5-1.5b-distillm2.yaml), saved as `math_sources/distillm2_qwen_config.yaml`.
- [Official model utilities](https://github.com/jongwooko/distillm-2/blob/master/src/alignment/model_utils.py), saved as `math_sources/distillm2_model_utils.py`.
- [Official SFT config](https://github.com/jongwooko/distillm-2/blob/master/training_configs/qwen2.5-1.5b-sft.yaml) and [pair formatter](https://github.com/jongwooko/distillm-2/blob/master/generate/reformat.py).
- [SPIN_iter1 dataset](https://huggingface.co/datasets/UCLA-AGI/SPIN_iter1), [metadata API](https://huggingface.co/api/datasets/UCLA-AGI/SPIN_iter1).

No fixed total optimizer-step count is asserted: 50k/batch128 implies approximately391 updates per full pass, but accumulation/last-batch behavior and published execution details are not available to prove the exact count.
