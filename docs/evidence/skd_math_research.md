# SKD math experiment source audit

Checked 2026-09-16. Sources: Xu et al., *Speculative Knowledge Distillation*, arXiv 2410.11325v2 (ICLR 2025); original v1 checked for dataset agreement; released Google Research source checkout at `speculative_kd`, commit `8025cf1351da4d3cb8c289afe2d11247623fbdfb`. This audit does not change training code or running jobs.

## What was actually trained

The paper has **two distinct math protocols**. Its MATH results are not obtained by fine-tuning on the MATH training split. `Math_CoT` is the released code's filename for the math-instruction task; the paper identifies the source corpus as **UltraInteract**, not MetaMathQA. The exact released row selection and transformations cannot be reconstructed merely from that filename.

| Paper protocol | Teacher SFT | Student distillation prompts | Validation | Evaluation |
|---|---|---|---|---|
| Task-specific GSM8K | 7,000 GSM8K training examples | Random 1,000-example subset of those 7,000; 100-example low-data ablation | Remaining 473 GSM8K training examples | GSM8K official test: 1,319 examples |
| Math instruction following | 10,000 UltraInteract examples | 1,000 prompts, plus a 10,000-prompt experiment | 1,000 UltraInteract examples, held out from the 11,000 sampled in total | Held-out MATH, **GSM-plus**, SVAMP, ASDiv |

The held-out GSM benchmark in the second row is **GSM-plus, not GSM8K**. SKD §4.1 does not give exact MATH/GSM-plus/SVAMP/ASDiv evaluation row counts, so the standard benchmark sizes must not be presented as independently verified SKD run counts. §5.2 describes the instruction mixture as including MathQA, math reasoning and tabular processing.

Evidence: [paper §4.1, printed pp. 5–6](https://arxiv.org/pdf/2410.11325v2#page=5), [§5.2, printed p. 7](https://arxiv.org/pdf/2410.11325v2#page=7), [HTML §4.1](https://arxiv.org/html/2410.11325v2#S4.S1). Original [v1 §4.1](https://arxiv.org/html/2410.11325v1#S4.S1) states the same datasets and split sizes.

## Models and training stages

- The paper's families are Gemma **1** and Qwen **2**: task-SFT Gemma-7B-it → Gemma-2B-it, and task-SFT Qwen2-7B-Instruct → Qwen2-0.5B-Instruct. These are not Qwen3/Llama3 experiments.
- Teachers are task-SFT first and then frozen. SFT runs last three epochs; the checkpoint with lowest validation loss is selected.
- Student task-SFT is an **experimental initialization option**, not an unconditional prerequisite. Table 1's GSM8K distillation and Table 7's math instruction results start from the instruction-tuned student without task-SFT. §5.3 additionally compares task-SFT student initialization, using the student training subset for SFT.
- SFT and supervised KD receive prompt/reference pairs. SKD receives prompts and builds fresh trajectories using student proposals with teacher token replacement. It then minimizes token-level forward KL from teacher to student on those trajectories. This is not distillation on the static reference solution or simply on unchanged student rollouts.

Evidence: [§§3.1–4, printed pp. 3–5](https://arxiv.org/pdf/2410.11325v2#page=3), [Table 1, p. 6](https://arxiv.org/pdf/2410.11325v2#page=6), [Table 7, p. 21](https://arxiv.org/pdf/2410.11325v2#page=21), [Appendix A, p. 19](https://arxiv.org/pdf/2410.11325v2#page=19).

## Paper hyperparameters

| Setting | GSM8K | Math instruction → held-out MATH etc. |
|---|---|---|
| Learning rate | 1e-5 | 1e-5 |
| Distillation batch size / accumulation | 8 / 1 | 8 / 1 |
| Prompt / generated-response token caps | 256 / 512 | 1,024 / 1,024 |
| Student distillation budget, 1k prompts | 375 optimizer steps | 375 optimizer steps |
| Larger/low-data budget | 125 steps for 100 prompts | **7,500** steps for 10k prompts, as printed |
| Checkpoint selection | Lowest validation loss | Lowest validation loss |
| SKD token acceptance | Teacher top K=25 | Teacher top K=25 |
| Student proposal block | 5 tokens | 5 tokens |
| Student trajectory sampling | temperature 0.5; top-p 0.5 | temperature 0.5; top-p 0.5 |
| Replacement teacher sampling | temperature 0.2; top-p 1 | temperature 0.2; top-p 1 |
| Evaluation | Greedy decoding; final-answer accuracy | Greedy decoding; final-answer accuracy |

The main paper states warmup ratio 0.1 and dropout 0.1 for fine-tuning, with dropout disabled while sampling. The main K=25 setting was fixed without task-specific K tuning; Appendix B subsequently examines other K values. Teacher sampling parameters were searched on validation sets over temperature {0.2, 0.3, 0.4} and top-p {0.5, 1}.

Evidence: [Appendix D, printed pp. 20–21](https://arxiv.org/pdf/2410.11325v2#page=20), [§3.2, printed p. 4](https://arxiv.org/pdf/2410.11325v2#page=4), [§4, printed p. 5](https://arxiv.org/pdf/2410.11325v2#page=5). Table values are extracted factual settings, not a verbatim reproduction of paper prose.

## Released code defaults and reproducibility limits

1. **Full-data duration mismatch.** Appendix D says 7,500 steps for 10k math prompts with batch 8. Arithmetic gives six passes, whereas the release's `experimental_setup.md` says math three epochs. Three passes would give 3,750 updates. The three-epoch statement is consistent with 375 updates for the 1k experiment. Do not silently change the printed 7,500 to 3,750 or advertise a single unambiguous full-data duration. [Released setup](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/experimental_setup.md).

2. **Generic YAML is a summarization example.** `config/kd_train.yaml` specifies `summ_1k`, 1,024/128 token caps, teacher top-p 0.5, seed 20, three epochs, bf16, 8 processes and accumulation 1. Those sampling/token defaults must be overridden for math/GSM. In particular the paper's teacher top-p is 1 for both math tasks. [KD YAML](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/config/kd_train.yaml).

3. **SFT configs are particular examples.** Released math SFT selects `google/gemma-7b-it`, `Math_CoT`, length 2,048, three epochs, LR 1e-5, warmup 0.1, cosine scheduler, per-device batch 4, seed 42, checkpoint/eval every 32 steps. Released GSM SFT selects `google/gemma-2b-it`, `gsm_1k`, length 384, three epochs, per-device batch 16, checkpoint/eval every 16 steps. They represent different SFT roles and do not establish identical SFT batch/length for every model family. [Math SFT YAML](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/config/sft/sft_config_math.yaml), [GSM SFT YAML](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/config/sft/sft_config_gsm.yaml).

4. **KD implementation details.** Release uses AdamW over all student parameters, frozen teacher, BF16, FlashAttention 2, gradient clipping norm 1, linear scheduler with **zero** warmup steps. The latter differs from the paper's broad 0.1 warmup statement. It sets five assistant tokens with constant schedule, splits prompts across processes, and generates/updates per prompt with the configurable accumulation count. Math file branches are `data/Math_CoT_train.json` / `Math_CoT_vali.json` and `data/math_train_1k.json` / `math_vali_1k.json`; no data preparation logic establishes exact UltraInteract record identities in this checkout. [Training code](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/train/ddp_skd.py#L330).

5. **Math test entry point is incomplete in the release.** `eval/eval_gsm.py` genuinely loads official `openai/gsm8k` test, greedily generates, parses text after `####`, and calls `math_equal`. `eval/eval_math.py` instead loads **`Math_CoT_vali.json`**, not MATH test; defaults to 1,024 new tokens, uses boxed-answer parsing and `math_equal`. It cannot be treated as a verified full MATH benchmark command. Both scripts hardcode Gemma's tokenizer. The README's math-evaluation link points to EvalPlus, while the paper identifies Yuan et al.'s mathematical-equivalence code. [GSM evaluation](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/eval/eval_gsm.py), [Math validation evaluator](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/eval/eval_math.py), [Appendix G, printed p. 24](https://arxiv.org/pdf/2410.11325v2#page=24).

6. **Compute and random seeds.** Released setup requests 8×A100 80GB for 7B teacher / 2B student. This is the published release configuration, not a measured requirement for the local Qwen3/Llama3 implementation. KD YAML's seed 20 and SFT YAML's seed 42 are code defaults. The paper does not identify a multi-seed average or a seed list; do not present those defaults as a verified paper seed protocol. [Released README](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/README.md).

7. **Data availability.** The release README links a Google Drive folder for original prepared data. The checkout does not contain its `Math_CoT` data. The paper proves the intended source is UltraInteract; it does not prove a newly sampled UltraInteract/MetaMath corpus matches the original artifacts. Exact split manifests, seeds and preprocessing remain unverified until those artifacts are inspected. [Author repository redirect to Google Research](https://github.com/xu1998hz/skd), [data link in release README](https://github.com/google-research/google-research/blob/8025cf1351da4d3cb8c289afe2d11247623fbdfb/speculative_kd/README.md#data).

## Consequence for the local experiment design

To reproduce SKD's reported GSM8K and MATH protocols, prepare separate training tracks: GSM8K-task distillation and UltraInteract math-instruction distillation. This is separate training **by original protocol**, not an architectural requirement to train one model per evaluation benchmark. A single shared math-distilled checkpoint evaluated on MATH and GSM8K is a valid new comparison design, but it does not reproduce both SKD protocols. The local MetaMathQA proposal would also be a new common protocol, not the SKD paper's original data setup.
