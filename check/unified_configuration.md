Use **one training environment for KD, GKD, ABKD, SKD and DistiLLM-2** after implementing the compatibility repairs. An independent process/environment may generate DistiLLM-2 offline responses with vLLM, but it must use the same checkpoint revisions, tokenizer, prompt rendering and decoding specification as the training-side reference implementation.

**Version specification, proposed v1.** These pins deliberately anchor the migration around Qwen3 support. They are not the newest releases, a solved transitive lock, or a validated GPU environment.

| Component | Proposed version | Reason / condition |
|---|---|---|
| Python | **3.11.13** | Meets DistiLLM-2's `>=3.10.9`; supported by published metadata of the selected Python packages; use one interpreter across all methods |
| PyTorch | **2.6.0**, CUDA **12.4** wheel profile | Common GPU math/autograd runtime; compatible published wheel set; actual driver/GPU validation required |
| Transformers | **4.51.3** | Qwen3 support; fixed generation/Trainer API target |
| tokenizers | 0.21.1 | Within Transformers' `>=0.21,<0.22` requirement |
| Accelerate | 1.6.0 | Shared device/distributed layer; migrate old usage explicitly |
| PEFT | 0.15.2 | Optional adaptation support; core run is full fine-tuning |
| TRL | **0.9.6** | Intentionally retain DistiLLM-2's copied DPO helper/config dependency to reduce migration surface; target helper symbols were inspected. This does not make its old Trainer overrides compatible automatically. |
| DeepSpeed | 0.16.7 | Shared optional sharding backend; Linux build and multi-model behavior must be tested |
| datasets | 3.5.0 | Shared data serialization/loading; define file schema and avoid depending on arbitrary external dataset code |
| huggingface-hub | 0.30.2 | Satisfies selected Transformers/PEFT metadata bounds |
| NumPy | 1.26.4 | Satisfies TRL's `<2.0` bound; avoid changing integer-conversion behavior during this port |
| Attention backend | `sdpa` initially | Shared PyTorch path; use `eager` if required for reference correctness; FlashAttention is optional |
| Optional FlashAttention | 2.7.4.post1, build-specific | Only after numerical parity and target CUDA/PyTorch/compiler checks; all five methods use the same backend in reported runs |
| Optional vLLM generation | 0.8.5 in a separate environment | Its metadata pins torch 2.6.0 and requires Transformers >=4.51.1; includes additional packages such as torchvision/xformers that training does not need |

The release requirements were fetched from primary [PyPI metadata](evidence/public_metadata.json). Checks found no contradictions in 27 direct constraints among the selected core/optional packages, but 114 active requirement entries refer to packages outside this short list. They have not been transitively resolved or built. In particular, “no direct conflicts” is not “pip check passed.” Keep [constraints-training-proposed.txt](constraints-training-proposed.txt) as a constraints proposal until the later Linux installation produces a complete lock with hashes.

Retaining old TRL is a tactical compatibility decision, not a recommendation to install an old end-to-end GKD trainer. KD/GKD come from inspected loss/sampling logic, and DistiLLM-2's trainer subclasses Transformers `Trainer`. If even the adapted TRL integration fails the smoke tests, vendor the few required collator/config utilities with attribution or port to a newer TRL version in a separately reviewed change. Change the common manifest at the same time; do not silently give one baseline a different stack.

**Installation work for the later coding task.** Create a fresh Linux environment, pin Python, then install torch from the CUDA 12.4 index. An intended command is:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

This command is copied as an intended installation step, not executed by this investigation. GPU support must be checked first. The archive lists CUDA 11.8, 12.4 and 12.6 builds for torch 2.6.0. [Official PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/).

Update the repositories' dependency declarations before installing them: existing `distillm-2/setup.py` requests incompatible old pins and imports vLLM into the training installation. Remove vLLM/FlashAttention from mandatory training dependencies and offer them as optional extras. Replace ABKD/DistiLLM's unpinned shell installers and SKD's bare `alignment` dependency; `alignment` in this workflow refers to the inspected alignment-handbook-derived code, not an arbitrary similarly named PyPI package. Inventory actual imports and add required lightweight packages (e.g. YAML, Click, safetensors, sentencepiece, evaluation utilities) before generating the full lock. Do not run all existing installers into the same environment.

**Model specification.** Main proposal follows the literal teacher sizes and names; the unresolved Llama shorthand is annotated explicitly.

| Experiment family | Teacher | Student | Revision policy / stage |
|---|---|---|---|
| `qwen3_8b_1p7b` | `Qwen/Qwen3-8B` | `Qwen/Qwen3-1.7B` | Teacher `b968826d9c46dd6066d109eabc6255188de91218`; student `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; post-trained, non-thinking experiment |
| `llama3_8b_llama32_1b` | `meta-llama/Meta-Llama-3-8B` | `meta-llama/Llama-3.2-1B` | API revisions `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920` / `4e20de362430cd3b72f300e6b0f18e50e7166e08`; base models; exact artifacts require authorized access |
| Optional replacement family | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.2-1B` | Teacher API revision `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`; explicit alternate teacher, not an extra mandatory experiment |

The metadata revisions record what was observed; after SFT, every run must instead reference the exact shared SFT checkpoint and its manifest. Pin teacher and student revisions independently. Never pass one model's revision to the other. If selecting Instruct Llama variants, change both exact IDs and rerun tokenizer/stopping checks. Do not invent `Llama-3-1B` or append `-Instruct` to the specified Qwen3 IDs without choosing a real checkpoint.

For Qwen, keep the full 151,936-dimensional output distribution and retain the original tokenizer special tokens. The downloaded tokenizer has 151,669 entries: do not resize its head to this smaller value merely because the older utility does so. For Llama, validate the full mapping and active special-token semantics before choosing one shared tokenizer. Equal lengths or matching sample sentences alone are insufficient. A mismatch must trigger an explicit mapping/re-tokenization design review, not arbitrary head resizing. These are within-family distillations; no Qwen-to-Llama logit matching is proposed.

**Shared training and data contract.** The following values are concrete starting proposals for the later implementation, not tuned settings or defaults verified on the user's GPU/data.

| Setting | Proposed initial value / rule |
|---|---|
| Training precision | BF16 parameters/forward where supported, FP32 distribution/loss reductions; frozen teacher, no teacher gradients |
| Student adaptation | Full fine-tuning for all five; optional separate all-method LoRA track, DoRA off unless explicitly selected |
| Teacher precision | BF16, no 4/8-bit quantization in the primary comparison |
| Prompt/response budget | At most 1,024 prompt tokens and 1,024 response tokens, total 2,048; task-specific change must affect all five methods |
| Batch accounting | Target 32 unique prompts per optimizer step; `world_size × microbatch_prompts × accumulation = 32` |
| Resource example | 4 processes × 1 prompt × 8 accumulation; hardware-dependent, not a required or validated server configuration |
| Optimizer | `torch.optim.AdamW`, LR 1e-5, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01; gradient clipping 1.0 |
| Schedule | Cosine, warmup ratio 0.03; start with one pass over the fixed prompt-ID set, record actual steps |
| Repetition | Seeds 42, 43, 44; same initialization manifest and same split per seed |
| Loss | Response-only, causal next-token alignment, no pad/prompt loss; distillation temperature 1.0 initially |
| Temperature convention | Distillation temperature and generation temperatures are separate fields; at T!=1 document/implement T² scaling for KL and separately define scaling for AB divergence |
| Data | Stable `example_id`, prompt/messages, reference response, split, provenance; tokenize from raw data for each pair; store explicit boundaries |
| Validation | Disjoint prompt IDs and source-example groups, fixed held-out set; checkpoint selection uses one task metric shared across methods |
| Reporting | Task metric, seed mean/std, training time, online/offline generation time, peak VRAM, optimizer steps, unique prompts, response tokens, teacher forward/generation tokens |

A default task/dataset is deliberately not invented. Fill in dataset revision/hash, split membership, metric, validation interval and final update budget before launch. The experiment manifest keeps these unresolved fields as null and must reject execution until they are provided. This is a concrete configuration schema for future code, not a claim to have designed the user's unknown task.

Within the base Llama track, create one explicit task prompt format and optionally a common SFT initialization, then reuse it across all methods; base models may not provide a chat template. Within the Qwen track, call the official template with `enable_thinking=False` and persist its revision/hash. Serialize the entire conversation and store the exact generation prefix; do not independently concatenate arbitrary prompt-only and assistant-only template outputs. Test prompt/full-sequence prefix equality, boundary token merges, EOS, multilingual text, padding and truncation.

**Method-specific settings.** Unify the infrastructure and comparison controls while preserving the algorithms' different sequence sources.

| Method | Sequence source in proposed primary comparison | Loss / method settings |
|---|---|---|
| KD | Fixed reference responses | Forward KL teacher→student, temperature 1, KD weight 1, hard-label CE weight 0 |
| ABKD | Same fixed reference responses as KD | Alpha=0.1, beta=0.8 initially; validate/tune on held-out data with disclosed equal tuning budget. Record this as ABKD on reference trajectories; the provided Qwen script's pseudo-response setup is a separate reproduction configuration. |
| GKD | Per batch, probability 0.5 of fresh current-student responses; otherwise reference responses | Generalized JS with student weight 0.9, teacher weight 0.1; no replay. Formula: M=0.1P_teacher+0.9P_student; loss=0.1 KL(P_teacher‖M)+0.9 KL(P_student‖M). |
| SKD | Current-student proposals, teacher intervention | Forward KL; teacher acceptance top-k 25, proposal block length 5, constant block schedule, expected-sequence-length heuristic off; no replay |
| DistiLLM-2 | Offline teacher and initial-student responses for each training prompt | `distillm_v2`, skew alphas 0.1/0.1, adaptive-alpha off, gradual-beta off (effective weighting beta=1), reference-log-prob precomputation off; chosen=teacher, rejected=student |

GKD's chosen setting follows the existing mixed-JSD script's mixture/divergence values, with fresh sampling to honor the on-policy branch. It is one specified GKD variant, not the only one in the paper. Pure on-policy forward-KL can be an optional ablation. The JS student weight here is 0.9; DistiLLM-2's unrelated `gkd` branch uses the opposite mixture weighting and offline paired data, so it is not equivalent.

ABKD cannot be made on-policy just by passing the legacy `--student-gen` flag with `--type ab`; the current branch conditions prevent it. Select sequence source independently of the loss in the future runner. This also prevents accidentally giving ABKD a richer pseudo dataset while KD receives only human references.

The DistiLLM-2 release averages each response over valid tokens, sums chosen/rejected losses, then averages pairs. Preserve that primary reduction for fidelity, and document its two-response scale. KD/ABKD/GKD/SKD should use a token mean over valid response predictions, computed correctly across accumulation/ranks. Do not claim these different reductions are identical. A common-reduction ablation must be separately labeled and tuned fairly.

**Generation contract.** Initial harmonized Qwen non-thinking sampling uses temperature 0.7, top-p 0.8, top-k 20, and max_new_tokens 1,024 for ordinary teacher/student generation. Apply the same values to GKD and SKD proposal generation where those roles occur. For the Llama base task, use an explicitly recorded starting sampler (temperature 0.7, top-p 0.9, top-k 0); it is a proposal, not an official recommended setting. SKD's acceptance-k=25 remains a different field from generation top-k. Verify that SKD does not inherit teacher defaults into the assistant inadvertently. The official Qwen guidance distinguishes thinking/non-thinking settings. [Qwen generation guidance](https://huggingface.co/Qwen/Qwen3-8B).

The SKD original YAML uses student 0.5/0.5 and teacher 0.2/0.5 temperature/top-p, unlike the harmonized experiment. Preserve those values in an optional original-protocol profile; do not report a harmonized-sampler run as an exact reproduction. Thinking-mode experiments also need their own much larger response budget and must not mix with the initial non-thinking comparison.

For DistiLLM-2, generate pairs after fixing the train/validation split and common initialization. Record each producing checkpoint hash. Offline generation occurs once in the primary released-code-style protocol; if iterative refresh is later enabled, count it in compute and define every refresh boundary. GKD and SKD require the current training student, so detached stale vLLM replicas are not acceptable without an explicit weight-synchronization mechanism.

All-method LoRA, if needed after profiling, should use the same rank 16, alpha 32, dropout 0.05 and target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), with DoRA false. These are proposed common settings, not DistiLLM-2's original rank/alpha/DoRA recipe. Confirm each selected model exposes these modules and preserve frozen teacher behavior.
