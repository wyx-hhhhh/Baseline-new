# Implementation and experiment protocol

`baseline_common` is a portable replacement training path built from the algorithm definitions and the archived source implementations. It does not import the old trainers or run their incompatible installation scripts. The existing repositories remain readable source references. The old `check` plan's five-method/ten-configuration target has been superseded by the user's four-method/eight-configuration request.

## Distinct methods

| Method | Training sequence | Objective | Original implementation evidence |
|---|---|---|---|
| KD | Fixed reference response, shared with ABKD | Full-vocabulary forward KL, teacher → student | `distillm/distillm/losses.py`, `distillm/finetune.py` |
| ABKD | Same fixed references | Alpha-beta divergence, alpha=.1, beta=.8; analytic singular limits supported | `abkd/distillation_llm/distillm/losses.py`, Qwen `scripts/qwen/ab/train_3B_7B_teacher.sh` |
| SKD | Fresh current-student proposals, teacher top-25 acceptance and teacher correction | Forward KL on the accepted/corrected trajectory | `speculative_kd/speculative_kd/train/ddp_skd.py`, custom `transformers/utils.py` |
| DistiLLM-2 | Offline teacher chosen response + initial-student rejected response | Released two skew divergences, alpha1=alpha2=.1; detached student contribution in reverse mixture | `distillm-2/src/distillm_trainer.py::get_batch_logps/dpo_loss`, `generate/reformat.py` |

All four update the full student and keep the teacher frozen/eval. KD reports true KL; the old helper reported soft-target cross entropy. They have equal student gradients at T=1 but differ by teacher entropy. Distribution math is FP32 and uses every model-head vocabulary entry, including the padded Qwen vocabulary tail. Alpha-beta math includes alpha=0, beta=0 and alpha+beta=0 limits. Nonfinite losses or gradients fail visibly.

DistiLLM-2 computes the mean divergence per response, sums chosen and rejected, then averages pairs. Other methods divide summed divergences by actual valid response tokens across the full optimizer step and all DDP ranks. These reductions are intentionally different. Adaptive alpha, gradual beta, DoRA, replay and reference log-prob precomputation are not enabled in this protocol. The fixed defaults match the released DistiLLM-2 branch; paired loss gradient tests verify the detached reverse mixture.

## Causal and tokenizer contract

Prompt and padding labels are `-100`; real EOS labels are retained, even when pad=EOS. Each label at position t+1 supervises logits at t exactly once. The first answer token and final EOS both contribute. Token IDs stay ordinary signed integer tensors/JSON integers; no 65535 or unsigned `-1` sentinel is used.

The default `full` policy requires equal complete token→ID maps, tokenizer encoding machinery and BOS/EOS semantics, plus equal actual output-head sizes. Qwen retains that policy. The updated Llama pair uses the explicit `llama3_shared` contract below. We use the **student chat template for both models**, with `enable_thinking=False` for Qwen. Different teacher/student template text is therefore informational; different token semantics is an error. The checkpoint metadata fingerprint includes tokenizer and generation files. Checkpoint weights are identified by path, filenames, sizes and modification times; **this is not a full cryptographic hash of all weight bytes**. Do not modify source checkpoints during experiments.

Reference sequences concatenate the exact generation prefix, separately encoded reference response, and EOS. Raw response whitespace is preserved. Overlong user content is shortened while retaining the chat header and assistant prefix. Reference responses are limited to the configured response budget including EOS. Generated responses retain native returned IDs and stop IDs; a response that exhausts the generation budget is not assigned a fabricated EOS. Offline pairs must match the current rendering and generation settings before they can train.

## Corrected SKD port

The sampler implements one unpadded prompt at a time, constant proposal blocks of five, actual emitted-token acceptance, early termination at EOS, and discard of the unaccepted block suffix on correction. Teacher/student temperature and top-p/top-k settings are independent. Acceptance ranks raw teacher logits; correction sampling applies the teacher generation filters. This avoids the original implementation's re-sampled proposal mismatch and arbitrary zero-probability top-k ties after warping. No teacher bonus token is appended after a fully accepted block.

The implementation recomputes prefixes and does not maintain KV caches between proposal blocks. It is a correctness-oriented complete SKD algorithm, with potentially substantial compute overhead for long outputs; it is not a claim to reproduce the original optimized throughput. Forward calls, processed prefix tokens, proposals, acceptances and interventions are counted. The installed Transformers generation code is never patched. Standard Hugging Face assisted decoding is not substituted for SKD.

## Memory and precision

The default 24 GB GPU profile places the BF16 teacher and student on separate GPUs. Student gradient checkpointing and checkpointed token-chunk loss intermediates reduce memory. Full model logits still exist; chunking does not eliminate that allocation or approximate the distribution.

`CPUAdamW` maintains FP32 master parameters, FP32 gradients on CPU during the optimizer step, and FP32 Adam moments; model parameters and autograd gradients are BF16 on GPU. Each update copies GPU gradients to CPU and updated master weights back to GPU. This supports full fine-tuning without silently changing the experiment to LoRA or quantizing the teacher. The straightforward synchronous copies/CPU Adam can be slower than fused distributed offload implementations. Native `torch.optim.AdamW` is used internally; no DeepSpeed build is required. If `optimizer_offload=false`, Adam states use the parameter dtype and GPU memory usage rises; that setting is a separate precision/resource variant.

The server has four 24 GB GPUs, not a single 96 GB address space. Optional two-rank DDP reserves two student GPUs and two replicated-teacher GPUs. It does not use ZeRO or model sharding. The single-process path needs two GPUs with the default placement. Tiny CPU tests use explicit `device=teacher_device=cpu` and FP32.

## Reproducibility and outputs

A manifest captures resolved config, checkpoint metadata identity, exact train/validation file hashes, runtime package versions, world size and checkpoint metric. Per-step logs record objective, gradient norm and learning rate. Checkpoints contain optimizer/master weights, scheduler, exact next step, counters and per-rank random state; the completed marker is written last. Resume requires identical settings/provenance/runtime/world size and the latest completed checkpoint in that run. Source models are never overwritten.

The natural-language metric is generated-response ROUGE-L F1 on held-out rows. Default training skips validation; the separate evaluation command scores final exports. Optional training validation maximizes ROUGE-L. Best-checkpoint metadata and the final HF export are separate. Method losses are not directly interchangeable scores; final downstream task quality is not established by these infrastructure smoke tests. Full benchmarks, original-paper reproduction, optimized SKD throughput and hyperparameter tuning remain separate experimental work.

Original references: [SKD paper](https://arxiv.org/abs/2410.11325), [DistiLLM-2 paper](https://arxiv.org/abs/2503.07067). Detailed attribution and source locations are preserved in `check/investigation_report.md`; the executable algorithm comments reference the local source files.

## Llama-3 / Llama-3.2 shared-token contract

The original Llama-3-8B teacher and Llama-3.2-1B student have identical BPE vocabulary/merges for all 128,000 ordinary tokens, but 249 special-token slots have different contents. The runner therefore does not compare those full heads blindly. `vocabulary_policy=llama3_shared` verifies the complete ordinary-token encoder plus identical BOS, end-of-text, header delimiters and end-of-turn IDs. It retains IDs 0–127999 and 128000, 128001, 128006, 128007, 128009 (128,005 categories total), excluding all 251 other reserved/control slots.

For every method, teacher and student logits are projected to that same support before normalization; KD/ABKD/skew loss formulas are otherwise unchanged. ROUGE-L evaluation generates on the same shared support and compares decoded continuations with untruncated references. This is conditional distillation on the full **shared** vocabulary, a declared protocol change from distilling all 128,256 output slots. It is not a learned/top-k approximation. The original model heads and embeddings are not resized or rewritten.

Native generation and both SKD proposal/teacher acceptance-correction paths suppress the excluded slots in the original ID space. The student-only end-of-message stop ID 128008 is excluded; shared stop IDs remain 128001 and 128009. Inputs and stored pairs containing unsupported control IDs fail explicitly. All ordinary language tokens remain available. The validated policy is recorded in run/pair provenance and exported model/tokenizer metadata; native HF generation exports retain the suppression and stop configuration. Only this explicitly selected Llama policy permits the checked differences; the default exact-map guard still rejects incompatible tokenizers.
