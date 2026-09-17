# Independent implementation review

> Update: the user subsequently selected `/nas/Models/Meta-Llama-3-8B-Instruct`. Missing Llama-3.1 references below describe the earlier request. Current launch instructions and verification are in [llama3_launch.md](llama3_launch.md).

Reviewed 2026-09-16 during implementation. This records the bounded review of `baseline_common/config.py`, `models.py`, `train.py`, `optim.py`, `scripts/train.py`, and `scripts/run_matrix.py`. It is not a claim of completed full-size training. Subsequent regression/GPU evidence belongs in the validation record.

## Findings and resolution

1. **Resume identity must be bound to the selected checkpoint.** The reviewed trainer compared the current output directory's manifest to the requested config, then accepted a separately supplied `--resume` checkpoint containing `complete.json`. It did not prove that checkpoint came from the same run/manifest. A foreign but architecturally compatible student/optimizer could therefore be loaded under the current run's provenance. Require an in-run checkpoint path or persist and verify its manifest hash.
2. **Resuming an old checkpoint needs an explicit policy.** When later checkpoints/log records already exist, loading an earlier step would append duplicate step metrics and later collide with an existing checkpoint directory. Reject stale continuation before model loading, or implement an explicit fork/rollback that preserves clear lineage.
3. **Preflight and trainer should agree about differing chat templates.** The environment checker initially required identical chat templates; the trainer intentionally permits different templates after verifying the full token map and tokenizer machinery and uses the student's rendering for both roles. A shared-student-template protocol can be valid, but the checker must not silently apply a different readiness rule. Template differences should be reported and the selected protocol recorded.

Follow-up code inspection confirms that resume now requires an in-run latest completed checkpoint and backs up uncommitted logs/partial checkpoint artifacts before continuation. The checkpoint state also retains `best_step`. These changes address findings 1–2; final behavioral evidence belongs in the implementation tests. Defensive parsing now also handles a partially written trailing JSON log row; regression tests verify that the complete original bytes are backed up and committed steps resume exactly. For finding 3, the selected protocol is explicitly **the student's template for both roles**; the checker now reports `chat_template_equal` as informational, while token mapping/machinery and generation EOS equality remain required. The exact local Qwen templates do match.

## Checks without a defect found

- The DDP arithmetic normalizes local token-loss sums by `world_size / global_valid_token_count`, followed by DDP's averaging. DistiLLM-2 uses global pair counts after summing the two per-response means. Dummy branches on ranks with fewer real examples keep backward/synchronization counts aligned. Subsequent two-rank native Qwen3/Llama tests passed for all four methods, including gradient checkpointing and a rank with no real tail example; see validation.md.
- The teacher is frozen and in evaluation mode, and its training forward executes under no-grad. Student training forwards and backward execute with gradients. The separate teacher/student devices suit the four 24 GiB GPUs; CPU FP32 optimizer masters/moments avoid forcing all Adam state onto a student GPU.
- `CPUAdamW` retains FP32 masters between updates and checkpoints; copying the rounded BF16 model parameters back into masters each step would lose small updates, and the reviewed implementation avoids that mistake.
- Causal loss uses prediction at token position `t` for target `t+1` and selects response positions from shifted labels. Real EOS targets remain active; padding/prompt positions use `-100`.
- Current-student SKD generation precedes its corresponding update, and generation mode changes are restored. Teacher stop IDs come from its generation config plus the shared tokenizer's EOS. The verified local Qwen teacher/student stop sets agree.
- DistiLLM-2 checks generating teacher/student artifact fingerprints and generation settings, joins by stable IDs, and checks exact stored prompt IDs against the current renderer. Fingerprints contain metadata hashes plus weight size/mtime; they are explicitly not complete weight-content hashes.

## Real local tokenizer probe

With `/software/anaconda3/envs/RAGAny/bin/python` and Transformers 4.57.1, the actual Qwen 1.7B tokenizer rendered generation prompts with `enable_thinking=False`. For responses `4`, a leading-space response, and Chinese text, the concatenated reference-token representation matched the official full chat template through the EOS token; the official template additionally emits a trailing newline after EOS, which is excluded from the training target here.

For a response beginning with a newline, the official Qwen template strips that leading newline while the reviewed reference encoder preserves the raw response. This is a protocol distinction to document/test, not proof of a causal-indexing defect. References containing literal special-token text require an intentional interpretation. The Llama student tokenizer loaded and rendered successfully, but the requested Llama teacher folder is empty, so its pair compatibility remains unverified.

## Scope limits

Memory accounting is not a peak-VRAM measurement. Full output logits still exist even though loss intermediates are recomputed in exact token chunks. A short real Qwen smoke test, followed by a workload-length memory check, is needed before claiming 1,024+1,024-token training fits. The reference SKD sampler recomputes prefixes without a KV cache and can be slow; generation token/forward counters make that cost visible. No semantic equivalence to the original repositories' full training pipelines is established solely by this review.
