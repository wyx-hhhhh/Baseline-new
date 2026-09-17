# Parallel launcher review — 2026-09-16

Scope: read-only review of `baseline_common/orchestration.py`,
`scripts/run_parallel.py`, `scripts/run_all.sh`, pair-generation progress, and
their contract with the trainer. No full-size training job was launched.

## Confirmed design behavior

- The two queues have separate child processes and GPU masks. Each child sees
  its student as logical GPU 0 and its teacher as logical GPU 1. Distributed
  launch variables are removed from the child environment.
- Baselines run sequentially within a queue. A failed or interrupted stage
  cannot advance that queue; the other queue can finish independently.
- Recovery validates the current run manifest before skipping or resuming.
  Resume keeps the original schedule and selects a committed checkpoint.
- Checkpoint model/tokenizer files, optimizer/scheduler state, per-rank RNG state,
  and step logs are flushed to storage before the completion marker is published.
  Final model exports are likewise flushed before their marker. Directory entries
  are flushed too; existing checkpoint siblings are not recursively traversed.
- The trainer handles SIGINT/SIGTERM at optimizer-step boundaries, preserves
  scheduler/optimizer/RNG state, and reports incomplete interruption with exit
  code 75. Restoring legacy CUDA RNG lists preserves the currently used logical
  devices and ignores saved states for now-invisible unused devices.
- Worker processes inherit GPU and campaign lock descriptors. Killing the
  launcher does not release an orphan worker's locks prematurely. A second
  launcher cannot duplicate its work while that worker remains alive.
- Pair generation commits each completed row, resumes missing rows using
  deterministic per-record seeds, preserves torn trailing bytes before repair,
  and verifies complete sidecars and output against manifest checksums.

## Findings supplied to implementation owner

These observations describe the review snapshot; regression tests and the final
implementation determine whether each finding remains applicable.

1. **First-manifest interruption — resolved during review:** a SIGKILL during the first atomic manifest
   write can leave only `.manifest.json.<random>.tmp` in the run directory. At
   initial review time, the coordinator rejected any nonempty directory without
   `manifest.json`, preventing automatic restart of this narrowly identifiable
   interrupted attempt. The implementation now archives such manifest-temp-only
   attempts, while continuing to reject arbitrary unknown output contents. An
   isolated temporary-directory probe verified that this case returns `restart`
   and preserves the original directory as an archived attempt.

2. **Nonempty files were insufficient completion evidence — resolved:** an isolated
   temporary fixture containing valid completion markers but one-byte model,
   optimizer, and RNG files initially passed the completion checker. The checker
   now validates safetensors headers/offsets, shard-index mappings, model/tokenizer
   structure, and PyTorch ZIP container metadata. It checks metadata CRCs without
   rereading every large tensor block; this is structural validation, not a full
   content checksum of all model/optimizer tensors. The new trainer's final export marker is
   required when its result includes the `interrupted` field; legacy completed
   exports remain supported.

3. **Runtime identity for pair generation — resolved:** pair identities cover checkpoints,
   input/validation hashes, tokenizer, seed, vocabulary, dtype, and generation
   settings, and now the PyTorch/Transformers/tokenizers package versions.
   Changing these packages during partial generation is rejected. Already complete
   legacy pair artifacts without recorded versions remain reusable after artifact
   validation, because reuse adds no rows from a different runtime.

## Verification context

Before this review, the trainer changes passed 36 single-process tests and eight
two-CPU-rank tests. Those include real native tiny Llama/Qwen training, signals
inside accumulation and checkpoint writing, a separate process exiting 75 then
resuming exactly, and recovery of a missing final export without extra updates.
The review used only source inspection and isolated temporary artifact probes;
these probes do not establish real-model GPU throughput or full-run quality.
After adding storage flushes, 17 targeted tests passed, including assertions that
all model/state files are synced before their commit marker and that a failed
flush leaves no committed checkpoint. Additional launcher and pair-generation
regression coverage is recorded with the final implementation validation.
