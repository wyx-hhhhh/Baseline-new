# Run both GPU groups and resume the whole baseline sequence

Training and evaluation are now separate: use this training command, then `bash /home/wyx/Baseline/scripts/evaluate_all.sh` for ROUGE-L. See [the two-command guide](train_evaluate.md).

Run training from any working directory:

```bash
bash /home/wyx/Baseline/scripts/run_all.sh
```

The script selects `/home/wyx/Baseline/.venv/bin/python` automatically, so a Conda switch is not needed. It starts both groups concurrently and runs methods sequentially within each group:

| Group | Student GPU | Teacher GPU | Method order |
|---|---:|---:|---|
| Qwen3-8B → Qwen3-1.7B | 0 | 1 | KD → ABKD → SKD → DistiLLM-2 |
| Llama-3-8B → Llama-3.2-1B | 2 | 3 | KD → ABKD → SKD → DistiLLM-2 |

Each group is one training process with its teacher and student on separate GPUs and FP32 optimizer state on CPU. It is not two replicated student workers. Each child sees only its two allocated GPUs, numbered locally as `cuda:0` and `cuda:1`. The Llama shared-vocabulary policy is retained. Default training seed is 42, and both groups use the already prepared Dolly split.

DistiLLM-2's teacher/student pairs are generated automatically immediately before its training stage. Generation uses the original teacher and initial student checkpoints, not the students trained by earlier methods. The pair producer seed is 42; its fixed pairs can be reused across training seeds. Existing matching pairs are verified and reused.

Preview the plan without starting jobs or creating run artifacts:

```bash
bash /home/wyx/Baseline/scripts/run_all.sh --dry-run
```

## Separate command for each model group

Run these in two terminals to train both groups concurrently:

```bash
bash /home/wyx/Baseline/scripts/run_all.sh --group qwen --qwen-gpus 0,1
bash /home/wyx/Baseline/scripts/run_all.sh --group llama --llama-gpus 2,3
```

Each command runs KD → ABKD → SKD → DistiLLM-2 sequentially for its selected family, including pair preparation and automatic resume. The script selects the project environment automatically. Group locks are independent, so these two commands can share the existing output/config/log structure. A duplicate command for the same group is rejected even if assigned different GPUs, preventing two workers from writing the same run. The combined command without `--group` remains available.

## Stop and resume

Press **Ctrl+C once** to stop. Training finishes the current optimizer step and writes a complete checkpoint before exiting. This can take time, especially with NAS checkpoint writes or a long SKD rollout. Pair generation preserves every fully saved response and can regenerate only the interrupted response when restarted. A second Ctrl+C forces worker exit, recovering only from artifacts already committed.

To resume, run **the exact same command with the same arguments**:

```bash
bash /home/wyx/Baseline/scripts/run_all.sh
```

No checkpoint path or manual stage selection is needed. The launcher:

1. Checks the resolved settings, source model identities, data hashes and runtime against the existing run manifest.
2. Skips a completed method only when its result, final export and committed checkpoint agree with the requested update budget.
3. Resumes an interrupted method from its latest fully committed checkpoint, retaining optimizer/master weights, scheduler, RNG and exact next optimizer step.
4. Preserves and archives an interrupted attempt that stopped before any checkpoint was completed, then restarts that method from its original student. Partial first-manifest writes are covered; arbitrary unidentified files are not overwritten.
5. Resumes teacher/student pair generation from its durable per-example logs, or validates the completed paired corpus. Source/model/generation-policy or generation-runtime changes are rejected for partial work.
6. Continues with the remaining methods after the interrupted method completes.

A sudden process kill resumes at the last committed checkpoint, so updates after that checkpoint may need to be repeated. Checkpoints default to every 100 optimizer steps and at the end; graceful Ctrl+C saves at the current completed step even between those intervals. Partially written checkpoints are preserved and excluded from resume selection. Model/state files and step logs are flushed before committing checkpoint markers. Safetensors structure and checkpoint ZIP metadata are checked before completed artifacts are reused; this is not a full checksum of every large tensor block. If training finished but final export was interrupted, the final checkpoint is loaded to finish exporting without another training update. The result is explicitly marked pending before re-exporting, so another interruption during that recovery cannot leave a stale completion claim.

Failures stop that group's queue; the other group continues. After resolving the failure, rerun the same command. A live launcher or orphan worker retains kernel locks, so a second launcher cannot start duplicate work on those GPUs. A stale status JSON or lock filename by itself does not count as a live process. If only the launcher was killed, a surviving worker can finish its current method; wait for that live worker to exit before relaunching.

## Logs and output

```text
/nas/Users/wyx/Baseline/
  orchestration/dolly/
    configs/                  exact resolved per-stage settings
    qwen.status.json          last observed group/stage status
    llama.status.json
    logs/qwen/                append-only stage/attempt logs
    logs/llama/
  runs/<pair>/dolly/<method>/seed_42/
    manifest.json
    checkpoints/step_XXXXXX/
    final/
    result.json
  data/dolly/<pair>/
    pairs.train.jsonl
    pairs.train.manifest.json
    pairs.train.progress.json
    pairs.train.teacher.partial.jsonl
    pairs.train.student.partial.jsonl
```

The partial generation logs remain as durable audit/recovery artifacts. They are not training inputs; only the complete paired file is consumed. Completed pair files and their checksums are verified before reuse.

For example, follow Qwen KD progress with:

```bash
tail -f /nas/Users/wyx/Baseline/orchestration/dolly/logs/qwen/kd_seed_42_train.log
```

For an SSH session, keep the launcher running after disconnecting:

```bash
nohup bash /home/wyx/Baseline/scripts/run_all.sh \
  > /nas/Users/wyx/Baseline/pipeline.log 2>&1 &
```

The existing NAS output root is already present. The foreground command remains useful when you want Ctrl+C to checkpoint and stop.

## Options

```bash
# Three training seeds per method in each group:
bash scripts/run_all.sh --seeds 42 43 44

# Select a different order or subset, still independently in both groups:
bash scripts/run_all.sh --methods skd abkd distillm2 kd

# Remap the two disjoint physical GPU pairs:
bash scripts/run_all.sh --qwen-gpus 2,3 --llama-gpus 0,1

# Example new campaign with more frequent checkpoints:
bash scripts/run_all.sh --output-root /nas/Users/wyx/Baseline/checkpoint20 --save-steps 20
```

Other supported flags include `--dataset`, `--data-root`, `--max-steps`, `--max-prompt-tokens`, `--max-new-tokens`, `--gradient-accumulation-steps`, and `--eval-steps`. A changed training budget is a new experiment rather than a continuation of the old scheduler; use a separate output root. Changing the dataset, token limits, producer seed or sampling settings also requires a separate prepared data/pair artifact location. `--data-root` contains `<dataset>/train.jsonl` and `<dataset>/validation.jsonl`; pair artifacts are shared under that dataset directory and do not follow `--output-root`.

## Verification scope

Automated tests exercise two concurrently running child processes, fixed per-group method order, isolated GPU masks and cleaned distributed-launch variables, failure isolation, complete-run skipping, latest-checkpoint selection, first-manifest recovery, graceful stop, duplicate-launch rejection and inherited orphan-worker locks. Trainer tests compare interrupted/resumed updates with uninterrupted native-model training. Pair tests cover interruption in either role, per-ID reproducibility, torn writes, immutable identity, corruption rejection and interrupted final publication.

Integration tests run the actual trainer and pair-generator subprocesses for **all eight tiny Qwen/Llama jobs**, first on CPU and then on the server's **four RTX 3090 GPUs** with BF16 and the requested 0–1 / 2–3 grouping. They verify actual model devices, finite losses/exports and two CUDA RNG states per worker. Rerunning skips all eight training jobs, loads no models, and preserves generated files. [Four-GPU test evidence](evidence/parallel-gpu-smoke.json). These are tiny models and short synthetic data; no full-size, full-corpus experiments were launched while implementing the scheduler.

Final verification: the full 244-test regression pass succeeded; after the final recovery refinement, four additional export-recovery cases and the actual four-GPU pipeline were rerun successfully. The current suite contains 249 cases. Logs: [full regression](evidence/parallel-full-tests.log), [final recovery](evidence/parallel-final-recovery-tests.log), [final GPU run](evidence/parallel-final-gpu-test.log).
