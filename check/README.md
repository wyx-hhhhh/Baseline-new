Investigation completed on 2026-09-16 for the four local Git checkouts.

Start with [the investigation report](investigation_report.md), then use [the implementation plan](codex_modification_plan.md) for subsequent code changes.

- [Unified configuration specification](unified_configuration.md): proposed Python/PyTorch stack, model pairs, baseline definitions, and experiment controls.
- [Machine-readable experiment specification](experiment_spec.json): ten baseline/model-pair combinations; a proposal for a future runner, not an existing repository's launch configuration.
- [Core training constraints](constraints-training-proposed.txt): exact migration targets, not a complete resolved lockfile.
- [Validation and evidence](validation_record.md): what was checked, what failed, and what still requires the GPU server.
- [Local audit results](evidence/audit_results.json) and [public metadata](evidence/public_metadata.json): commits, source URLs, model revisions, file hashes, and package requirements.

Recommended migration target: **Python 3.11.13, PyTorch 2.6.0 with CUDA 12.4, Transformers 4.51.3**. These are deliberately selected research compatibility pins, not a claim about the newest releases or an environment already validated for training. All five baselines should use the same training stack after the planned repairs.

The literal Llama request is recorded as `meta-llama/Meta-Llama-3-8B` → `meta-llama/Llama-3.2-1B`. There is no original Llama 3 release with a 1B model. Llama 3.1-8B is documented as an alternative teacher, not silently substituted. Base versus Instruct variants and the target server remain open experiment choices.

Only this `check` directory was written. Repository training code, the installed Python environment, and model weights were not changed. The evidence scripts download public metadata/tokenizers and run small CPU checks; they do not run experiments.
