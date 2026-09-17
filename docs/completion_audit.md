# Request completion audit

> Update: the user subsequently selected `/nas/Models/Meta-Llama-3-8B-Instruct`. Missing Llama-3.1 references below describe the earlier request. Current launch instructions and verification are in [llama3_launch.md](llama3_launch.md).

The requested code preparation, environment inspection, dataset analysis, and documented launch workflow are delivered. Complete corpus experiments were not requested to be launched during preparation; only explicitly labelled smoke runs were performed. The missing Llama teacher is a reported external prerequisite, not a substituted model or a claimed successful Llama full-size run.

| Requirement | Current evidence |
|---|---|
| Read check reports and preserve current four-method scope | docs/implementation.md; docs/README.md |
| Exact two local Instruct model pairs and eight configs | docs/evidence/config-matrix.json |
| Check current environment and specify required versions/packages | docs/environment.md; requirements-training.txt; requirements-training.lock; docs/environment-pinned.json |
| Implement preparation, generation, distinct losses/sampling, training, saving and resume | baseline_common/; scripts/; 115 pinned-stack tests in docs/validation.md |
| Write models/checkpoints beneath requested NAS root | All eight resolved configs; actual Qwen exports listed in docs/evidence/qwen-kd-smoke-result.json and qwen-method-smoke.json |
| Plan sensible folder structure and runnable commands | docs/README.md |
| Identify original training datasets and user data preparation needs | docs/datasets.md; docs/data-verification.json |
| Place new analysis/documentation under ~/Baseline/docs | docs/ contains all new reports/runbooks/evidence |

Known prerequisites/limits remain visible in the main runbook: supply the empty Llama-3.1 teacher directory; generate corpus-specific DistiLLM-2 pairs before its formal run; select the desired corpus and full run budget; validate full-budget memory and full-size NCCL before using that optional profile. Native tiny Qwen3/Llama tests, actual short Qwen checks, and environment checks are separately identified.
