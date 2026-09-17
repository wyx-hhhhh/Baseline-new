#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASELINE_ROOT"
if [[ "${1:-}" == "--validation-only" ]]; then
  shift
  exec "$BASELINE_ROOT/.venv/bin/python" "$BASELINE_ROOT/scripts/run_evaluation.py" "$@"
fi
exec "$BASELINE_ROOT/.venv/bin/python" "$BASELINE_ROOT/scripts/run_nl_evaluation.py" "$@"
