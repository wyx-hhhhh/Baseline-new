#!/usr/bin/env bash
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASELINE_ROOT"
exec "$BASELINE_ROOT/.venv/bin/python" "$BASELINE_ROOT/scripts/run_parallel.py" "$@"
