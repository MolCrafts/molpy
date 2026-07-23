#!/usr/bin/env bash
# CI lint job parity, always in a fresh venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec bash scripts/precommit-venv-run.sh -- bash -c '
  set -euo pipefail
  ruff format --check src/ tests/
  ruff check src/ tests/
  ty check src/molpy/
'
