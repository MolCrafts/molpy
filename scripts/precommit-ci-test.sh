#!/usr/bin/env bash
# CI test job parity, always in a fresh venv.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

dir=tests/tests-data
if [ ! -e "$dir/README.md" ]; then
  mkdir -p "$dir"
  curl -fsSL https://github.com/molcrafts/tests-data/archive/refs/heads/master.tar.gz \
    | tar -xz -C "$dir" --strip-components=1 --exclude='*/con' --exclude='*/con/*'
fi

exec bash scripts/precommit-venv-run.sh --dev -- \
  pytest tests/ -m "not external" -n auto
