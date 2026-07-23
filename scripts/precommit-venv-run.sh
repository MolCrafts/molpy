#!/usr/bin/env bash
# Run a command inside a brand-new throwaway venv (CI-parity: clean install).
# Usage: scripts/precommit-venv-run.sh [--dev] [--doc] -- <cmd> [args...]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_DEV=0
WITH_DOC=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) WITH_DEV=1; shift ;;
    --doc) WITH_DOC=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--dev] [--doc] -- <command> [args...]" >&2
  exit 2
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/molpy-precommit-XXXXXX")"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

python -m venv "$TMP/venv"
# shellcheck disable=SC1091
source "$TMP/venv/bin/activate"
python -m pip install -q -U pip

# Always pin lint tools to CI versions (see .github/workflows/ci.yml).
python -m pip install -q "ruff==0.15.13" "ty==0.0.37"

if [[ "$WITH_DEV" -eq 1 ]]; then
  python -m pip install -q -e ".[dev]"
fi
if [[ "$WITH_DOC" -eq 1 ]]; then
  python -m pip install -q -e ".[doc]"
fi

exec "$@"
