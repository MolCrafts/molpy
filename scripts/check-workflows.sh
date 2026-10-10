#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset $(git rev-parse --local-env-vars)
ref=$(sed -n 's/^CI_REF=//p' .github/partners.env)
[[ "$ref" =~ ^[0-9a-f]{40}$ ]] || { echo "CI_REF must be an immutable commit" >&2; exit 1; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q "$work"
git -C "$work" fetch -q --depth 1 https://github.com/MolCrafts/molcrafts-ci.git "$ref"
git -C "$work" checkout -q --detach FETCH_HEAD
env -u VIRTUAL_ENV uv run --no-project --script "$work/actions/check-workflows/check_workflows.py" "$PWD"
