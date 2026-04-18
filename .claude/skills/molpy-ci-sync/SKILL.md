---
name: molpy-ci-sync
description: Audit and fix CI/pre-commit parity. Patches .pre-commit-config.yaml and .github/workflows/ci.yml so every check runs both locally at commit time and in CI. Works on existing and new projects.
argument-hint: "[path to project root, defaults to current directory]"
user-invocable: true
---

# CI / Pre-commit Sync

Read CLAUDE.md for project conventions.

Project root: $ARGUMENTS (default: current directory)

## Procedure

### Step 1 — Detect project shape

```bash
# Check what exists
ls pyproject.toml setup.cfg setup.py 2>/dev/null   # Python project?
ls .pre-commit-config.yaml 2>/dev/null              # pre-commit present?
ls .github/workflows/*.yml 2>/dev/null              # CI present?
ls docs/ mkdocs.yml 2>/dev/null                     # docs present?
find . -name "*.ipynb" -maxdepth 3 | head -3        # notebooks present?
```

Classify as:
- **new-project** — no `.pre-commit-config.yaml` AND no `.github/workflows/`
- **existing-partial** — one exists but not the other
- **existing-both** — both exist, proceed to audit

### Step 2 — Audit (existing projects)

Invoke `molpy-ci-auditor` agent on the project root.

For **new-project**: skip to Step 4 (scaffold).

### Step 3 — Fix gaps (existing projects)

For each `CI_ONLY` gap reported by `molpy-ci-auditor`:

**Missing `ty check` in pre-commit** → append to `.pre-commit-config.yaml`:
```yaml
- repo: local
  hooks:
    - id: ty-check
      name: ty check
      entry: ty check src/molpy/
      language: system
      pass_filenames: false
      types: [python]
```

**Missing `pytest` in pre-commit** → append under the `local` repo block:
```yaml
    - id: pytest
      name: pytest (fast)
      entry: pytest tests/ -x -q -m "not external" --ignore=tests/test_mcp --no-header -rN
      language: system
      pass_filenames: false
      stages: [pre-commit]
```

**Version drift (ruff unpinned in CI)** → add version pin to CI install step:
```yaml
run: pip install ruff==<version-from-pre-commit-rev>
```

For each `PRECOMMIT_ONLY` gap: flag to user as advisory — these are local-only hooks (nbstripout, file hygiene) that are intentionally absent from CI. No action needed unless the user wants otherwise.

After patching, run:
```bash
pre-commit run --all-files 2>&1 | tail -20
```

Report exit code and any new failures introduced by the added hooks.

### Step 4 — Scaffold (new projects)

If **new-project**, write the canonical configs:

**`.pre-commit-config.yaml`** (canonical MolPy standard):
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
        args: ['--unsafe']
      - id: check-added-large-files
        args: ['--maxkb=1024']
      - id: check-merge-conflict
      - id: mixed-line-ending
      - id: check-toml
      - id: check-json

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.12
    hooks:
      - id: ruff-format
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: local
    hooks:
      - id: ty-check
        name: ty check
        entry: ty check src/<package>/
        language: system
        pass_filenames: false
        types: [python]

      - id: pytest
        name: pytest (fast)
        entry: pytest tests/ -x -q -m "not external" --no-header -rN
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

Replace `src/<package>/` with the actual package name.

**`.github/workflows/ci.yml`** (canonical MolPy standard):
```yaml
name: CI

on:
  push:
    branches: [master, main, dev, dev-*]
    tags: ['v*']
  pull_request:
    branches: [master, main]
  workflow_dispatch:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install ruff==0.11.12 ty
      - run: ruff format --check src/ tests/
      - run: ruff check src/
      - run: ty check src/<package>/

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.12', '3.13']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v -m "not external" --cov=src/<package> --cov-report=xml
      - if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

After scaffolding, run:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files 2>&1 | tail -20
```

### Step 5 — Verify and report

Re-run `molpy-ci-auditor` to confirm zero gaps remain.

Report:
```
CI/PRE-COMMIT SYNC COMPLETE
============================
Changes made:
  ✅ Added ty-check hook to .pre-commit-config.yaml
  ✅ Added pytest hook to .pre-commit-config.yaml
  ⚠️  ruff version drift — manual fix needed (see above)

Post-patch pre-commit run: ✅ all passed / ❌ N failures (details below)

Remaining gaps: 0
```

## Escape hatch

Hooks can be skipped per-commit without disabling them permanently:
```bash
SKIP=pytest git commit -m "wip: mid-feature"
SKIP=ty-check,pytest git commit -m "wip: draft"
```

## Output

Summary table of changes made + post-patch pre-commit result. One-line footer: `CI sync complete — N changes applied, N gaps remaining`.
