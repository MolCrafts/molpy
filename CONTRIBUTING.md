# Contributing to MolPy

Thank you for your interest in contributing to MolPy!

The full contributor handbook lives in the documentation — this file is just the
entry point, so the rules have a single home and cannot drift:

- **[Development Setup](https://docs.molcrafts.org/molpy/developer/development-setup/)** — clone, editable install, pre-commit hooks, running tests
- **[Contributing Workflow](https://docs.molcrafts.org/molpy/developer/contributing/)** — branches, conventional commits, the PR checklist
- **[Coding Style](https://docs.molcrafts.org/molpy/developer/coding-style/)** — ruff formatting, type hints, Google-style docstrings, the mutation contract
- **[Testing](https://docs.molcrafts.org/molpy/developer/testing/)** — pytest conventions, markers, coverage expectations
- **[Architecture Overview](https://docs.molcrafts.org/molpy/developer/architecture-overview/)** — module map and extension points, read before larger changes

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/molpy.git
cd molpy
uv sync --extra dev
prek install --hook-type pre-commit --hook-type pre-push
uv run --no-project --with 'tox>=4.23' --with ruff==0.16.1 --with ty==0.0.65 tox -e lint
uv run --locked --extra dev python -m pytest tests/ -n auto
```

## Hooks

`prek install --hook-type pre-commit --hook-type pre-push` (or the same with
`pre-commit`) installs both hook types from `.pre-commit-config.yaml`; every
CI job has a hook running the same command. **Never `git commit --no-verify`
or `git push --no-verify`, and never merge a red pull request.**

- **pre-commit** (staged files, cheap, in place): file hygiene (whitespace,
  final newline, YAML/TOML/JSON, merge markers, large files) and `tox -e lint`
  (ruff format + ruff check + ty), as ci.yml `lint`.
- **pre-push**:
  - the pre-commit hooks again on `--all-files`;
  - `scripts/partners.py check` — no `[tool.uv.sources]` path entry (a CI
    runner has no sibling checkout to point at; release.yml refuses one);
  - `uv lock --check` (ci.yml `lint`) when pyproject.toml or uv.lock changed;
  - the molrs pin resolves to a published PyPI release;
  - the docs build, as Cloudflare Pages builds it (`.[doc]` in a fresh env,
    then `zensical build --clean --strict` in a clean copy of the tree), when
    docs/, src/, zensical.toml or pyproject.toml changed;
  - the unit suite, `uv run --locked --python 3.12 --extra dev python -m
    pytest tests/ -n auto` (ci.yml `test`; `uv.lock` is committed, so a stale
    lock fails here as it does in CI).
- **Dispatch on the MolCrafts cluster:** the unit suite is the one heavy
  hook. Its entry goes through `scripts/hook-run.sh`, which hands the command
  to `$MOLCRAFTS_HOOK_RUNNER` when that is set and it is not already inside a
  Slurm job. The cluster's shared `core.hooksPath` sets it to
  `.build-alloc/hookrun`, which runs the command on a compute node (allocation
  `$USER-hooks`; it fails after 20 minutes without a node, never passes).
  Everything else runs in place, so a commit never waits for Slurm. Elsewhere
  nothing sets the variable and every hook runs locally.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to
follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation:** [docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/)
