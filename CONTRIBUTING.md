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
uvx pre-commit install
python scripts/check.py verify
```

## Hooks

`prek install --hook-type pre-commit --hook-type pre-push` (or the same with
`pre-commit`) installs both hook types from `.pre-commit-config.yaml`. Every
check runs exactly once: static checks run only in the hooks (CI does not
repeat them), and the test suite runs once locally and once per CI leg. **Never `git commit --no-verify`
or `git push --no-verify`, and never merge a red pull request.**

- **pre-commit** (cheap, in place): the CI profile's commit stage, the
  workflow scheme (check-workflows) and the pinned lint tools (ruff format +
  ruff check + ty). Only here: no CI job runs them.
- **pre-push**: `scripts/check.py verify` runs the commit stage once on all
  files, then checks dependencies, tests and strict docs in one cached sibling
  layout. Every dependency comes from the full SHA in `.github/partners.env`;
  the resolver is shared with CI at `CI_REF`. `uv lock --check`, locked sync and
  `uv pip check` catch stale metadata and incompatible installed dependencies.
  On a push to `master` or `nightly` it also tests the declared PyPI ranges
  independently of path sources; it always builds and checks the release
  distributions (`package`, which no CI job repeats). No changed-file filters skip a gate.
  Native CI also tests Windows and macOS.
- **Dispatch on the MolCrafts cluster:** the docs build and the unit suite
  compile molrs. Their entries go through `scripts/hook-run.sh`, which hands
  the command to `$MOLCRAFTS_HOOK_RUNNER` when that is set and it is not
  already inside a Slurm job. The cluster's shared `core.hooksPath` sets it to
  `.build-alloc/hookrun`, which runs the command on a compute node (allocation
  `$USER-hooks`; it fails after 20 minutes without a node, never passes), and
  sets `$MOLCRAFTS_PARTNER_CACHE` so the partner checkouts and their builds stay
  warm between pushes. Everything else runs in place, so a commit never waits
  for Slurm. Elsewhere nothing sets the variables and every hook runs locally,
  in the same cached partner layout.

## CI

CI runs only what a laptop cannot: the cross-OS tests (`test.yml`) and the
release artifacts (`release.yml`). Every push of any branch runs `test`, on a
fork as on MolCrafts. A pull request into `dev`, `master` or `main` runs it
again, except one inside a fork (its push already ran the full tier).

| workflow | feature-branch push to MolCrafts | everything else: any push to a fork, `dev`/`master`/`main`, pull requests, tags, dispatches | upstream only |
| --- | --- | --- | --- |
| `test.yml` | fast: `test / python (ubuntu-latest, 3.12)`, which also builds the docs (zensical `--strict`) in its tested environment | full: `test / python` on all three OSes with Python 3.12, plus Linux/Python 3.14; the docs in the Linux 3.12 leg | Cloudflare Pages deploys the site from MolCrafts |
| `nightly.yml` | — | — | nightly: test and coverage snapshots to molcrafts-ci; a `nightly` branch push: `molcrafts-molpy-nightly` |
| `release.yml` | — | dispatch: dry run (guard, published check, build, uploads nothing) | `v*` tag: a guard that the commit's `test.yml` is green (no lint or test matrix of its own), one `published` run (Linux, Python 3.12, partners from PyPI), then PyPI and the GitHub Release ([release process](docs/developer/release-process.md)) |

So a fork branch gets the full tier on its push: push to your fork, wait for
green, then open the pull request into MolCrafts `dev`. Branches pushed to
MolCrafts itself (Dependabot's) get the fast tier, and their pull requests the
full one. The `require-green-ci` (`dev`) and `protect-master` rulesets require
the full tier's jobs and `test / context`. Every workflow's first job,
`<file> / context`, runs the pinned `MolCrafts/molcrafts-ci/actions/ci-context`,
which decides the tier, fork vs upstream and the pull-request dedup; the other
jobs read its outputs. Shared setup is molcrafts-ci's
pinned `MolCrafts/molcrafts-ci/actions/<name>` (`setup-rust`, `setup-python`,
`setup-partners`).

Each `test / python` job runs the test gate as
`scripts/check.py --report .ci-out test`, which also writes JUnit and coverage
files (pytest-cov over `molpy`), and the pinned
`MolCrafts/molcrafts-ci/actions/report` puts their passed, failed and skipped
counts and line and branch coverage in the run and pull request summary. This
is a report only: there are no thresholds, and it never fails the job.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to
follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation:** [docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/)
