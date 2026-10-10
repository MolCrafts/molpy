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
`pre-commit`) installs both hook types from `.pre-commit-config.yaml`; every
CI job has a hook running the same command. **Never `git commit --no-verify`
or `git push --no-verify`, and never merge a red pull request.**

- **pre-commit** (staged files, cheap, in place): file hygiene (whitespace,
  final newline, YAML/TOML/JSON, merge markers, large files) and the pinned lint tools
  (ruff format + ruff check + ty), as lint.yml `lint / hooks`.
- **pre-push**: `scripts/check.py verify` reruns hygiene/static checks on all
  files, then checks dependencies, tests and strict docs in one cached sibling
  layout. Every dependency comes from the full SHA in `.github/partners.env`;
  the resolver is shared with CI at `CI_REF`. `uv lock --check`, locked sync and
  `uv pip check` catch stale metadata and incompatible installed dependencies.
  No changed-file filters skip a gate. Native CI also tests Windows and macOS.
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

One workflow per kind of work. Every push of any branch runs `lint`, `test`
and `docs`, on a fork as on MolCrafts. A pull request into `dev`, `master` or
`main` runs them again, except one inside a fork (its push already ran the
full tier).

| workflow | feature-branch push to MolCrafts | everything else: any push to a fork, `dev`/`master`/`main`, pull requests, tags, dispatches | upstream only |
| --- | --- | --- | --- |
| `lint.yml` | `lint / hooks` (hygiene, static lint and workflow scheme on every file) | same | — |
| `test.yml` | fast: `test / python (ubuntu-latest, 3.12)` | full: `test / python` on all three OSes with Python 3.12, plus Linux/Python 3.14 | — |
| `docs.yml` | `docs / build` (zensical `--strict`) | same | Cloudflare Pages deploys the site from MolCrafts |
| `nightly.yml` | — | — | nightly: test and coverage snapshots to molcrafts-ci; a `nightly` branch push: `molcrafts-molpy-nightly` |
| `release.yml` | — | dispatch: dry run (tests, builds, uploads nothing) | `v*` tag: PyPI and the GitHub Release ([release process](docs/developer/release-process.md)) |

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

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to
follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation:** [docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/)
