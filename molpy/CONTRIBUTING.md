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

molpy lives in the molrs repository, at `molpy/`. Fork that repository, then:

```bash
git clone https://github.com/YOUR_USERNAME/molrs.git
cd molrs
uvx pre-commit install
scripts/check.sh molpy molpy_docs
```

## Hooks and CI

molpy uses the molrs repository's hooks and CI; see that repository's
`AGENTS.md`. molpy has two gates in `scripts/check.sh`:

- `molpy` checks formatting, types (ty) and the unit suite, on the molrs wheel
  built from the same checkout.
- `molpy_docs` builds the site with `zensical --strict`.

The pre-push `verify` runs both after `python`. CI runs `python molpy` on
every OS leg of `test / python`, and `docs molpy_docs` in `test / docs`.
**Never `git commit --no-verify` or `git push --no-verify`, and never merge a
red pull request.** Releases are frozen until the molpy-unification API fold
([release process](docs/developer/release-process.md)).

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to
follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molrs/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molrs/issues)
- **Documentation:** [docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/)
