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
scripts/check.sh python docs
```

## Hooks and CI

Every gate is one function of `scripts/check.sh`, called by the prek hooks
and by CI alike; `scripts/check.sh verify` (the pre-push hook) runs them all
once. The Python ones are `python` (build one molpy wheel from the checkout,
check format and types, run the whole Python suite on it), `types`
(`molpy/python/molpy/_core.pyi` against the compiled module) and `docs` (this
manual, `--strict`). CI runs `python` on every OS leg of `test / python` and
`docs` in `test / docs`.
**Never `git commit --no-verify` or `git push --no-verify`, and never merge a
red pull request.** Releases are frozen until the molpy-unification API fold
([release process](docs/developer/release-process.md)).

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to
follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation:** [docs.molcrafts.org/molpy](https://docs.molcrafts.org/molpy/)
