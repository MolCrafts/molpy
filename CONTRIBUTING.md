# Contributing to MolPy

Thank you for your interest in contributing to MolPy!

The full contributor handbook lives in the documentation — this file is just the
entry point, so the rules have a single home and cannot drift:

- **[Development Setup](https://molpy.molcrafts.org/developer/development-setup/)** — clone, editable install, pre-commit hooks, running tests
- **[Contributing Workflow](https://molpy.molcrafts.org/developer/contributing/)** — branches, conventional commits, the PR checklist
- **[Coding Style](https://molpy.molcrafts.org/developer/coding-style/)** — ruff formatting, type hints, Google-style docstrings, the mutation contract
- **[Testing](https://molpy.molcrafts.org/developer/testing/)** — pytest conventions, markers, coverage expectations
- **[Architecture Overview](https://molpy.molcrafts.org/developer/architecture-overview/)** — module map and extension points, read before larger changes

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/molpy.git
cd molpy
pip install -e ".[dev]"
pre-commit install
pytest tests/ -m "not external"
```

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to
follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/MolCrafts/molpy/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/MolCrafts/molpy/issues)
- **Documentation:** [molpy.molcrafts.org](https://molpy.molcrafts.org/)
