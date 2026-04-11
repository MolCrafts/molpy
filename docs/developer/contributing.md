# Contributing

This page describes the workflow for contributing code, tests, or documentation to MolPy.

## Before you start

Search [existing issues](https://github.com/MolCrafts/molpy/issues) to avoid duplicating work. Read the [code of conduct](https://github.com/MolCrafts/molpy/blob/master/CODE_OF_CONDUCT.md).


## Workflow

Fork the repository, clone your fork, and create a branch from `master`. Branch names should indicate the type of change: `feature/morse-potential`, `fix/pdb-reader-crash`, `docs/typifier-guide`.

Implement the change with tests. Run local checks before pushing:

```bash
ruff format --check src tests
ruff check src
pytest tests/ -v -m "not external"
```

Open a pull request with a clear summary. The PR description should include what changed, why, and how to verify it.


## Pull request checklist

- [ ] Scope is focused — one logical change per PR
- [ ] New behavior has tests
- [ ] Public API changes have type hints and docstrings
- [ ] Backward-incompatible changes are called out in the description
- [ ] All commits have clear messages (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`)


## What makes a good PR description

A good description answers five questions:

1. **What problem does this solve?** — link to an issue or describe the pain point
2. **What changed?** — summarize the code changes (not a line-by-line diff)
3. **Why this approach?** — explain design choices, especially if alternatives exist
4. **What are the risks?** — note breaking changes, performance implications, or edge cases
5. **How was it tested?** — describe test strategy, include output or screenshots if helpful


## Documentation expectations

When behavior changes, update the relevant docs in `docs/`. The three doc layers have different standards:

- **Concepts** (human-first) — narrative explanation with code after context
- **Guides** (human-first) — end-to-end workflows with runnable code
- **API Reference** (agent-first) — quick-reference tables + mkdocstrings auto-generation

If the change adds a new public symbol, it should appear in the appropriate API page. If it changes user-facing behavior, the relevant guide or concept page should be updated.
