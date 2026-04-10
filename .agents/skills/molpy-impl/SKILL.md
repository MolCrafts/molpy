---
name: molpy-impl
description: Full implementation workflow from spec to production-ready code. Use when implementing new potentials, I/O formats, parsers, builders, compute operators, or any non-trivial feature.
argument-hint: <feature description or spec path>
user-invocable: true
---

Implement the following feature in MolPy: $ARGUMENTS

**Execution discipline**: Before writing any code, enter **Plan Mode** to lay out the full plan, then create **Tasks** for each phase below. Update task status as work progresses (`in_progress` → `completed`). This enforces a structured, auditable workflow — the agent must not skip phases or jump ahead without completing prior tasks.

**Phase 1 — Literature Review** (for physical models only)
If the feature involves a potential, force field, compute operator, or typifier, invoke `/molpy-litrev` first. Abort if no credible scientific basis is found.

**Phase 2 — Spec**
If `$ARGUMENTS` is a file path, read it. Otherwise invoke `/molpy-spec` to generate a detailed spec.

**Phase 3 — Architecture Design**
Use the `molpy-architect` agent to:
- Validate against the 9-layer dependency graph
- Identify target package and affected modules
- Verify patterns (immutability, formatter registry, adapter pattern)
- Produce a module impact map

**Phase 4 — TDD (RED)**
Use the `molpy-tester` agent to write failing tests:
- Happy path with typical inputs
- Edge cases (empty input, single atom, invalid parameters)
- Immutability test (input objects not mutated)
- Scientific validation (known analytical values, limiting cases)
- I/O round-trip tests (if applicable)
- External tool tests marked `@pytest.mark.external`
- Run `pytest tests/test_<pkg>/ -v` — confirm tests FAIL

**Phase 5 — Implement (GREEN)**
Write code following these rules:
- Python 3.12+ APIs
- Immutable data flow: return new objects, never mutate inputs
- Type hints on all public APIs
- Google-style docstrings with units for physical quantities
- Functions < 50 lines, files < 800 lines
- Scientific reference in module docstring
- Use existing base classes (ForceFieldReader/Writer, BondPotential, etc.)
- Optional imports via adapter pattern with try/except

Run `pytest tests/test_<pkg>/ -v` — confirm tests PASS.

**Phase 6 — Review**
Run `/molpy-review` on all modified files.

**Phase 7 — Documentation**
Use the `molpy-documenter` agent to add Google-style docstrings, unit annotations, Reference sections, and update `docs/` if needed.

**Phase 8 — Final Verification**
Run in parallel: `/molpy-arch`, `/molpy-test`, `/molpy-perf`, `/molpy-docs`.

Report: files created/modified, test results, coverage, literature references, remaining TODOs.
