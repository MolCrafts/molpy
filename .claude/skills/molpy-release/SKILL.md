---
name: molpy-release
description: Pre-merge / release gate. Runs all blocking checks in parallel and emits a GO / NO-GO verdict. Use before merging to master or cutting a release tag.
argument-hint: "[branch or tag description, e.g. 'v0.4.0']"
user-invocable: true
---

# MolPy Release Gate

Read CLAUDE.md for project conventions.

Release label (if provided): $ARGUMENTS

## Procedure

### Step 1 — Lint & pre-commit (inline, run first)

```bash
pre-commit run --all-files 2>&1
```

If pre-commit is not installed: `pip install pre-commit && pre-commit install`.

Capture exit code. Any failure is **BLOCKING**.

### Step 2 — Parallel checks (invoke all at once)

Invoke these four agents **in a single parallel message**:

1. **Test suite** → invoke `/molpy-test` (no arguments — full project)
   Checks: all tests pass, coverage ≥80% overall, ≥90% for `core/`.

2. **Docstring completeness** → invoke `/molpy-api-doc` on `src/molpy/`
   Checks: all public symbols have Google-style docstrings + type hints + units.

3. **Docs currency** → invoke `/molpy-documenter` on `docs/`
   Checks: `docs/` pages reflect current public API; no stale parameter names or removed methods referenced.

4. **Release readiness** → invoke `molpy-release-checker` agent
   Checks: CHANGELOG entry, version bump, breaking-change notes, no debug residue, `__init__.py` exports.

### Step 3 — Aggregate verdict

Collect all findings. Classify each as BLOCKING or ADVISORY:

| Category | BLOCKING if... |
|---|---|
| Tests | any test fails OR coverage below threshold |
| Docstrings | any public API missing docstring or type hint |
| Docs | any stale API reference in `docs/` |
| Release | BLOCKING finding from release-checker |
| Pre-commit | exit code ≠ 0 |

Render the verdict table:

```
═══════════════════════════════════════════════
RELEASE GATE: <label or branch>
═══════════════════════════════════════════════

VERDICT: GO ✅  /  NO-GO ❌

BLOCKING ISSUES  (must fix before merge)
  ❌ [category] file:line — description
  ...

ADVISORY  (fix soon, does not block)
  ⚠️  [category] file:line — description
  ...

SUMMARY
  Tests:       ✅/❌  XX% coverage
  Docstrings:  ✅/❌  N missing
  Docs:        ✅/❌  N stale refs
  Release:     ✅/❌  N BLOCKING, N HIGH
  Pre-commit:  ✅/❌
═══════════════════════════════════════════════
```

**GO** requires: zero BLOCKING issues across all five categories.

## Output

Single verdict table as above. One-line footer: `Release gate complete — GO ✅` or `Release gate complete — NO-GO ❌ (N blocking issues)`.
