---
name: molpy-api-doc
description: Audit and write agent-friendly API documentation — docstrings, type hints, parameter/return specs. Use after implementing a feature or during code review to ensure APIs are machine-readable.
argument-hint: "[path or module]"
user-invocable: true
---

Audit and write API documentation for: $ARGUMENTS

If no path given, audit all files modified in `git diff --name-only HEAD`.

Target audience: **AI agents and LLM tools** consuming MolPy via tool-calling or code generation, plus human developers reading inline docs.

---

## What "agent-friendly" means

An API is agent-friendly when an LLM can call it correctly from the docstring alone:
- Parameter types are explicit (no "see source")
- Units are in the docstring, not scattered in comments
- Return type and shape are unambiguous
- Side effects and mutation are declared
- Raises section lists every exception path

---

## Audit Checklist

For every **public** function, method, and class in scope:

**Docstring completeness**
- [ ] One-line summary (imperative mood: "Compute...", "Return...", "Parse...")
- [ ] Args section: every parameter has `name (type): description [unit if physical]`
- [ ] Returns section: type + description + shape for arrays
- [ ] Raises section: every `raise` in the body is listed
- [ ] Example section for non-trivial APIs (≥ 1 usage, expected output as comment)

**Type hints**
- [ ] All public function signatures have type annotations (Python 3.12+ style)
- [ ] No bare `Any` without a comment explaining why
- [ ] `Optional[X]` written as `X | None`

**Physical quantities**
- [ ] Every physical parameter documents its unit: `r (float): distance in Å`
- [ ] Return values with units documented: `Returns: float — energy in kcal/mol`
- [ ] Convention documented when ambiguous (e.g., `K` vs `k/2` for harmonic)

**Mutation / side effects**
- [ ] If the function mutates an input, it is declared: `Mutates: mol — adds bonds in-place`
- [ ] If it returns a new object (correct immutable style), say so: `Returns a new Atomistic`

---

## Steps

**Step 1 — Scan scope**

List all public symbols in `$ARGUMENTS` using `mcp__molpy__list_symbols`. Flag each as: ✅ documented, ⚠️ partial, ❌ missing.

**Step 2 — Audit existing docstrings**

For each symbol, check against the checklist above. Note specific gaps (missing unit, no Raises, bare Any).

**Step 3 — Write missing docs**

Use the `molpy-documenter` agent to fill gaps. Format: Google-style docstrings.

Example of a complete docstring:

```python
def compute_rdf(
    frame: Frame,
    r_max: float,
    n_bins: int = 100,
    species: tuple[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function g(r).

    Args:
        frame: Snapshot containing atomic positions and box.
        r_max: Maximum distance in Å.
        n_bins: Number of histogram bins.
        species: Pair of element symbols to filter (e.g. ``("O", "H")``).
            If None, all pairs are included.

    Returns:
        Tuple of (r, g_r):
            - r: ndarray of shape (n_bins,) — bin centers in Å
            - g_r: ndarray of shape (n_bins,) — g(r) values (dimensionless)

    Raises:
        ValueError: If r_max exceeds half the minimum box dimension.

    Example:
        >>> r, g = compute_rdf(frame, r_max=10.0)
        >>> g[0]  # -> ~0.0 (excluded volume near origin)
    """
```

**Step 4 — Report**

```
API DOC AUDIT: <path>

Summary: N public symbols
  ✅ Fully documented: N
  ⚠️  Partial: N  (list gaps)
  ❌ Missing: N   (list symbols)

Physical units missing: [list]
Type hints missing: [list]
Mutation undeclared: [list]

Files written/updated: [list]
```
