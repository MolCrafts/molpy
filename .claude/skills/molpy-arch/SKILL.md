---
name: molpy-arch
description: Validate code against MolPy's 9-layer architecture and module dependency rules. Use before code review, after refactoring imports, or when adding cross-package dependencies.
argument-hint: "[path or module]"
user-invocable: true
---

Validate architecture for: $ARGUMENTS

If no path given, check all files modified in `git diff --name-only HEAD`.

**Dependency Rules**

```
ALLOWED (→ = can import from):
  core/              ← everything inside molpy
  parser/            ← io, typifier, builder, reacter
  io/                ← builder, pack, engine
  typifier/          ← builder
  compute/           ← (application code only)
  builder/           ← pack
  reacter/           ← builder
  pack/              ← (application code only)
  engine/            ← (application code only)
  wrapper/adapter/   ← builder, pack, engine
```

Lower layers CANNOT import from higher layers. `core/` imports NOTHING from molpy.

**Checks**

1. **Import direction**: Scan `.py` files for import violations against the rules above.
2. **Circular imports**: Detect A→B→A patterns.
3. **Immutability**: Flag in-place mutation of input objects (`.append()`, `obj[key] = val` on inputs).
4. **Identity hashing**: `Entity` and `Link` must use `id(self)`, not value-based `__eq__`.
5. **Formatter registry**: ForceField writers must use `_formatters` dict, not hard-coded serialization.
6. **Optional imports**: External tools (RDKit, OpenBabel) must use adapter pattern with try/except fallback.
7. **Star imports**: Flag `from module import *` (except in `__init__.py`).
8. **Side effects**: Flag code execution at module level (beyond imports and class definitions).

**Output format**:
```
ARCHITECTURE VALIDATION: <path>

✅ Import directions: OK
❌ <violation description> (file:line)
⚠️ <warning description> (file:line)

N ERRORS, M WARNINGS
```
