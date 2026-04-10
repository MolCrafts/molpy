# Coding Style

MolPy values explicitness over cleverness. Code should be readable without external context, testable in isolation, and safe from hidden side effects.


## Immutability

This is the most important rule. Transformations return new objects. Inputs are never modified in place.

```python
# wrong: mutates input
def add_hydrogens(mol):
    mol.add_atoms(...)

# right: returns new object
def add_hydrogens(mol):
    new_mol = mol.copy()
    # ... populate new_mol
    return new_mol
```

The `typify()` method returns a new `Atomistic`. The `Struct.copy()` method deep-copies entities and links. Follow these patterns in new code.


## Functions and files

Keep functions under 50 lines and focused on one task. Keep files under 800 lines. If a module grows beyond that, extract cohesive groups into new files.


## Naming

Use `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants. Use MolPy-specific terms precisely: `topology` means the bond graph (not "connections"), `atom type` means the force field identifier (not "kind"), `struct` refers to the MolPy `Struct` base class (not a generic "structure").


## Type hints

Public APIs must have type hints. Private helpers should have them when the signature is not obvious. Use `from __future__ import annotations` for forward references.


## Imports

Order: standard library, then third-party packages, then `molpy` imports. Separate groups with a blank line. Use absolute imports within `molpy` (`from molpy.core.frame import Frame`, not relative imports).


## Error handling

Validate inputs at the boundary — where data enters a public function. Raise specific exceptions (`ValueError`, `TypeError`, `FileNotFoundError`) with messages that include the actual value and the expected constraint. Never silently swallow exceptions.


## Docstrings

Use Google-style docstrings for public functions and classes. Include `Args`, `Returns`, and `Raises` sections. For physical quantities, always state units (A, kcal/mol, radians). For array arguments, state the expected shape.


## Formatting

Black is the single source of truth for formatting. No configuration overrides. Run `black src tests` before committing. Pre-commit hooks enforce this automatically.


## Ready-to-commit checklist

A change is ready when:

- [ ] Code is readable without extra explanation
- [ ] Functions are under 50 lines
- [ ] No mutation of input objects
- [ ] Tests cover the changed behavior
- [ ] Public APIs have type hints and docstrings
- [ ] `black --check src tests` passes
- [ ] `pre-commit run --all-files` passes
