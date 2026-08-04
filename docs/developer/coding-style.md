# Coding Style

MolPy values explicitness over cleverness. Code should be readable without external context, testable in isolation, and safe from hidden side effects.


## Mutation: core vs helpers

Two layers, two rules — do not mix them up.

### Core data model (`Atomistic`, `Frame`, …) — mutate in place

`def_atom`, `def_bond`, `get_topo`, `move`, `rotate`, `merge`, and friends
modify the receiver and return `self` (or the created entity) for chaining.
`.copy()` is the **explicit** opt-in for an independent deep copy.

```python
import molpy as mp

mol = mp.io.read_smiles("CCO")
mol.get_topo(gen_angle=True, gen_dihe=True)   # writes angles/dihedrals on mol
work = mol.copy().get_topo(gen_angle=True)    # independent graph + topology
```

### Higher-level helpers (`builder`, `typifier`, `op`, …) — do not surprise the caller

Workflow helpers must not mutate a caller-owned structure unexpectedly. Copy
first, or build and return a new structure.

```python
# wrong: helper silently mutates the caller's mol
def add_hydrogens(mol):
    mol.def_atom(...)  # caller did not ask for this

# right: return a new object
def add_hydrogens(mol):
    new_mol = mol.copy()
    # ... populate new_mol
    return new_mol
```

`typify()` returns a new `Atomistic`. Follow the helper rule in `builder` /
`typifier` / similar packages; follow the core rule inside `molpy.core`.


## Functions and files

Keep functions under 50 lines and focused on one task. Keep files under 800 lines. If a module grows beyond that, extract cohesive groups into new files.


## Code identifiers

Use `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants. Canonical data-field names such as `element`, `charge`, and `mol_id` are defined in [Naming Conventions](../tutorials/naming-conventions.md); do not introduce local variants in developer docs or extension examples. Use MolPy-specific terms precisely: `topology` means the bond graph (not "connections"), `atom type` means the force field identifier (not "kind"), `struct` refers to the MolPy `Struct` base class (not a generic "structure").


## Type hints

Public APIs must have type hints. Private helpers should have them when the signature is not obvious. Use `from __future__ import annotations` for forward references.


## Imports

Order: standard library, then third-party packages, then `molpy` imports. Separate groups with a blank line. Use absolute imports within `molpy` (`from molpy import Frame`, not relative imports).


## Error handling

Validate inputs at the boundary — where data enters a public function. Raise specific exceptions (`ValueError`, `TypeError`, `FileNotFoundError`) with messages that include the actual value and the expected constraint. Never silently swallow exceptions.


## Docstrings

Use Google-style docstrings for public functions and classes. Include `Args`, `Returns`, and `Raises` sections. For physical quantities, always state units (A, kcal/mol, radians). For array arguments, state the expected shape.


## Formatting

Ruff is the single source of truth for formatting and linting. Run `ruff format src tests` and `ruff check src` before committing. Pre-commit hooks enforce this automatically.


## Ready-to-commit checklist

A change is ready when:

- [ ] Code is readable without extra explanation
- [ ] Functions are under 50 lines
- [ ] Core APIs mutate in place intentionally; helpers do not mutate caller-owned inputs
- [ ] Tests cover the changed behavior
- [ ] Public APIs have type hints and docstrings
- [ ] `ruff format --check src tests` passes
- [ ] `ruff check src` passes
- [ ] `pre-commit run --all-files` passes
