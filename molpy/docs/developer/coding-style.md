# Coding Style

MolPy values explicitness over cleverness. Code should be readable without external context, testable in isolation, and safe from hidden side effects.


## Mutation: core vs helpers

Two layers, two rules — do not mix them up.

### Core data model (`Atomistic`, `Frame`, …) — mutate in place

`add_atom`, `add_bond`, `generate_topology`, `translate`, `rotate` and
`merge` modify the receiver. Atom and relation additions return live views;
`generate_topology` returns counts, `merge` returns the added atom views,
and rigid transforms return `None`. Use `copy.copy(graph)` for an independent
graph, or `frame.copy()` for an independent frame.

```python
from copy import copy
import molpy as mp

mol = mp.io.smiles.read_smiles("CCO").to_atomistic()
mol.generate_topology(gen_angle=True, gen_dihedral=True)  # writes angles/dihedrals on mol
work = copy(mol)                                         # independent graph
work.generate_topology(gen_angle=True, clear_existing=True)  # mol is untouched
```

### Higher-level helpers (`builder`, `typifier`, `op`, …) — do not surprise the caller

Workflow helpers must not mutate a caller-owned structure unexpectedly. Copy
first, or build and return a new structure.

```python
from copy import copy
# wrong: helper silently mutates the caller's mol
def add_hydrogens(mol):
    mol.add_atom(...)  # caller did not ask for this

# right: return a new object
def add_hydrogens(mol):
    new_mol = copy(mol)
    # ... populate new_mol
    return new_mol
```

`typify()` returns a new `Atomistic`. Follow the helper rule in `builder` /
`ff.typifier` / similar packages; follow the core rule for the data model.


## Functions and files

Keep functions under 50 lines and focused on one task. Keep files under 800 lines. If a module grows beyond that, extract cohesive groups into new files.


## Code identifiers

Use `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants. Canonical data-field names such as `element`, `charge`, and `mol_id` are defined in [Naming Conventions](../tutorials/naming-conventions.md); do not introduce local variants in developer docs or extension examples. Use MolPy-specific terms precisely: `topology` means the bond graph (not "connections"), `atom type` means the force field identifier (not "kind").

molpy follows molrs's naming rules:

- **No module name starts with `_`.** What is public is what an `__all__`
  lists; a module no `__all__` lists (`molpy.engine.lammps`) is an
  implementation module, reached through its package's names
  (`mp.engine.LammpsEngine`).

- **A name states its job.** No module is named after a container word —
  `data`, `base`, `types`, `common`, `utils`, `helpers`, `env` — public or
  private: the bundled files are `molpy.resources`, the engine base class lives
  in `engine/engine.py`, a tool's environment in `tool/environment.py`.
  Classes and functions follow (`EnvironmentSpec`, `tool_settings`,
  `TypeAssignment`, `source_forcefield`).
- **Acronyms are cased as words**: `LammpsEngine`, `OpenmmEngine`,
  `SmilesIr`, `CgBond`, `Rdf`, `Msd`,
  `Lbfgs`, `PairLjCut`. numpy's `DType` and `HBond` (H is the element) keep
  their spelling.
- **Counts are `n_*`** (numpy / freud): `n_atoms`, `n_pairs`, `n_rings()`,
  `n_query_atoms` — never `num_*`.
- **One door per file format**, named after the format:
  `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]`, `_str` / `_bytes` /
  `_trajectory` for text, bytes and multi-frame input; no door picks a format
  from a file extension.


## Type hints

Public APIs must have type hints. Private helpers should have them when the signature is not obvious. Use `from __future__ import annotations` for forward references.


## Imports

Order: standard library, then third-party packages, then `molpy` imports. Separate groups with a blank line. Use absolute imports within `molpy` (`from molpy.core import Frame`, not relative imports).


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
