# Extending Typifiers

Typifiers operate on molecular graphs. The core contract is:

```python
def typify(mol: T) -> T: ...
```

`T` must be a graph type, normally `molrs.Atomistic` for all-atom force fields
or a coarse-grained graph type for united-atom and coarse-grained force fields.
The returned object must be the same concrete graph type as the input. A
typifier must not return a `Frame`; call `.to_frame()` after typification when a
writer or potential compiler needs columnar data.

## Required API Shape

Use typifier verbs consistently:

- `typify(mol)` runs the complete pipeline for that force field.
- Stepwise helpers, when they are public, must be named `typify_*`:
  `typify_atoms`, `typify_bond`, `typify_angle`, `typify_dihedral`,
  `typify_improper`.
- Do not expose `assign_*`, `classify_*`, `typify_frame`, or `typify_full`.
- Do not add `from_forcefield(ff)`. A typifier constructor should load or build
  its own parameter tables and keep the force-field object it needs internally.

The complete pipeline must cover every topology class the force field supports.
For OPLS-AA that currently means atoms, bonds, angles, and dihedrals. For MMFF it
also includes MMFF out-of-plane impropers and stretch-bend parameters. If a
force field has no improper table, do not synthesize one just to satisfy a
generic abstraction.

## Matcher Boundary

The matcher is an implementation detail of a typifier, not the typifier itself.
Use molrs SMARTS matching directly:

```python
pattern = molrs.SmartsPattern("[C:1][O:2]")
matches = pattern.find_matches(mol)
```

Matches are bindings: atom ids plus optional mapping labels. They are not
graphs and not frames. Do not reintroduce the old Python igraph matcher or
MolPy-side layered matcher classes; OPLS-AA and MMFF matching now live in molrs.

## CL&P as a MolPy-Side Extension

CL&P stays in MolPy until its coverage and tests are complete. Treat it as an
overlay typifier, not as a subclass of `OPLSAATypifier`.

The current shell only loads the force field:

```python
from molpy.typifier import ClpTypifier

typifier = ClpTypifier()
ff = typifier.ff
```

The eventual implementation should follow this shape:

```python
from __future__ import annotations

import molpy as mp


class ClpTypifier:
    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self.ff = self.load_forcefield()
        self._tables = self._build_tables(self.ff)

    def typify(self, mol: molrs.Atomistic) -> molrs.Atomistic:
        if not isinstance(mol, molrs.Atomistic):
            raise TypeError("CL&P typifier expects molrs.Atomistic")

        typed = self.typify_atoms(mol)
        typed = self.typify_bonds(typed)
        typed = self.typify_angles(typed)
        typed = self.typify_dihedrals(typed)
        return typed

    def typify_atoms(self, mol: molrs.Atomistic) -> molrs.Atomistic:
        # Use molrs.SmartsPattern and CL&P-specific priority rules.
        raise NotImplementedError

    def typify_bonds(self, mol: molrs.Atomistic) -> molrs.Atomistic:
        raise NotImplementedError

    def typify_angles(self, mol: molrs.Atomistic) -> molrs.Atomistic:
        raise NotImplementedError

    def typify_dihedrals(self, mol: molrs.Atomistic) -> molrs.Atomistic:
        raise NotImplementedError
```

Keep the overlay rules explicit:

- Load canonical OPLS-AA data first, then CL&P XML at a higher layer.
- CL&P-specific SMARTS and priority rules belong in CL&P metadata, not in
  generic OPLS-AA code.
- Use molrs match bindings to choose atoms; use graph setters to write `type`,
  `class`, `charge`, and bonded parameters.
- Return a new `Atomistic` at every step. Do not mutate caller-owned input.

## Tests

New typifiers need focused tests at three levels:

- Atom coverage: expected `type`, `class`, and charge on representative real
  molecules.
- Topology coverage: expected bond, angle, dihedral, and improper counts plus
  parameter columns after `typify()`.
- Build parity: `typifier.build(mol)` must consume the same topology as
  `typifier.typify(mol).to_frame()`.

For CL&P, start with one imidazolium cation and one anion fixture, then add the
full ionic-liquid set only after the small fixtures are stable.
