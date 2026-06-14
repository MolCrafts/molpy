# Typifier

SMARTS-based atom typing and force field parameter assignment.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `OplsTypifier` | Full OPLS-AA typing pipeline | OPLS force fields |
| `GaffTypifier` | Full GAFF typing pipeline | GAFF / GAFF2 force fields |
| `ClpTypifier` | Full CL&P typing pipeline (OPLS engine + built-in `clp.xml`) | Ionic-liquid force fields |
| `PairTypifier` | Pair (LJ) parameter assignment | Standalone nonbonded typing |
| `LayeredTypingEngine` | Dependency-aware SMARTS matching engine | Custom typing engines |
| `DependencyAnalyzer` | Computes SMARTS pattern dependency levels | Custom typing engines |
| `.typify(struct)` | Assign all types (atom → pair → bond → angle → dihedral) | One-call complete typing |

Each force-field typifier is a single orchestrator class: one `typify()` call
runs atom typing, then pair parameters, then bond/angle/dihedral types derived
from the atom assignments. Individual stages can be disabled with the
`skip_*_typing` constructor flags — there are no separate public
atom-only or bond/angle/dihedral typifier classes.

## Canonical example

```python
import molpy as mp
from molpy.typifier import OplsTypifier

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=True)
typed_mol = typifier.typify(mol)  # returns NEW Atomistic
```

## Key behavior

- `typify()` returns a **new** `Atomistic` — the original is not modified
- `strict_typing=True` raises on untyped atoms; `False` silently skips them
- Atom typing uses SMARTS pattern matching with priority/override resolution
- Bonded types are derived from atom type assignments (CT-OH → bond type CT-OH)

## Related

- [Guide: Force Field Typification](../user-guide/06_typifier.ipynb)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Force-Field Typifiers (OPLS, base orchestrator, pair)

::: molpy.typifier.atomistic

### GAFF Typifier

::: molpy.typifier.gaff

### CL&P Typifier

::: molpy.typifier.clp

### Layered Typing Engine

::: molpy.typifier.layered_engine

### Dependency Analyzer

::: molpy.typifier.dependency_analyzer

### Adapter

::: molpy.typifier.adapter

### Graph

::: molpy.typifier.graph
