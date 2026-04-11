# Typifier

SMARTS-based atom typing and force field parameter assignment.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `OplsAtomisticTypifier` | Full OPLS-AA typing pipeline | OPLS force fields |
| `GaffTypifier` | Full GAFF typing pipeline | GAFF / GAFF2 force fields |
| `OplsAtomTypifier` | OPLS atom-only typing | When you only need atom types |
| `GaffAtomTypifier` | GAFF atom-only typing | When you only need atom types |
| `BondTypifier` | Bond type assignment from atom types | Standalone bonded typing |
| `AngleTypifier` | Angle type assignment from atom types | Standalone bonded typing |
| `DihedralTypifier` | Dihedral type assignment from atom types | Standalone bonded typing |
| `PairTypifier` | Pair (LJ) parameter assignment | Standalone nonbonded typing |
| `TypifierBase` | ABC for all typifiers | Custom typifier implementations |
| `.typify(struct)` | Assign all types (atom → pair → bond → angle → dihedral) | One-call complete typing |

## Canonical example

```python
import molpy as mp
from molpy.typifier import OplsAtomisticTypifier

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsAtomisticTypifier(ff, strict_typing=True)
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

### Base

::: molpy.typifier.base

### OPLS Typifier

::: molpy.typifier.opls

### GAFF Typifier

::: molpy.typifier.gaff

### Bond Typifier

::: molpy.typifier.bond

### Angle Typifier

::: molpy.typifier.angle

### Dihedral Typifier

::: molpy.typifier.dihedral

### Pair Typifier

::: molpy.typifier.pair

### Adapter

::: molpy.typifier.adapter

### Dependency Analyzer

::: molpy.typifier.dependency_analyzer

### Graph

::: molpy.typifier.graph
