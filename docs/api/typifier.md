# Typifier

Graph typification and force-field parameter assignment.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `OPLSAATypifier` | Full OPLS-AA typing pipeline re-exported from `molrs.typifier` | OPLS-AA all-atom force fields |
| `MMFFTypifier` | Full MMFF94 typing pipeline re-exported from `molrs.typifier` | MMFF all-atom force fields |
| `ClpTypifier` | CL&P force-field loader shell; typification pending molpy-side rewrite | Ionic-liquid force fields |
| `PairTypifier` | Pair (LJ) parameter assignment | Standalone nonbonded typing |
| `.typify(mol)` | Return a new graph of the same concrete type | One-call complete typing |

The typifier contract is `typify(mol: T) -> T`: input and output are molecular
graphs, not frames. Built-in all-atom typifiers accept `molrs.Atomistic` and
return `molrs.Atomistic`; convert with `.to_frame()` only after typification.
When a force field defines bonded terms, `typify()` must label the complete
topology it supports: atoms, bonds, angles, dihedrals, and force-field-specific
impropers.

GAFF is **not** a typifier here: GAFF atom types and AM1-BCC charges are
obtained by delegating to AmberTools (`antechamber` / `prepgen`) through the
[Wrapper](wrapper.md) package, not through a `*Typifier` class.

## Canonical example

```python
import molpy as mp
from molpy.typifier import OPLSAATypifier

typifier = OPLSAATypifier(strict=True)
typed_mol = typifier.typify(mol)  # returns a new Atomistic
frame = typed_mol.to_frame()
```

## Key behavior

- `typify()` returns a **new** graph — the original is not modified
- `OPLSAATypifier(strict=True)` raises on missing OPLS-AA bonded terms
- SMARTS matching is implemented in molrs; MolPy no longer carries an igraph matcher
- Matcher APIs return match bindings, while typifier APIs return typed graphs

## Related

- [Guide: Force Field Typification](../user-guide/06_typifier.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Typifier Re-exports

::: molpy.typifier

### Pair Typifier

::: molpy.typifier.atomistic

### CL&P Typifier

::: molpy.typifier.clp

### MMFF Typifier

::: molpy.typifier.mmff
