# Adapter

Bidirectional sync between MolPy objects and external library representations.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Adapter[I, E]` | ABC for internal ↔ external sync | Custom adapter implementations |
| `RDKitAdapter` | Sync `Atomistic` ↔ RDKit `Mol` | 3D generation, substructure search |
| `OpenBabelAdapter` | Sync `Atomistic` ↔ OpenBabel `OBMol` | Format conversion |

## Canonical example

```python
from molpy.adapter import RDKitAdapter
from rdkit.Chem import AllChem

adapter = RDKitAdapter(internal=mol)
rd_mol = adapter.get_external()
AllChem.EmbedMolecule(rd_mol)
adapter.set_external(rd_mol)
adapter.sync_to_internal()
optimized = adapter.get_internal()
```

## Key behavior

- `get_external()` auto-syncs internal → external if needed
- `get_internal()` auto-syncs external → internal if needed
- RDKit and OpenBabel are optional dependencies; adapters fail gracefully if not installed

## Related

- [Concepts: Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md)

---

## Full API

### Base

::: molpy.adapter.base

### RDKit

::: molpy.adapter.rdkit

### OpenBabel

::: molpy.adapter.openbabel
