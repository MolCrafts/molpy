# Adapter

Bidirectional sync between MolPy objects and an external library's
representation — the in-memory half of MolPy's two bridging patterns. The other
half is the [wrapper](wrapper.md), which shells out to a binary instead.

MolPy keeps **one worked example of adapter**: `RDKitAdapter` here. Packing is
**not** an adapter — use [molpack](https://docs.molcrafts.org/molpack/)
(`pip install molcrafts-molpack`). An example is not a dependency — RDKit is an
optional extra (`pip install "molcrafts-molpy[rdkit]"`), importing molpy never
requires it, and no molpy code path routes through it.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Adapter[I, E]` | ABC for internal ↔ external sync | Your own integration |
| `RDKitAdapter` | Sync `Atomistic` ↔ RDKit `Mol` | Reaching RDKit's *own* algorithms |

## Canonical example

The examples below share this setup:

```python
import molpy as mp

mol = mp.io.read_smiles("CCO")
```

```python
# docs: skip — RDKit optional adapter example; not unit-tested
import molpy as mp
from molpy.adapter import RDKitAdapter
from rdkit.Chem import AllChem

mol = mp.io.read_smiles("CCO")

adapter = RDKitAdapter(internal=mol)
rd_mol = adapter.get_external()

AllChem.EmbedMolecule(rd_mol) # RDKit's algorithm, on RDKit's object
AllChem.MMFFOptimizeMolecule(rd_mol)

adapter.set_external(rd_mol)
adapter.sync_to_internal()
optimized = adapter.get_internal() # back to a molpy Atomistic
```

## Do not use it for what MolPy already does

The adapter exists to reach algorithms MolPy does not implement. For anything
below, the native path is the supported one and needs no third-party install:

| Task | Native |
|------|--------|
| 3D embedding | [`Conformer`](conformer.md) — ETKDGv3 → torsion refinement → MMFF94 cleanup |
| Hydrogens / aromaticity / stereo | `mp.Perceive().find_hydrogens(...)` / `.find_aromaticity(...)` |
| SMILES / SMARTS | `mp.io.read_smiles(...)`, `mp.SmilesIR`, `mp.SmartsPattern` — see [Parser](parser.md) |
| Ring queries | `mp.RingInfo(mol)` |
| GAFF types | [AmberTools wrapper](wrapper.md) — antechamber delegation |

```python
mol_3d, report = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(mol)
```

## Key behavior

- `get_external()` auto-syncs internal → external if needed
- `get_internal()` auto-syncs external → internal if needed
- RDKit is optional; `molpy.adapter.RDKitAdapter` is `None` when it is not installed
- an adapter does **data synchronisation only** — executing an external binary
 belongs in a [wrapper](wrapper.md)

## Related

- [Concepts: Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md)
- [Conformer](conformer.md) — the native embedder

---

## Full API

### Base

::: molpy.adapter.base

### RDKit

::: molpy.adapter.rdkit
