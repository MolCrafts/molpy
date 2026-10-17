# Adapter

In-memory conversion between MolPy objects and an external library's
representation — the in-memory half of MolPy's two bridging patterns. The other
half is the [wrapper](wrapper.md), which shells out to a binary instead.

An adapter is a pair of pure functions, `to_<lib>` / `from_<lib>`: each builds
a new object and leaves its input untouched. MolPy keeps **one worked example**:
`to_rdkit` / `from_rdkit`. Packing is **not** an adapter — use
[molpack](https://docs.molcrafts.org/molpack/) (`pip install molcrafts-molpack`).
An example is not a dependency — RDKit is an optional extra
(`pip install "molcrafts-molpy[rdkit]"`), importing molpy never requires it,
and no molpy code path routes through it.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `to_rdkit(atomistic)` | New sanitized RDKit `Mol` from an `Atomistic` | Handing a structure to RDKit's *own* algorithms |
| `from_rdkit(mol, conf_id=-1)` | New `Atomistic` from an RDKit `Mol` | Bringing RDKit's result back |
| `generate_3d(atomistic)` | RDKit ETKDGv3 embedding (+ UFF), as a round trip | Comparing against RDKit's embedder |
| `PROP_PREFIX` | `"molpy:"`, the namespace of carried RDKit properties | Reading carried components on the RDKit side |

## Canonical example

The examples below share this setup:

```python
import molpy as mp

mol = mp.io.smiles.SmilesIr.parse("CCO").to_atomistic()
```

```python
# docs: skip — RDKit optional adapter example; not unit-tested
import molpy as mp
from molpy.adapter import from_rdkit, to_rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

mol = mp.io.smiles.SmilesIr.parse("CCO").to_atomistic()

rd_mol = Chem.AddHs(to_rdkit(mol))
AllChem.EmbedMolecule(rd_mol)  # RDKit's algorithm, on RDKit's object
AllChem.MMFFOptimizeMolecule(rd_mol)

optimized = from_rdkit(rd_mol)  # a new molpy Atomistic; `mol` is unchanged
```

## What crosses the boundary

| | To RDKit | Back to MolPy |
|--|----------|---------------|
| Element, formal charge, bond type, coordinates | RDKit's own fields (one conformer) | RDKit's values win |
| Bracket hydrogen count (`h_count`, SMILES `[nH]`) | Sets RDKit's H count on that atom | Carried back unchanged |
| Every other atom / bond component (`name`, `res_name`, `res_id`, `type`, `charge`, `mass`, `bond_number`, a bond's `type`, …) | RDKit property `molpy:<key>` | Restored on the atoms and bonds RDKit kept |
| Scalar `Atomistic.props` | Mol property `molpy:<key>` | Restored |

What does **not** survive the round trip:

- **Atoms or bonds RDKit removes** (`RemoveHs`, fragment stripping) are absent
  from the result. Atoms or bonds RDKit creates (`AddHs`) carry RDKit-owned
  fields only.
- **Angles, dihedrals, impropers and ports**: only atoms and bonds cross.
  Regenerate the rest with `generate_topology`.
- **Stereo** is not translated. molpy's `stereo` component does not become RDKit
  chiral tags or bond stereo, and RDKit's chiral tags, bond stereo and CIP
  labels do not come back. The `stereo` component itself rides along untouched,
  so it is stale if RDKit changed that atom's neighbours.
- **Recomputed facts**: when RDKit changes a bond's type, the carried
  `bond_number` is replaced by the number the new type implies (none for
  aromatic). Nothing else is recomputed — an atom whose element RDKit changes
  keeps its old `mass` and `type`.
- **Extra conformers**: `from_rdkit` reads one (`conf_id`).
- A graph prop that is not a bool, int, float or str makes `to_rdkit` raise
  `TypeError` rather than drop it silently.

## Do not use it for what MolPy already does

The adapter exists to reach algorithms MolPy does not implement. For anything
below, the native path is the supported one and needs no third-party install:

| Task | Native |
|------|--------|
| 3D embedding | [`Conformer`](conformer.md) — ETKDGv3 → torsion refinement → MMFF94 cleanup |
| Hydrogens / aromaticity / stereo | `mp.perceive.add_hydrogens(...)` / `mp.perceive.assign_aromaticity(...)` |
| SMILES / SMARTS | `mp.io.smiles.SmilesIr.parse(...).to_atomistic()`, `mp.perceive.SmartsPattern` — see [Notation](notation.md) |
| Ring queries | `mp.perceive.perceive_rings(mol)` → `RingSet` |
| GAFF types | [AmberTools wrapper](wrapper.md) — antechamber delegation |

```python
mol_3d, report = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(mol)
```

## Key behavior

- every function returns a new object and never mutates its input
- `import molpy.adapter` works without RDKit; reading `to_rdkit`,
  `from_rdkit` or `generate_3d` then raises `ImportError` naming the extra to
  install
- an adapter does **in-memory conversion only** — executing an external binary
  belongs in a [wrapper](wrapper.md)

## Related

- [Concepts: Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md)
- [Conformer](conformer.md) — the native embedder

---

## Full API

::: molpy.adapter.to_rdkit

::: molpy.adapter.from_rdkit

::: molpy.adapter.generate_3d
