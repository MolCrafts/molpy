[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/getting-started/core-concepts.ipynb)

# Core Concepts

MolPy's design centers on a clear separation of concerns that keeps editing intuitive, data fast, and workflows reproducible.

Design philosophy (why):
- Identity vs data: entities (atoms, links) are unique identities; bulk data lives in columnar arrays.
- Graph → Arrays pipeline: build and edit as a graph (`Atomistic`), compute and export from arrays (`Frame`).
- Derived topology: angles/dihedrals are derived from bonds to avoid stale caches and to keep logic explicit.
- Parameters are separate: force-field typing is independent of structure to preserve portability and clarity.

Core building blocks (what):
1. Entity & Link (graph model): `Atom`, `Bond`, `Angle`, `Dihedral`.
2. Block & Frame (columnar data): tables of NumPy arrays keyed by block name (`atoms`, `bonds`, ...).
3. Topology (connectivity engine): detects higher-order interactions and patterns.
4. ForceField & Typifier (parameters): assigns types and physical parameters.

Typical flow (how): Atomistic (edit) → Topology (derive) → Frame (arrays) → I/O → Simulation → Analysis.

## 1. Entity & Link: identity-first graph modeling

MolPy uses a graph model for editing because molecular construction is fundamentally about **connectivity** and **identity**.

Two distinct layers matter:
- **Identity**: "this specific atom instance" (unique object)
- **Data**: attributes attached to it (`symbol`, `x/y/z`, `charge`, `type`, ...)

This is why two atoms with identical data are still different entities: they can participate in different bonds, reactions, selections, and edits.

```python
import molpy as mp

# Two atoms can have identical data but different identity
a1 = mp.Atom(symbol="H", xyz=[0, 0, 0])
a2 = mp.Atom(symbol="H", xyz=[0, 0, 0])

print("Same symbol?", a1["symbol"] == a2["symbol"])
print("Same identity?", a1 is a2)

# Identity makes entities safe as keys in graphs / dicts
labels = {a1: "first", a2: "second"}
print(labels[a1], labels[a2])
```

### Links: connectivity without ownership

A `Link` connects entities but does not "own" them. This matters because you can:
- remove a bond without deleting atoms
- replace a bond graph and re-derive topology
- keep atom identities stable while editing connectivity

Common link types: `Bond` (2), `Angle` (3), `Dihedral` (4).

```python
c1 = mp.Atom(symbol="C")
c2 = mp.Atom(symbol="C")

bond = mp.Bond(c1, c2, order=1.0)
print("Bond endpoints:", bond.endpoints)

# The link is a separate identity too
print("Same bond instance?", bond is mp.Bond(c1, c2, order=1.0))
```

## 2. Block & Frame: array-first data for speed

`Atomistic` is ideal for editing, but simulation I/O and numeric work are best done with arrays.

A **Block** is a columnar table (dict of NumPy arrays) for one "kind" of thing (atoms, bonds, ...).
A **Frame** bundles multiple blocks plus metadata (box, timestep, force field info).

This is the core split:
- `Atomistic`: edit/compose on a graph
- `Frame`: export/compute on arrays

```python
from molpy.core.frame import Block
import numpy as np

atoms = Block(
    {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 0.0, 0.0]),
        "z": np.array([0.0, 0.0, 0.0]),
        "symbol": np.array(["O", "H", "H"], dtype=object),
    }
 )

print("nrows:", atoms.nrows)
print("x:", atoms["x"])
```

### Frame: a named collection of blocks + metadata

A `Frame` is what writers/readers operate on. It's the stable representation for file formats, trajectories, and engine handoff.

A `Frame` contains:
- blocks like `atoms`, `bonds`, `angles`
- metadata like `box`, `timestep`, provenance, etc.

```python
frame = mp.Frame()
frame["atoms"] = atoms
frame.box = mp.Box.cubic(10.0)

print("Has atoms block?", "atoms" in frame)
print("Blocks:", list(frame.to_dict()["blocks"].keys()))
print("Box:", frame.box)
```

## 3. Topology: derived interactions from bonds

Topology answers questions like:
- what is connected to what?
- what angles/dihedrals exist given the bond graph?
- what paths/rings/components exist?

Important: MolPy treats many interactions as **derived** rather than manually stored. In practice:
1) you define atoms + bonds
2) topology derives higher-order interactions (angles/dihedrals)
3) when bonds change, you derive again

This is why the Quickstart doesn't teach "loop over angles and manually manage them": that's a maintenance trap.

```python
from molpy.core.topology import Topology

# Simple chain: 0-1-2-3
topo = Topology()
topo.add_atoms(4)
topo.add_bonds([(0, 1), (1, 2), (2, 3)])

print("atoms:", topo.n_atoms, "bonds:", topo.n_bonds)
print("angles:", topo.n_angles, "dihedrals:", topo.n_dihedrals)
print("connected?", topo.is_connected())
print("shortest path 0->3:", topo.get_shortest_paths(0, 3)[0])
```

## 4. ForceField & Typifier: parameters are not structure

A force field is not "the molecule". It's a parameter catalog that can be applied to many molecules.

MolPy splits this into:
- `ForceField`: stores styles and type tables (atoms, bonds, angles, dihedrals, pairs)
- `Typifier`: assigns types/parameters onto a structure based on rules and context

This separation is what makes workflows reproducible and engine-agnostic. You can rebuild a structure and re-apply typing deterministically.

```python
import molpy as mp

ff = mp.AtomisticForcefield(name="ToyFF", units="real")
astyle = ff.def_atomstyle("full")
ct = astyle.def_type("CT", type_="CT", class_="CT", mass=12.01)
hc = astyle.def_type("HC", type_="HC", class_="HC", mass=1.008)

bstyle = ff.def_bondstyle("harmonic")
bstyle.def_type(ct, hc, k=340.0, r0=1.09)

print("AtomTypes:", [t.name for t in ff.get_types(mp.AtomType)])
print("BondTypes:", [t.name for t in ff.get_types(mp.BondType)])
```

## Summary: choosing the right layer

- Use `Atomistic` when you need to **build and edit** (graph CRUD).
- Use topology helpers (`get_topo` / `Topology`) when you need **derived interactions** and graph queries.
- Use `Frame` when you need **I/O and fast array operations**.
- Use `ForceField` + `Typifier` when you need **reproducible parameters**.

If you remember one sentence: **edit on graphs, compute/export on arrays.**
