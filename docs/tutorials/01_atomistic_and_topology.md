# Atomistic and Topology

How do you add a leaving group, break a bond, or ask “is this carbon aromatic”
before any coordinates go near an engine?

A coordinate list cannot answer. You need a **connectivity model**: which atoms
are bonded, and what follows from that. In MolPy that model is an editable
molecular graph.

**`Atomistic` is a molecular graph you edit.** Atoms are nodes, bonds are edges.
Angles, dihedrals, and rings are *derived* from connectivity when you ask — not
hand-maintained tables that go stale.

What it is **not**: a simulation snapshot, a force-field file, or a packed box.
Those live on `Frame` and friends after the chemistry is settled
([Block and Frame](02_block_and_frame.md)).

## Why a graph?

Simulation tools face two kinds of question.

*Geometric* questions — “how far apart are these two atoms?” — need coordinates.
*Chemical* questions — “which atoms share a covalent bond?”, “what happens when
I break this bond?” — need connectivity.

A plain list of positions answers only the first kind. A molecular graph answers
both *identities* and *relations*: you know *which* carbon is bonded to *which*
oxygen, not merely that row 3 sits near row 5. That is why building, editing,
and reacting molecules happen here, before the system is handed to an engine.

## Building a molecule

Start empty, then add atoms and bonds. Properties are free-form keywords —
`element`, coordinates, `charge`, or anything else your workflow needs.

```python
import molpy as mp

mol = mp.Atomistic(name="ethanol")
c1 = mol.def_atom(element="C", name="C1", x=0.0, y=0.0, z=0.0)
c2 = mol.def_atom(element="C", name="C2", x=1.54, y=0.0, z=0.0)
o = mol.def_atom(element="O", name="O1", x=2.0, y=1.4, z=0.0)
h_o = mol.def_atom(element="H", name="HO", x=2.9, y=1.4, z=0.0)

mol.def_bond(c1, c2, order=1)
mol.def_bond(c2, o, order=1)
mol.def_bond(o, h_o, order=1)
print(f"{len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# -> 4 atoms, 3 bonds
```

Those atoms and bonds are live objects inside the graph — not detached copies.
Change a property and the molecule changes immediately.

```python
print(c1["element"])  # -> C
c1["charge"] = -0.18
print(c1["charge"])  # -> -0.18

bond = mol.bonds[0]
print(bond.itom["name"], bond.jtom["name"], bond.get("order"))
# -> C1 C2 1
```

## Connectivity lives in the molecule, not in the atom

An atom does not store its own neighbour list. The container owns connectivity,
so graph operations stay explicit and consistent.

```python
neighbors = mol.get_neighbors(c2)
print([n["name"] for n in neighbors])
# -> ['C1', 'O1']
```

Removing an atom removes incident bonds with it — you never keep a dangling edge.

```python
print(f"Before: {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# -> Before: 4 atoms, 3 bonds
mol.remove_entity(h_o)
print(f"After:  {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# -> After:  3 atoms, 2 bonds
```

## Topology is derived from bonds

Engines need angles and dihedrals as well as bonds. Maintaining those lists by
hand is brittle: every bond edit would require a matching update.

MolPy treats bonded topology as **derived from the bond graph**. `get_topo`
reads the current bonds and writes perceived angles and dihedrals into the same
`Atomistic` (in place; returns `self` for chaining). If the bond graph changes
later, call `get_topo` again.

```python
propane = mp.Atomistic(name="propane")
ca = propane.def_atom(element="C", name="C1", x=0.0, y=0.0, z=0.0)
cb = propane.def_atom(element="C", name="C2", x=1.54, y=0.0, z=0.0)
cc = propane.def_atom(element="C", name="C3", x=3.08, y=0.0, z=0.0)
propane.def_bond(ca, cb)
propane.def_bond(cb, cc)

print(len(propane.angles), len(propane.dihedrals))
# -> 0 0
propane.get_topo(gen_angle=True, gen_dihe=True)
print(len(propane.angles), len(propane.dihedrals))
# -> 1 0

for angle in propane.angles:
    print(" — ".join(a["name"] for a in angle.endpoints))
# -> C1 — C2 — C3
```

Graph-distance queries (neighbours within $n$ bonds, BFS distances) run on the
same structure once connectivity is defined:

```python
print([a["name"] for a in propane.get_topo_neighbors(cb, radius=1)])
# -> ['C1', 'C2', 'C3']
dists = propane.get_topo_distances(ca)
print({a["name"]: d for a, d in dists.items()})
# -> {'C1': 0, 'C2': 1, 'C3': 2}
```

## Composition and copies

Independent clones use `copy()`. Merges use `+`; many copies use `replicate`.

```python
water = mp.Atomistic(name="water")
ow = water.def_atom(element="O", x=0.0, y=0.0, z=0.0)
h1 = water.def_atom(element="H", x=0.957, y=0.0, z=0.0)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.0)
water.def_bond(ow, h1)
water.def_bond(ow, h2)

two = water + water.copy().move([5.0, 0.0, 0.0])
print(len(two.atoms), len(two.bonds))
# -> 6 4

box = water.replicate(4, lambda m, i: m.move([i * 4.0, 0.0, 0.0]))
print(len(box.atoms))
# -> 12
```

!!! note "Bulk numeric access"
    When you need every $x$ coordinate as an array — not each atom as a Python
    object — use `mol.atoms["x"]`, `mol.xyz`, or `mol.column("x")`. Prefer
    `list(mol.atoms)` only when you need identity-stable objects for editing or
    graph algorithms.

## When to stay here, when to leave

Stay on `Atomistic` while the **chemistry** is still under discussion: add
atoms, define bonds, inspect connectivity, run reactions.

When the chemistry is stable and the next job is export, analysis, or
simulation, move to arrays: [Block and Frame](02_block_and_frame.md).

## Check yourself

1. Why can two atoms with identical element and coordinates still be different
   atoms in MolPy?
2. After `remove_entity` on a terminal hydrogen, how many bonds should remain on
   the heavy-atom skeleton of the ethanol example above?
3. If you add a bond after `get_topo`, are the angle lists automatically up to
   date? What do you call?

## See also

- [Block and Frame](02_block_and_frame.md) — arrays for compute and I/O
- [API: Core](../api/core.md) — full surface of `Atomistic`, `Atom`, `Bond`
