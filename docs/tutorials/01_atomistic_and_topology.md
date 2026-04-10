[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/tutorials/01_atomistic_and_topology.ipynb)

# Atomistic and Topology

After reading this page you will be able to build a molecule from atoms and bonds, query its connectivity, and derive angles and dihedrals automatically.

## Why a graph?

Molecular simulation tools must answer two kinds of questions. *Geometric* questions — "how far apart are these two atoms?" — only need coordinates. *Chemical* questions — "which atoms share a covalent bond?", "is this carbon part of an aromatic ring?", "what happens when I break this bond?" — need a connectivity model.

A plain list of positions cannot answer chemical questions. A molecular graph can. That is why MolPy starts with a graph representation: it is the natural data structure for building, editing, and reasoning about molecular topology before the system is handed off to a simulation engine.


## A molecule is a graph you can edit

In MolPy, `Atomistic` is an editable molecular graph. Atoms are nodes, bonds are edges. Unlike a table of coordinates, this representation preserves identity: you know *which* carbon is bonded to *which* oxygen, not just that row 3 happens to sit near row 5.

The reason this matters is practical. At the start of most workflows, the molecule is not finished. You still need to add atoms, remove leaving groups, mark reactive sites, or check connectivity. All of those operations are graph operations. `Atomistic` is the layer where you do them.

Building a molecule starts with an empty container.

```python
import molpy as mp

mol = mp.Atomistic(name="ethanol")
```

`def_atom` creates an atom and adds it to the graph. You pass whatever properties you need as keyword arguments — `symbol`, coordinates, `charge`, or anything else. There is no fixed schema.

```python
c1 = mol.def_atom(symbol="C", name="C1", x=0.0, y=0.0, z=0.0)
c2 = mol.def_atom(symbol="C", name="C2", x=1.54, y=0.0, z=0.0)
o  = mol.def_atom(symbol="O", name="O1", x=2.0, y=1.4, z=0.0)
h_o = mol.def_atom(symbol="H", name="HO", x=2.9, y=1.4, z=0.0)
```

`def_bond` connects two existing atoms. Like atoms, bonds accept arbitrary keyword attributes.

```python
mol.def_bond(c1, c2, order=1)
mol.def_bond(c2, o, order=1)
mol.def_bond(o, h_o, order=1)
print(f"{len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# 4 atoms, 3 bonds
```

At this point `mol` holds the heavy-atom skeleton of ethanol. Atoms and bonds are live objects inside the graph — not copies of data. The next sections show what you can do with them.


## Atoms and bonds behave like dictionaries

Every `Atom` and `Bond` is a dictionary-like object. You read and write properties with bracket notation or `.get()`.

```python
print(c1["symbol"])       # "C"
print(c1.get("charge"))   # None — charge was never set

c1["charge"] = -0.18
print(c1["charge"])       # -0.18
```

This works for bonds too. A `Bond` exposes its two endpoints through `.itom` and `.jtom`.

```python
bond = mol.bonds[0]
print(bond.itom, bond.jtom)   # <Atom: C> <Atom: C>
print(bond.get("order"))      # 1
```

Because atoms are references, modifying an atom object immediately changes the graph. There is no separate "commit" step.

```python
for atom in mol.atoms:
    if atom["symbol"] == "C":
        atom["hybridization"] = "sp3"

print(c2["hybridization"])  # "sp3"
```


## Connectivity lives in the container, not in the atom

An atom does not know its own neighbors. The `Atomistic` container manages all connectivity. This separation keeps atoms lightweight and makes graph operations explicit.

`get_neighbors` returns a list of atoms directly bonded to a given atom.

```python
neighbors = mol.get_neighbors(c2)
print([n["name"] for n in neighbors])  # ['C1', 'O1']
```

You can combine this with a loop to inspect the bonds around an atom.

```python
for bond in mol.bonds:
    if c2 in bond.endpoints:
        partner = bond.itom if bond.jtom is c2 else bond.jtom
        print(f"C2 —({bond.get('order')})— {partner['name']}")
```

Because neighbors are ordinary Python objects, building higher-level queries is straightforward. Here is a function that collects all atoms within *n* hops of a starting atom.

```python
def n_hop_neighbors(mol, start, n):
    visited = {start}
    shell = {start}
    for _ in range(n):
        next_shell = set()
        for atom in shell:
            for nb in mol.get_neighbors(atom):
                if nb not in visited:
                    visited.add(nb)
                    next_shell.add(nb)
        shell = next_shell
    return visited - {start}

print({a["name"] for a in n_hop_neighbors(mol, c1, 2)})
# {'O1', 'C2'}
```


## Removing an atom keeps the graph consistent

Deleting an atom with `remove_entity` automatically removes all incident bonds. You never end up with a dangling bond pointing to a missing atom.

```python
print(f"Before: {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# Before: 4 atoms, 3 bonds

mol.remove_entity(h_o)

print(f"After:  {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# After:  3 atoms, 2 bonds

print([n["name"] for n in mol.get_neighbors(o)])  # ['C2']
```

The O–H bond disappeared along with the hydrogen. The remaining graph is internally consistent.


## Copying produces an independent clone

`copy()` deep-copies all atoms and bonds into a new `Atomistic` object. The two graphs are fully independent — modifying one does not affect the other.

```python
mol_copy = mol.copy()

c1_copy = [a for a in mol_copy.atoms if a["name"] == "C1"][0]
c1_copy["name"] = "C1_copy"

print(c1["name"])       # "C1" — original unchanged
print(c1_copy["name"])  # "C1_copy"
```


## Systems compose with + and replicate

Two `Atomistic` objects can be merged with `+`. The result is a new object containing all atoms and bonds from both sides.

```python
water = mp.Atomistic(name="water")
ow = water.def_atom(symbol="O", x=0.0, y=0.0, z=0.0)
h1 = water.def_atom(symbol="H", x=0.957, y=0.0, z=0.0)
h2 = water.def_atom(symbol="H", x=-0.239, y=0.927, z=0.0)
water.def_bond(ow, h1)
water.def_bond(ow, h2)

two_waters = water + water.copy().move([5.0, 0.0, 0.0])
print(f"{len(two_waters.atoms)} atoms, {len(two_waters.bonds)} bonds")
# 6 atoms, 4 bonds
```

For many copies, `replicate` is more convenient. It takes a count and an optional transform function.

```python
box = water.replicate(4, lambda mol, i: mol.move([i * 4.0, 0.0, 0.0]))
print(f"{len(box.atoms)} atoms")  # 12
```


## Topology is derived, not stored

Molecular dynamics needs more than bonds. It needs angles (three-atom sequences) and dihedrals (four-atom sequences). Maintaining those by hand is error-prone — every time you add or remove a bond, every angle and dihedral list would need updating.

MolPy treats topology as a *derived view*. You call `get_topo` on an `Atomistic` object, and it reads the current bond graph to produce the full set of angles and dihedrals. If the graph changes, you re-derive.

Let's see this on a fresh molecule where all the heavy atoms are still present.

```python
propane = mp.Atomistic(name="propane")
ca = propane.def_atom(symbol="C", name="C1", x=0.0, y=0.0, z=0.0)
cb = propane.def_atom(symbol="C", name="C2", x=1.54, y=0.0, z=0.0)
cc = propane.def_atom(symbol="C", name="C3", x=3.08, y=0.0, z=0.0)
propane.def_bond(ca, cb)
propane.def_bond(cb, cc)

print(f"Before: {len(propane.angles)} angles, {len(propane.dihedrals)} dihedrals")
# Before: 0 angles, 0 dihedrals

propane.get_topo(gen_angle=True, gen_dihe=True)

print(f"After:  {len(propane.angles)} angles, {len(propane.dihedrals)} dihedrals")
# After:  1 angles, 0 dihedrals
```

Each `Angle` and `Dihedral` holds references to its endpoint atoms through `.endpoints`, just like a `Bond`.

```python
for angle in propane.angles:
    names = [a["name"] for a in angle.endpoints]
    print(" — ".join(names))
# C1 — C2 — C3
```


## The Topology object exposes graph algorithms

`get_topo` also returns a `Topology` object — a thin wrapper around an igraph graph. This gives you access to shortest paths, connected components, degree queries, and subgraph matching without leaving MolPy.

```python
topo = propane.get_topo()

print(f"atoms: {topo.n_atoms}, bonds: {topo.n_bonds}")
print(f"angles: {topo.n_angles}, dihedrals: {topo.n_dihedrals}")
```

Because `Topology` inherits from `igraph.Graph`, standard graph algorithms are available directly.

```python
print(f"connected: {topo.is_connected()}")
print(f"degrees:   {topo.degree()}")

path = topo.get_shortest_paths(0, topo.n_atoms - 1)[0]
print(f"shortest path 0→2: {path}")
# [0, 1, 2]
```

For a larger molecule the same tools scale naturally: finding ring systems, checking connectivity after a bond deletion, or measuring topological distances between functional groups all reduce to standard graph queries on the `Topology` object.


## When to stay here, when to move on

Use `Atomistic` as long as the structure itself is under discussion — adding atoms, defining bonds, inspecting connectivity, running reactions. This is the layer for *chemical editing*.

Once the chemistry is stable and your next task is export, analysis, or simulation, the right representation changes. [Block and Frame](02_block_and_frame.md) carry the same system as aligned arrays with explicit metadata — a better fit for numerical work and file I/O.

See also: [Block and Frame](02_block_and_frame.md), [API Reference: Core](../api/index.md).
