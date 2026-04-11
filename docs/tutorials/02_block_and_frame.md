# Block and Frame

This page explains how MolPy stores molecular data in aligned tables, groups those tables into complete snapshots, and uses them for export and downstream computation.

## Why two representations?

Molecular dynamics engines (LAMMPS, GROMACS, OpenMM) do not read molecular graphs. They read flat tables of coordinates, atom types, and index-based topology. Conversely, building a molecule from SMILES or assembling a polymer requires graph traversal, not array slicing.

MolPy keeps both representations explicit rather than hiding the conversion. `Atomistic` is the graph you edit; `Block` and `Frame` are the tables you export. If you are familiar with other tools: a `Frame` plays a similar role to a LAMMPS data file, a GROMACS `.gro` + `.top` pair, or an MDAnalysis `Universe` snapshot — but as a pure in-memory data structure rather than a file format.

## From graph to table

An `Atomistic` object is the right place to edit chemistry: add an atom, remove a bond, query neighbors. But once the chemistry is stable, the next question is usually numerical — distances, energies, file export. For that kind of work, aligned arrays are far more convenient than a graph of dictionary-like objects.

MolPy uses two data structures for this purpose. A `Block` is one columnar table: column names map to NumPy arrays, and every column refers to the same set of rows. A `Frame` is a named collection of blocks plus free-form metadata, representing one complete system snapshot.

The split is deliberate. A block answers "what are the atoms?" or "what are the bonds?" — one table for one kind of data. A frame answers "what is the full state of this system right now?" by grouping related tables together.


## Block: a columnar table backed by NumPy

Creating a block is as simple as passing a dictionary of array-like values. Each value becomes a NumPy array automatically.

```python
import molpy as mp
import numpy as np

atoms = mp.Block({
    "element": ["O", "H", "H"],
    "x": [0.000, 0.957, -0.239],
    "y": [0.000, 0.000, 0.927],
    "z": [0.000, 0.000, 0.000],
})

print(atoms.nrows)          # 3
print(list(atoms.keys()))   # ['element', 'x', 'y', 'z']
```

Reading a column returns an `np.ndarray`. This means all of NumPy is immediately available — no conversion step, no special accessor.

```python
print(atoms["x"].dtype)         # float64
print(atoms["element"].dtype)   # <U1 (Unicode string)
```

A common pattern is stacking numeric columns into a 2D array for vectorized computation.

```python
xyz = atoms[["x", "y", "z"]]   # shape (3, 3)
r = np.linalg.norm(xyz, axis=1)
print(r)
```


## Row selection returns a new Block

Slicing, boolean masks, and fancy indexing all produce a new `Block`. The original is never modified.

```python
hydrogens = atoms[atoms["element"] == "H"]
print(hydrogens.nrows)           # 2
print(hydrogens["x"])            # [0.957, -0.239]

first_two = atoms[0:2]
print(first_two["element"])      # ['O', 'H']
```

If you need a single scalar value, index the column first, then the row.

```python
print(atoms["x"][0])   # 0.0
```


## Adding and removing columns

Setting a key inserts or overwrites a column. Deleting a key removes it. Both operations follow standard Python mapping conventions.

```python
atoms_with_r = atoms.copy()
atoms_with_r["r"] = np.linalg.norm(atoms_with_r[["x", "y", "z"]], axis=1)
print(list(atoms_with_r.keys()))   # ['element', 'x', 'y', 'z', 'r']

del atoms_with_r["r"]
print(list(atoms_with_r.keys()))   # ['element', 'x', 'y', 'z']
```


## Renaming columns

`Block.rename()` changes a column key in place. This is used internally by the I/O formatter system to translate between format-specific and canonical field names.

```python
b = mp.Block({"q": [0.1, -0.2], "x": [1.0, 2.0]})
b.rename("q", "charge")
print(list(b.keys()))   # ['x', 'charge']
```


## Copy semantics matter

`Block.copy()` is shallow: the mapping is copied, but the underlying NumPy arrays are shared. In-place mutation of an array affects both the original and the copy.

```python
shallow = atoms.copy()
shallow["x"][0] = 999.0
print(atoms["x"][0])    # 999.0 — original changed too!
```

If you need full independence, copy the arrays explicitly. The safest pattern is to copy each column you intend to modify:

```python
# Rebuild clean data for the rest of the page
atoms = mp.Block({
    "element": ["O", "H", "H"],
    "x": [0.000, 0.957, -0.239],
    "y": [0.000, 0.000, 0.927],
    "z": [0.000, 0.000, 0.000],
})

deep = atoms.copy()
deep["x"] = deep["x"].copy()
deep["x"][0] = 999.0
print(atoms["x"][0])    # 0.0 — original unchanged
```

!!! tip "Avoiding mutation"
    The idiomatic MolPy pattern is to avoid in-place array mutation entirely. Instead of modifying a column, assign a new array: `block["x"] = block["x"] + 1.0`. This always produces an independent copy and is consistent with MolPy's immutable-data philosophy.


## Frame: a named collection of Blocks

A molecular system usually needs more than one table. Atom coordinates are one table, bond indices are another, and the snapshot itself has metadata — a timestep, a description, provenance. `Frame` groups all of that into one object.

```python
frame = mp.Frame(
    blocks={
        "atoms": {
            "element": ["O", "H", "H"],
            "x": [0.000, 0.957, -0.239],
            "y": [0.000, 0.000, 0.927],
            "z": [0.000, 0.000, 0.000],
        },
        "bonds": {
            "atomi": [0, 0],
            "atomj": [1, 2],
        },
    },
    timestep=0,
    description="water",
)
```

Keyword arguments beyond `blocks` are stored in `frame.metadata`, a plain dictionary.

```python
print(frame.metadata)   # {'timestep': 0, 'description': 'water'}
```

Accessing a block by name returns a `Block`. From there, all column operations work the same way.

```python
atoms = frame["atoms"]
print(atoms["x"])   # [0.000, 0.957, -0.239]
```

You can add, replace, or delete blocks at any time.

```python
frame["tags"] = {"label": ["oxygen", "hydrogen", "hydrogen"]}
print(type(frame["tags"]))   # <class 'molpy.core.frame.Block'>

del frame["tags"]
print("tags" in frame)       # False
```


## Box is a first-class attribute

A periodic simulation cell is attached directly to `frame.box`, not stored in metadata. This ensures `Frame.copy()` preserves the box and I/O round-trips work correctly.

```python
frame.box = mp.Box.cubic(20.0)
print(frame.box.lengths)   # [20. 20. 20.]

# copy() preserves box
frame2 = frame.copy()
print(frame2.box.lengths)   # [20. 20. 20.]
```

`frame.box` is `None` when no box has been assigned (e.g., for isolated molecules).


## Serialization round-trips through dictionaries

Both `Block` and `Frame` support `to_dict()` and `from_dict()` for JSON-friendly serialization. This is the stable way to persist or transmit system state without tying yourself to a specific file format.

```python
payload = frame.to_dict()
print(sorted(payload.keys()))   # ['blocks', 'metadata']

restored = mp.Frame.from_dict(payload)
print(sorted(restored.to_dict()["blocks"].keys()))   # ['atoms', 'bonds']
```


## When Block and Frame are the right choice

Use `Atomistic` when you still need to edit the molecular graph — adding atoms, defining bonds, querying connectivity. Use `Block` and `Frame` when the chemistry is settled and your next task involves arrays, export, or analysis.

The two representations can coexist. Many workflows keep an `Atomistic` around for reference while producing frames for numerical work. The important thing is knowing which object carries the meaning you care about at each stage.

Once your system lives in a periodic cell, coordinates alone are not enough — distances depend on the simulation box. That is the subject of the next page.

See also: [Atomistic and Topology](01_atomistic_and_topology.md), [Box and Periodicity](03_box_and_periodicity.md).
