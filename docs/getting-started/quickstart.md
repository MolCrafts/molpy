# Quickstart

Two ways in. The **fast path** runs the whole pipeline in six lines. The **full
walkthrough** then builds a TIP3P water box by hand — template, typing, box,
export — so you see every boundary you will later automate.

## The fast path: SMILES to a typed system

```python
import molpy as mp

mol = mp.io.smiles.SmilesIr("CCO").to_atomistic()  # ethanol from SMILES (heavy atoms)
mol, _ = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(
    mol
)  # add hydrogens + 3D coordinates
typifier = mp.ff.typifier.OplsAaTypifier()  # carries the OPLS-AA library
typed = typifier.typify(mol)  # assign force-field types
ff = typifier.forcefield()  # the parameters of the types just assigned

frame = typed.to_frame()  # columnar arrays
print(frame["atoms"].n_rows, "typed atoms")  # 9 typed atoms
```

That is the entire MolPy story — parse, embed, typify, convert. Every guide in
this manual is a variation on those boundaries. Now do the same thing with
full control, one explicit step at a time.

## The full walkthrough: a TIP3P water box

The rest of this page builds a small TIP3P water box end to end and exports
LAMMPS input files:

- A LAMMPS data file (`.data`) describing the typed system
- A LAMMPS force-field file (`.ff`) containing TIP3P coefficients

```python
import math
from pathlib import Path

import molpy as mp
```

### 1. Define a TIP3P water molecule

`Atomistic` is MolPy's chemistry-first container: a molecular graph where atoms
are nodes and bonds are edges. The bundled `tip3p.xml` is read into the store
units, **Å** and radians, so the coordinates are in Å too.

```python
water_template = mp.Atomistic(name="water_tip3p")

# TIP3P geometry: r(O-H) = 0.9572 Å, H-O-H = 104.52°
o = water_template.def_atom(element="O", name="O", mass=15.999, charge=-0.834, x=0.0, y=0.0, z=0.0)
h1 = water_template.def_atom(element="H", name="H1", mass=1.008, charge=0.417, x=0.9572, y=0.0, z=0.0)
h2 = water_template.def_atom(element="H", name="H2", mass=1.008, charge=0.417, x=-0.2400, y=0.9266, z=0.0)

water_template.def_bond(o, h1, order=1)
water_template.def_bond(o, h2, order=1)

# perceive angles from the bonds, in place (returns the counts added, not self)
water_template.generate_topology(gen_angle=True, gen_dihedral=False)

print("atoms:", len(water_template.atoms), "bonds:", len(water_template.bonds))
print("angles:", len(water_template.angles))
print("atom names:", [a.get("name") for a in water_template.atoms])
```

### 2. Assign TIP3P types

Load the bundled `tip3p.xml` and put its type names on the template.

> **Note:** nothing here has to *decide* a type. A rigid TIP3P water is three
> named sites, so the atoms, the bonds and the angle simply carry the
> force field's type labels; the parameters are looked up by label when the
> force field is compiled or exported. Reach for a typifier
> (`mp.ff.typifier.OplsAaTypifier`, …) when the types are the unknown.

The reader names each bonded type by joining its endpoint atom types with `::`
(the atom-type names contain `-`). A label is matched to a type name exactly, so
it is spelled the way the force field lists it:

```python
ff = mp.io.read_openmm_xml_forcefield(mp.resources.get_path("forcefield/tip3p.xml"))
print("bond types:", [t.name for t in ff.get_types("bond")])
print("angle types:", [t.name for t in ff.get_types("angle")])

for atom, atom_type in zip(water_template.atoms, ["tip3p-O", "tip3p-H", "tip3p-H"]):
    atom["type"] = atom_type
for bond in water_template.bonds:
    bond["type"] = "tip3p-O::tip3p-H"
for angle in water_template.angles:
    angle["type"] = "tip3p-H::tip3p-O::tip3p-H"

print("atom types:", [a.get("type") for a in water_template.atoms])
print("bond types:", [b.get("type") for b in water_template.bonds])
print("angle types:", [a.get("type") for a in water_template.angles])
```

### 3. Instantiate and transform a molecule

A template is a reusable `Atomistic`; an instance is a copy you place in a
larger system. Transforms are deterministic rigid-body operations —
`translate(delta)`, `rotate(axis, angle, about=None)` (angle in radians) and
`scale([sx, sy, sz], about=None)` — that move the structure in place and return
it, so they chain:

```python
water_instance = water_template.copy().rotate(
    [0.0, 0.0, 1.0], math.pi / 2.0, about=[0.0, 0.0, 0.0]
).translate([5.0, 0.0, 0.0])

print("instance center (Å):", water_instance.center().tolist())
```

### 4. Build a water box

Place copies on a simple 3D grid inside a cubic periodic box. (This is a
deterministic grid, not a packing algorithm — for clash-free packing at target
density, see [Packing Systems](../user-guide/09_packing.md).)

```python
n = 4  # molecules per edge
spacing = 3.2  # Å

water_box = mp.Atomistic(name="water_box_tip3p")
for idx in range(n**3):
    ix, iy, iz = idx % n, (idx // n) % n, idx // (n * n)
    mol = water_template.copy().rotate([0.0, 0.0, 1.0], 0.1 * idx)
    water_box.merge(mol.translate([ix * spacing, iy * spacing, iz * spacing]))

box = mp.Box.cube(n * spacing)
print("box lengths (Å):", box.lengths.tolist())
print("box atoms:", len(water_box.atoms), "box bonds:", len(water_box.bonds))
```

`merge` copies the type labels with the atoms, bonds and angles, so the box is
typed without a second pass.

### 5. Convert `Atomistic` to `Frame`

`Frame` is the columnar container — named tables plus the simulation box and
metadata. Writers operate on `Frame`, so this is the boundary where your edited
graph becomes exportable tables.

```python
frame = water_box.to_frame()
frame.box = box  # box is a first-class Frame attribute; writers read frame.box

# atom_style full needs a molecule id: one per connected component
frame["atoms"]["mol_id"] = mp.Topology.from_frame(frame).connected_components() + 1

print("atoms rows:", frame["atoms"].n_rows)
print("bonds rows:", frame["bonds"].n_rows)
print("angles rows:", frame["angles"].n_rows)
```

The LAMMPS data writer refuses a bonded frame without `mol_id`, so this step is
not optional.

### 6. Export to LAMMPS files

The force-field writer takes the frame as well: every type label of the frame
is looked up in `ff`, and only the coefficients the system uses are written.
The `pair_style` line needs a cutoff. A cutoff is a run setting, not a
force-field parameter, so you declare it on the pair styles; molpy never
invents one.

```python
ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
ff.get_style("pair", "coul/cut")["cutoff"] = 10.0

out_dir = Path("quickstart-output")
out_dir.mkdir(parents=True, exist_ok=True)

mp.io.write_lammps_data(out_dir / "water_box_tip3p.data", frame)
mp.io.write_lammps_forcefield(out_dir / "water_box_tip3p.ff", ff, frame)

print("wrote:", out_dir / "water_box_tip3p.data")
print("wrote:", out_dir / "water_box_tip3p.ff")
```

## What you built

- A TIP3P water molecule as an editable `Atomistic` graph — derived angles,
  and type labels that name the bundled `tip3p.xml` parameters.
- 64 molecules placed deterministically in a periodic box.
- A `Frame` with box attached, exported as LAMMPS data + force-field files.

**Next:** the [Example Gallery](examples.md) for more workflows to copy, or
the [data-model tutorials](../tutorials/index.md) to understand each object
you just used.
