# Quickstart

Two ways in. The **fast path** runs the whole pipeline in six lines. The **full
walkthrough** then builds a TIP3P water box by hand — template, typing, box,
export — so you see every boundary you will later automate.

## The fast path: SMILES to a typed system

```python
import molpy as mp

mol   = mp.parser.parse_molecule("CCO")                  # ethanol from SMILES (heavy atoms)
mol   = mp.adapter.generate_3d(mol, add_hydrogens=True)  # add hydrogens + 3D coordinates
ff    = mp.io.read_xml_forcefield("oplsaa.xml")          # bundled OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(mol)         # assign force-field types

frame = typed.to_frame()                                 # columnar arrays
print(frame["atoms"].nrows, "typed atoms")               # 9 typed atoms
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
from pathlib import Path

import numpy as np
import molpy as mp
from molpy.io.forcefield import read_xml_forcefield
from molpy.typifier import OplsTypifier
```

### 1. Define a TIP3P water molecule

`Atomistic` is MolPy's chemistry-first container: a molecular graph where atoms
are nodes and bonds are edges. TIP3P's bond length in the bundled `tip3p.xml`
is in **nm**, so we build coordinates in **nm** to match.

```python
water_template = mp.Atomistic(name='water_tip3p')

theta = 1.82421813418  # rad (TIP3P H-O-H angle)
r_oh = 0.09572  # nm (TIP3P O-H bond length)

o = water_template.def_atom(element='O', name='O', x=0.0, y=0.0, z=0.0, charge=-0.834)
h1 = water_template.def_atom(element='H', name='H1', x=r_oh, y=0.0, z=0.0, charge=0.417)
h2 = water_template.def_atom(
    element='H',
    name='H2',
    x=r_oh * float(np.cos(theta)),
    y=r_oh * float(np.sin(theta)),
    z=0.0,
    charge=0.417,
 )

water_template.def_bond(o, h1, order=1)
water_template.def_bond(o, h2, order=1)

# get_topo perceives angles/dihedrals on a *copy* (non-mutating) — capture it
water_template = water_template.get_topo(gen_angle=True, gen_dihe=False)

print('atoms:', len(water_template.atoms), 'bonds:', len(water_template.bonds))
print('angles:', len(list(water_template.links.bucket(mp.Angle))))
print('atom names:', [a.get('name') for a in water_template.atoms])
```

### 2. Assign TIP3P types

Load the bundled `tip3p.xml` and let `OplsTypifier` assign atom types, bonded
types, and nonbonded parameters — deterministically, from built-in data only.

> **Note:** `OplsTypifier` is MolPy's general-purpose SMARTS-based typing
> engine, not an OPLS-only tool. It matches atoms against the patterns in
> whatever force field you load — here `tip3p.xml` — and assigns the types that
> file defines. The name refers to the matching engine, not the force field.

```python
ff = read_xml_forcefield('tip3p.xml')

typifier = OplsTypifier(
    ff,
    skip_atom_typing=False,
    skip_dihedral_typing=True,
    strict_typing=True,
 )
water_template = typifier.typify(water_template)

print('atom types:', [a.get('type') for a in water_template.atoms])
print('bond types:', [b.get('type') for b in water_template.bonds])
print('angle types:', [a.get('type') for a in water_template.links.bucket(mp.Angle)])
print('example LJ params on O:', {k: water_template.atoms[0].get(k) for k in ['sigma', 'epsilon']})
```

### 3. Instantiate and transform a molecule

A template is a reusable `Atomistic`; an instance is a copy you place in a
larger system. Transforms are deterministic rigid-body operations:

```python
water_instance = water_template.copy()

water_instance.rotate(axis=[0.0, 0.0, 1.0], angle=float(np.pi / 2.0), about=[0.0, 0.0, 0.0])
water_instance.move(delta=[0.5, 0.0, 0.0])

coords = np.array([[a['x'], a['y'], a['z']] for a in water_instance.atoms], dtype=float)
print('instance center (nm):', coords.mean(axis=0).tolist())
```

### 4. Build a water box

Place copies on a simple 3D grid inside an orthogonal periodic box. (This is a
deterministic grid, not a packing algorithm — for clash-free packing at target
density, see [Packing Systems](../user-guide/09_packing.md).)

```python
nx, ny, nz = 4, 4, 4
spacing = 0.32  # nm
n_total = nx * ny * nz

water_box_atomistic = mp.Atomistic(name='water_box_tip3p')
mol_id = 1
idx = 0

for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            mol = water_template.copy()
            mol.rotate(axis=[0.0, 0.0, 1.0], angle=float(0.1 * idx), about=[0.0, 0.0, 0.0])
            mol.move(delta=[ix * spacing, iy * spacing, iz * spacing])
            for atom in mol.atoms:
                atom['mol_id'] = mol_id
            water_box_atomistic.merge(mol)
            mol_id += 1
            idx += 1

water_box_atomistic = typifier.typify(water_box_atomistic)

box = mp.Box.orth([nx * spacing, ny * spacing, nz * spacing])
print('box lengths (nm):', box.lengths.tolist())
print('box atoms:', len(water_box_atomistic.atoms), 'box bonds:', len(water_box_atomistic.bonds))
```

### 5. Convert `Atomistic` to `Frame`

`Frame` is the columnar container — named tables plus the simulation box and
metadata. Writers operate on `Frame`, so this is the boundary where your edited
graph becomes exportable tables.

```python
frame = water_box_atomistic.to_frame()
frame.box = box  # box is a first-class Frame attribute; writers read frame.box

atoms = frame['atoms']
n_atoms = atoms.nrows

atoms['id'] = np.arange(1, n_atoms + 1, dtype=int)
atoms['mol_id'] = np.asarray(atoms['mol_id'], dtype=int)
atoms['charge'] = np.asarray(atoms['charge'], dtype=float)

print('atoms rows:', frame['atoms'].nrows)
print('bonds rows:', frame['bonds'].nrows)
# `to_frame()` only emits a block for link kinds that exist. This TIP3P
# template carries bonds but no explicit angle links, so there is no
# 'angles' block — guard before accessing it.
if 'angles' in frame:
    print('angles rows:', frame['angles'].nrows)
```

### 6. Export to LAMMPS files

```python
out_dir = Path('quickstart-output')
out_dir.mkdir(parents=True, exist_ok=True)

mp.io.write_lammps_data(out_dir / 'water_box_tip3p.data', frame, atom_style='full')
mp.io.write_lammps_forcefield(out_dir / 'water_box_tip3p.ff', ff)

print('wrote:', out_dir / 'water_box_tip3p.data')
print('wrote:', out_dir / 'water_box_tip3p.ff')
```

## What you built

- A TIP3P water molecule as an editable `Atomistic` graph — then types,
  parameters, and derived angles from the bundled `tip3p.xml`.
- 64 molecules placed deterministically in a periodic box.
- A `Frame` with box attached, exported as LAMMPS data + force-field files.

**Next:** the [Example Gallery](examples.md) for more workflows to copy, or
the [data-model tutorials](../tutorials/index.md) to understand each object
you just used.
