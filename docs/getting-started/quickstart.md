[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/getting-started/quickstart.ipynb)

# Quickstart

This quickstart constructs a small TIP3P water box end to end in MolPy. It defines a water molecule, assigns TIP3P types, places multiple molecules in a periodic box, converts the result to a `Frame`, and exports LAMMPS input files.

**Final output:**
- A LAMMPS data file (`.data`) describing the typed system
- A LAMMPS force-field file (`.ff`) containing TIP3P coefficients

```python
from pathlib import Path

import numpy as np
import molpy as mp
from molpy.io.forcefield import read_xml_forcefield
from molpy.typifier import OplsAtomisticTypifier
```

## 1. Define a TIP3P Water Molecule

`Atomistic` is MolPy's chemistry-first container: a molecular graph where atoms are nodes and bonds are edges.

This quickstart uses the TIP3P geometry and parameters (from MolPy's built-in `tip3p.xml`). TIP3P's bond length in that file is in **nm**, so we build coordinates in **nm** to match.

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

water_template.get_topo(gen_angle=True, gen_dihe=False)

print('atoms:', len(water_template.atoms), 'bonds:', len(water_template.bonds))
print('angles:', len(list(water_template.links.bucket(mp.Angle))))
print('atom names:', [a.get('name') for a in water_template.atoms])
```

## 2. Assign TIP3P Types

We load MolPy's built-in TIP3P force field file (`tip3p.xml`), then use `OplsAtomisticTypifier` to assign:
- atom types
- bonded types (bonds/angles)
- nonbonded parameters (sigma/epsilon)

This is deterministic and uses only MolPy's built-in API + built-in force field data.

```python
ff = read_xml_forcefield('tip3p.xml')

typifier = OplsAtomisticTypifier(
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

## 3. Instantiate and Transform a Molecule

A template is a reusable `Atomistic`. An instance is a copy you place in a larger system.

Here we show a deterministic rigid-body transform: rotate around $z$ and translate.

```python
water_instance = water_template.copy()

water_instance.rotate(axis=[0.0, 0.0, 1.0], angle=float(np.pi / 2.0), about=[0.0, 0.0, 0.0])
water_instance.move(delta=[0.5, 0.0, 0.0])

coords = np.array([[a['x'], a['y'], a['z']] for a in water_instance.atoms], dtype=float)
print('instance center (nm):', coords.mean(axis=0).tolist())
```

## 4. Build a Water Box

We place waters on a simple 3D grid inside an orthogonal periodic box. This is deterministic and is not a packing algorithm.

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

## 5. Convert `Atomistic` to `Frame`

`Frame` is MolPy's columnar container (named tables like `atoms`, `bonds`, ...) plus metadata such as the simulation box and force field.

Writers operate on `Frame`, so this is the boundary where your edited graph becomes exportable tables.

```python
frame = water_box_atomistic.to_frame()
frame.metadata['box'] = box

atoms = frame['atoms']
n_atoms = atoms.nrows

atoms['id'] = np.arange(1, n_atoms + 1, dtype=int)
atoms['mol_id'] = np.asarray(atoms['mol_id'], dtype=int)
atoms['charge'] = np.asarray(atoms['charge'], dtype=float)

print('atoms rows:', frame['atoms'].nrows)
print('bonds rows:', frame['bonds'].nrows)
print('angles rows:', frame['angles'].nrows)
```

## 6. Export to LAMMPS Files

We write:
- A LAMMPS data file for the structure
- A LAMMPS force-field file for TIP3P parameters

```python
out_dir = Path('quickstart-output')
out_dir.mkdir(parents=True, exist_ok=True)

mp.io.write_lammps_data(out_dir / 'water_box_tip3p.data', frame, atom_style='full')
mp.io.write_lammps_forcefield(out_dir / 'water_box_tip3p.ff', ff)

print('wrote:', out_dir / 'water_box_tip3p.data')
print('wrote:', out_dir / 'water_box_tip3p.ff')
```

## 7. Summary

- Built a TIP3P water molecule as an editable `Atomistic` graph.
- Loaded TIP3P parameters from MolPy's built-in `tip3p.xml`.
- Assigned TIP3P atom types and used `OplsAtomisticTypifier` to assign bonded and nonbonded parameters.
- Placed many molecules on a deterministic grid in a periodic box.
- Converted to a `Frame` and wrote LAMMPS data + force-field files.
