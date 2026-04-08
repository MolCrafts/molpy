# I/O

File readers and writers for molecular data, force fields, and trajectories.

## Quick reference

### Data files

| Function | Format | Direction |
|----------|--------|-----------|
| `read_pdb` / `write_pdb` | PDB | read/write |
| `read_lammps_data` / `write_lammps_data` | LAMMPS data | read/write |
| `read_gro` / `write_gro` | GROMACS GRO | read/write |
| `read_mol2` | MOL2 | read |
| `read_xyz` | XYZ | read |
| `read_h5` / `write_h5` | HDF5 | read/write |
| `read_amber_inpcrd` | AMBER inpcrd | read |

### Force fields

| Function | Format | Direction |
|----------|--------|-----------|
| `read_xml_forcefield` | OpenMM/OPLS XML | read |
| `XMLForceFieldWriter` | OpenMM/OPLS XML | write |
| `LAMMPSForceFieldWriter` | LAMMPS coefficients | write |
| `GromacsForceFieldWriter` | GROMACS .itp | write |
| `read_amber` | AMBER prmtop + inpcrd | read |

### Trajectories

| Function | Format | Direction |
|----------|--------|-----------|
| `read_lammps_trajectory` | LAMMPS dump | read (lazy) |
| `read_xyz_trajectory` | XYZ trajectory | read (lazy) |
| `read_h5_trajectory` | HDF5 trajectory | read (lazy) |

## Canonical examples

```python
import molpy as mp

# Read/write structure
frame = mp.io.read_pdb("molecule.pdb")
mp.io.write_lammps_data("system.data", frame, atom_style="full")

# Read force field
ff = mp.io.read_xml_forcefield("oplsaa.xml")

# Write LAMMPS force field
from molpy.io.forcefield import LAMMPSForceFieldWriter
LAMMPSForceFieldWriter("system.ff").write(ff, atom_types={"CT", "HC"})

# Read trajectory (lazy)
traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
for frame in traj:
    process(frame)

# Write full LAMMPS system (data + ff)
mp.io.write_lammps_system("output_dir", frame, ff)
```

## Related

- [Concepts: Block and Frame](../tutorials/02_block_and_frame.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Factory Functions

::: molpy.io.readers
    options:
      members: true
      filters: ["!^Base"]

::: molpy.io.writers
    options:
      members: true

### ForceField Modules

#### Base
::: molpy.io.forcefield.base

#### LAMMPS
::: molpy.io.forcefield.lammps

#### XML
::: molpy.io.forcefield.xml

#### GROMACS Topology
::: molpy.io.forcefield.top

#### AMBER
::: molpy.io.forcefield.amber

### Data Modules

#### LAMMPS
::: molpy.io.data.lammps
::: molpy.io.data.lammps_molecule

#### PDB
::: molpy.io.data.pdb

#### GRO
::: molpy.io.data.gro

#### Mol2
::: molpy.io.data.mol2

#### H5
::: molpy.io.data.h5

#### Amber
::: molpy.io.data.amber

#### AC
::: molpy.io.data.ac

#### Top
::: molpy.io.data.top

#### XYZ
::: molpy.io.data.xyz

### Trajectory Modules

#### Base
::: molpy.io.trajectory.base

#### LAMMPS
::: molpy.io.trajectory.lammps

#### H5
::: molpy.io.trajectory.h5

#### XYZ
::: molpy.io.trajectory.xyz
