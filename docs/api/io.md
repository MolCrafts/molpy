# I/O

File readers and writers for molecular data, force fields, and trajectories.
Every name is a function on `mp.io`; most are the native core's own readers and
writers, re-exported by identity.

## Quick reference

### Data files

| Function | Format | Direction |
|----------|--------|-----------|
| `read_pdb` / `write_pdb` | PDB | read/write |
| `read_lammps_data` / `write_lammps_data` | LAMMPS data | read/write |
| `read_lammps_molecule` / `write_lammps_molecule` | LAMMPS molecule template | read/write |
| `read_gro` / `write_gro` | GROMACS GRO | read/write |
| `read_top` / `write_top` | GROMACS topology (structure) | read/write |
| `read_mol2` / `write_mol2` | MOL2 | read/write |
| `read_xyz` / `write_xyz` | XYZ | read/write |
| `read_xsf` / `write_xsf` | XSF (crystallographic) | read/write |
| `read_cube` / `write_cube` | Gaussian Cube | read/write |
| `read_chgcar` | VASP CHGCAR | read |
| `read_amber_inpcrd` | AMBER inpcrd | read |
| `read_amber_ac` | Antechamber AC | read |
| `read_smiles` / `write_smarts` | SMILES / local SMARTS | read / write |

### Force fields

| Function | Format | Direction |
|----------|--------|-----------|
| `read_xml_forcefield` / `write_xml_forcefield` | OpenMM/OPLS XML | read/write |
| `read_lammps_forcefield` / `write_lammps_forcefield` | LAMMPS `*.ff` include | read/write (engine units ↔ LAMMPS `real`) |
| `read_lammps_data_coeffs` / `write_lammps_data_coeffs` | LAMMPS data `* Coeffs` | read/write |
| `read_gromacs_forcefield` / `write_gromacs_forcefield` | GROMACS `.top` / `.itp` directives | read/write |
| `read_amber` | AMBER prmtop (+ inpcrd) | read |

### Trajectories

| Function | Format | Direction |
|----------|--------|-----------|
| `read_lammps_trajectory` / `write_lammps_trajectory` | LAMMPS dump | read (lazy) / write |
| `write_lammps_dump_local` | LAMMPS dump local (bonds) | write |
| `read_xyz_trajectory` / `write_xyz_trajectory` | XYZ trajectory | read (lazy) / write |
| `read_pdb_trajectory` / `write_pdb_trajectory` | Multi-MODEL PDB | read (list) / write |
| `read_gro_trajectory` / `write_gro_trajectory` | Multi-frame GRO | read (list) / write |
| `read_dcd_trajectory` / `write_dcd_trajectory` | DCD | read (lazy) / write |
| `read_trr_trajectory` / `write_trr_trajectory` | GROMACS TRR | read (lazy) / write |
| `read_xtc_trajectory` / `write_xtc_trajectory` | GROMACS XTC | read (lazy) / write |
| `read_mrec_trajectory` / `write_mrec_trajectory` | mrec store | read / write (whole trajectory) |
| `mrec.TrajectoryReader` | mrec store | read (lazy cursor) |
| `mrec.TrajectoryWriter` | mrec store | write (append-first) |

Names pair: `read_X` / `write_X` for one frame, `read_X_trajectory` /
`write_X_trajectory` for a sequence. One-frame mrec stores use `read_mrec` /
`write_mrec` (snapshot) and `read_mrec_system` / `write_mrec_system`
(topology); `read_mrec_meta(path)` reads a store's identity document and
`mrec_sections(path)` lists what a store holds. A force field rides in the
`forcefield` section: `write_mrec(..., forcefield=ff)` /
`write_mrec_system(..., forcefield=ff)` or `write_mrec_forcefield(path, ff)`
write it, `read_mrec_forcefield(path)` returns a `mrec.ForceFieldSection` (or
`None`), and `ForceField.from_section(section)` turns it back into a force
field.

### Logs

| Function | Format | Direction |
|----------|--------|-----------|
| `read_lammps_log` / `parse_lammps_log_text` | LAMMPS log | read |

## Canonical examples

```python
# docs: skip — reads offline artifact files; I/O unit-tested with fixtures
import molpy as mp

# Read/write structure
frame = mp.io.read_pdb("molecule.pdb")
mp.io.write_lammps_data("system.data", frame)

# Read force field (XML or LAMMPS *.ff)
ff = mp.io.read_xml_forcefield(mp.data.get_forcefield_path("tip3p.xml"))
ff = mp.io.read_lammps_forcefield("system.ff")

# Write the LAMMPS coefficients the frame's type labels use
mp.io.write_lammps_forcefield("system.ff", ff, frame)

# Read trajectory (lazy)
traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
for frame in traj:
    process(frame)

# Read LAMMPS run output
log = mp.io.read_lammps_log("log.lammps")
thermo = log.runs[0].thermo
print(thermo.columns)
```

## Related

- [Concepts: Block and Frame](../tutorials/02_block_and_frame.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Readers molpy owns

::: molpy.io.readers

### Writers molpy owns

::: molpy.io.writers

### LAMMPS data files

::: molpy.io.data.lammps
::: molpy.io.data.lammps_bond_react

### mrec (scientific record stores)

::: molpy.io.mrec
