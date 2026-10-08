# I/O

Every file reader and writer: structure, trajectory and force-field files,
SMILES, `*.mrec` records, frame bytes and LAMMPS logs. `mp.io` mirrors
`molrs.io` by identity (`mp.io.read_pdb is molrs.io.read_pdb`). A file
factory has one of two shapes: a `read_<fmt>` / `write_<fmt>` function at the
top of `mp.io`, or a `<Fmt>Reader` / `<Fmt>Writer` class of the format's own
submodule. A class that belongs to one format lives in that submodule —
`mp.io.smiles` (`SmilesIr`, `SmilesError`, …), `mp.io.cgsmiles`
(`CgSmilesIr`, …), `mp.io.lammps` (`LammpsLog`, `LammpsDumpReader`,
`BondReactTemplate`, …), `mp.io.mrec` (`MrecReader`, `MrecWriter`, …), and
each trajectory format's reader (`mp.io.pdb.PdbReader`, `mp.io.xyz.XyzReader`,
`mp.io.gro.GroReader`, `mp.io.dcd.DcdReader`, `mp.io.trr.TrrReader`,
`mp.io.xtc.XtcReader`). Each is a molpy module mirroring the molrs one; molpy
adds its metric readers to `mp.io.lammps` and `mp.io.mrec`, and
`mp.io.mlp_jsonl` holds `MlpJsonlMetricReader`.

## Quick reference

### Data files

| Function | Format | Direction |
|----------|--------|-----------|
| `read_pdb` / `write_pdb` | PDB | read/write |
| `read_lammps_data` / `write_lammps_data` | LAMMPS data | read/write |
| `read_lammps_molecule` / `write_lammps_molecule` | LAMMPS molecule template | read/write |
| `read_gro` / `write_gro` | GROMACS GRO | read/write |
| `read_mol2` / `write_mol2` | MOL2 | read/write |
| `read_xyz` / `write_xyz` | XYZ | read/write |
| `read_xsf` / `write_xsf` | XSF (crystallographic) | read/write |
| `read_cube` / `write_cube` | Gaussian Cube | read/write |
| `read_vasp_chgcar` | VASP CHGCAR | read |
| `read_amber_inpcrd` | AMBER inpcrd (optionally into an existing frame) | read |
| `read_amber_prmtop` | AMBER prmtop (structure) | read |
| `read_amber_ac` | Antechamber AC | read |
| `read_smiles_str` / `write_smiles_str` | one molecule from / to SMILES (connectivity only; a `.`-separated set is refused) | read |
| `read_cgsmiles_str` | one molecule from CGsmiles, its lowest level expanded into atoms (topology only) | read |
| `smiles.SmilesIr`, `cgsmiles.CgSmilesIr` | SMILES / CGsmiles text (`SmilesIr(s).to_atomistic()`, `.components()`) | parse / emit |
| `write_lammps_bond_react_system`, `write_lammps_bond_react_map`, `lammps.BondReactTemplate` | LAMMPS `fix bond/react` | write |
| `read_msgpack_frame_bytes` / `write_msgpack_frame_bytes` | one frame in the wire encoding `mp.stream.Publisher` streams (`"msgpack"` / `"json"`) | read/write |

### Force fields

The files map onto `mp.ff.forcefield.ForceField`, the data model
`mp.ff.forcefield` owns.


| Function | Format | Direction |
|----------|--------|-----------|
| `read_openmm_xml_forcefield` / `write_openmm_xml_forcefield` | OpenMM/OPLS XML | read/write |
| `read_lammps_forcefield` / `write_lammps_forcefield` | LAMMPS `*.ff` include | read/write (engine units ↔ LAMMPS `real`) |
| `read_lammps_data_coeffs` / `read_lammps_data_coeffs_str` / `write_lammps_data_coeffs_str` | LAMMPS data `* Coeffs` | read (file or text) / write (text) |
| `read_gromacs_top_forcefield` / `write_gromacs_top_forcefield` | GROMACS `.top` / `.itp` directives | read/write |
| `read_gromacs_top_system` / `write_gromacs_top_system` | GROMACS topology: force field + typed frame | read/write |
| `read_amber_prmtop_forcefield` / `write_amber_frcmod` | AMBER prmtop / frcmod | read / write |
| `read_amber_prmtop_system` | AMBER prmtop: force field + typed frame (per-pair 1-4 weights in `pairs`) | read |

### Trajectories

| Function | Format | Direction |
|----------|--------|-----------|
| `read_lammps_dump_trajectory` / `write_lammps_dump_trajectory` | LAMMPS dump | read (lazy) / write |
| `write_lammps_dump_local` | LAMMPS dump local (bonds) | write |
| `read_xyz_trajectory` / `write_xyz_trajectory` | XYZ trajectory | read (lazy) / write |
| `read_pdb_trajectory` / `write_pdb_trajectory` | Multi-MODEL PDB | read (lazy) / write |
| `read_gro_trajectory` / `write_gro_trajectory` | Multi-frame GRO | read (lazy) / write |
| `read_dcd_trajectory` / `write_dcd_trajectory` | DCD | read (lazy) / write |
| `read_trr_trajectory` / `write_trr_trajectory` | GROMACS TRR | read (lazy) / write |
| `read_xtc_trajectory` / `write_xtc_trajectory` | GROMACS XTC | read (lazy) / write |
| `read_mrec_trajectory` / `write_mrec_trajectory` | mrec record | read / write (whole trajectory) |
| `mrec.MrecReader` | mrec record | read (lazy cursor) |
| `mrec.MrecWriter` | mrec record | write (append-first) |

Names pair: `read_X` / `write_X` for one frame, `read_X_trajectory` /
`write_X_trajectory` for a sequence. A `*.mrec` record is read and written
whole by functions at the top of `mp.io`: one-frame records use `read_mrec_frame` /
`write_mrec_frame` (snapshot) and `read_mrec_system` / `write_mrec_system`
(topology); `read_mrec_meta(path)` reads a record's identity document and
`mrec.section_names(path)` lists what a record holds. A force field rides in
the `forcefield` section: `write_mrec_frame(..., forcefield=ff)` /
`write_mrec_system(..., forcefield=ff)` or `write_mrec_forcefield(path, ff)`
write it, `read_mrec_forcefield(path)` returns a `mrec.ForceFieldSection` (or
`None`), and `section.to_forcefield()` turns it back into a force
field.

### Logs

| Function | Format | Direction |
|----------|--------|-----------|
| `read_lammps_log` / `read_lammps_log_str` | LAMMPS log (a path / in-memory text) → `lammps.LammpsLog` | read |

### Metric readers (molpy's)

molpy publishes three readers in the `molcrafts.metric_readers` entry-point
group, for any molcrafts viewer to find by format. Each turns a parsed file
into plottable series records; none parses a format itself.

| Class | Format | Entry point |
|-------|--------|-------------|
| `lammps.LammpsLogMetricReader` | `log.lammps` thermo tables | `lammps_log` |
| `mlp_jsonl.MlpJsonlMetricReader` | `*.mlp.jsonl` metric records (tailable) | `mlp_jsonl` |
| `mrec.MrecMetricReader` | the `step` / `time` series of a `*.mrec` trajectory | `mrec` |

## Canonical examples

```python
# docs: skip — reads offline artifact files; I/O unit-tested with fixtures
import molpy as mp

# Read/write structure
frame = mp.io.read_pdb("molecule.pdb")
mp.io.write_lammps_data("system.data", frame)

# Read force field (XML or LAMMPS *.ff)
ff = mp.io.read_openmm_xml_forcefield(mp.resources.get_path("forcefield/tip3p.xml"))
ff = mp.io.read_lammps_forcefield("system.ff")

# Write the LAMMPS coefficients the frame's type labels use
mp.io.write_lammps_forcefield("system.ff", ff, frame)

# Read trajectory (lazy)
traj = mp.io.read_lammps_dump_trajectory("dump.lammpstrj")
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

### LAMMPS data files

::: molpy.io.read_lammps_data

::: molpy.io.write_lammps_data

### LAMMPS `fix bond/react`

::: molpy.io.lammps.BondReactTemplate

::: molpy.io.write_lammps_bond_react_system

### mrec (scientific records)

The whole-record functions are at the top of `mp.io`: `read_mrec_frame` /
`write_mrec_frame`, `read_mrec_system` / `write_mrec_system`, `read_mrec_trajectory`
/ `write_mrec_trajectory`, `read_mrec_forcefield` / `write_mrec_forcefield`
and `read_mrec_meta`. `mp.io.mrec` mirrors `molrs.io.mrec`: `MrecReader`,
`MrecWriter`, `SequenceSchema`, `ForceFieldSection`, `section_names`, `pack_mrec_zip`
and `validation`, plus molpy's `MrecMetricReader`.

### Metric readers

::: molpy.io.lammps.LammpsLogMetricReader

::: molpy.io.mlp_jsonl.MlpJsonlMetricReader

::: molpy.io.mrec.MrecMetricReader
