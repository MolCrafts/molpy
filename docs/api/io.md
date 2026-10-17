# I/O

`mp.io` mirrors `molrs.io`. Single-frame formats have `read_<format>(source)`
and `write_<format>(sink, frame)` functions. Pass a `pathlib.Path` for a file,
`bytes` or `mp.io.Text(text)` for in-memory input, and a `bytearray` for
in-memory output. A bare string cannot identify whether it is a path or file
contents. Multi-frame readers and writers live in their format modules, for
example `mp.io.lammps.LammpsDumpReader(path)` and
`mp.io.lammps.LammpsDumpWriter(path)`.

| Data | API |
|------|-----|
| Structure files | `read_pdb` / `write_pdb`, `read_mol2` / `write_mol2`, `read_gro` / `write_gro`, `read_xyz` / `write_xyz`, `read_lammps_data` / `write_lammps_data`, plus CIF, CUBE, XSF, POSCAR, SDF and AMBER readers |
| Force fields | `mp.io.lammps.LammpsForcefieldReader` / `LammpsForcefieldWriter`; `mp.io.gromacs.GromacsTopForcefieldReader` / `GromacsTopForcefieldWriter`; corresponding classes in `amber`, `openmm_xml` and `xml` |
| Trajectories | `<Format>Reader` / `<Format>Writer` in `pdb`, `gro`, `xyz`, `lammps`, `dcd`, `trr` and `xtc` |
| SMILES and CGsmiles | `mp.io.smiles.SmilesIr.parse(text)`, `mp.io.cgsmiles.CgSmilesIr.parse(text)` |
| Scientific records | `mp.io.write_mrec(path, frame=..., trajectory=..., forcefield=...)`, `mp.io.mrec.MrecFile(path)`, `mp.io.mrec.MrecReader(path)` |
| Wire frames | `mp.io.read_wire_frame` / `mp.io.write_wire_frame` |
| Logs and metrics | `mp.io.read_lammps_log`; molpy's `lammps.LammpsLogMetricReader`, `mrec.MrecMetricReader` and `mlp_jsonl.MlpJsonlMetricReader` |

The format-specific force-field classes use `read(source)` and
`write(sink, forcefield)`. GROMACS and AMBER readers also expose
`read_system(source)` for a force field and its typed frame. LAMMPS data
coefficients are read with `LammpsForcefieldReader().read_data_coeffs(frame)`.

For a record, `MrecFile.sections()` lists its sections and `frame()`,
`system()`, `meta()` and `forcefield()` read them. `MrecReader` accesses its
trajectory by index or iteration.

### Metric readers (molpy's)

molpy publishes three readers in the `molcrafts.metric_readers` entry-point
group, for any molcrafts viewer to find by format. Each turns a parsed file
into plottable series records; none parses a format itself.

| Class | Format | Entry point |
|-------|--------|-------------|
| `lammps.LammpsLogMetricReader` | `log.lammps` thermo tables | `lammps_log` |
| `mlp_jsonl.MlpJsonlMetricReader` | `*.mlp.jsonl` metric records (tailable) | `mlp_jsonl` |
| `mrec.MrecMetricReader` | the `step` / `time` series of a `*.mrec` trajectory (store directory or packed `*.mrec.zip`) | `mrec` |

Every reader enforces a viewer's request the same way:

- `since` is a raw index: the number of records of the reader's unfiltered,
  unstrided stream to skip (LAMMPS: row-major over thermo rows and columns;
  mrec: every `step` record, then every `time` record; mlp_jsonl: line order).
- `stride` keeps every *n*-th sample of each series after `since` and the
  `metric_type` / `keys` filters; `limit` caps the records yielded.
- A record's `s` is the simulation step where the source records one (the
  LAMMPS `Step` column, the mrec `step` column), else the sample's ordinal in
  its source; mlp_jsonl records keep the `s` their writer gave them.
- `MlpJsonlMetricReader` skips only a torn last line (no trailing newline, not
  yet parseable); any other malformed line raises `ValueError` with its line
  number.

## Example

```python
from pathlib import Path
import molpy as mp

frame = mp.io.read_pdb(Path("molecule.pdb"))
mp.io.write_pdb(Path("copy.pdb"), frame)

record = Path("snapshot.mrec")
mp.io.write_mrec(record, frame=frame)
restored = mp.io.mrec.MrecFile(record).frame()
```

## Full API

### LAMMPS data files

::: molpy.io.read_lammps_data

::: molpy.io.write_lammps_data

### LAMMPS `fix bond/react`

::: molpy.io.lammps.BondReactTemplate

::: molpy.io.write_lammps_bond_react_system

### mrec (scientific records)

`mp.io.write_mrec(path, frame=..., system=..., trajectory=..., forcefield=...)`
writes a record. `mp.io.mrec.MrecFile(path)` reads its `sections()`, `meta()`,
`frame()`, `system()` and `forcefield()`. `mp.io.mrec.MrecReader(path)` reads
trajectory frames lazily.

### Metric readers

::: molpy.io.lammps.LammpsLogMetricReader

::: molpy.io.mlp_jsonl.MlpJsonlMetricReader

::: molpy.io.mrec.MrecMetricReader
