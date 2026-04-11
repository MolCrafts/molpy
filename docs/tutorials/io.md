# I/O: Reading, Writing, and Extending File Formats

This page describes how MolPy reads and writes molecular data, force fields, and trajectories, and how additional formats can be added through the reader and writer interfaces.

MolPy's I/O system has three layers: **data** (single-frame structures), **trajectory** (multi-frame sequences), and **forcefield** (parameter files). Each layer has its own reader/writer base class and extension pattern.


## Data files: Frame in, Frame out

Data readers consume a file and return a `Frame`. Data writers take a `Frame` and produce a file. The `Frame` is the universal exchange object — it carries atom tables, bond tables, metadata, and optionally a box.

### Reading

Every format has a factory function at `mp.io.read_*`:

```python
import molpy as mp

frame = mp.io.read_pdb("molecule.pdb")
frame = mp.io.read_gro("system.gro")
frame = mp.io.read_mol2("ligand.mol2")
frame = mp.io.read_lammps_data("system.data", atom_style="full")
frame = mp.io.read_xyz("coords.xyz")
frame = mp.io.read_h5("snapshot.h5")
```

All readers follow the same pattern: pass a file path, get a `Frame` back. An optional `frame` argument lets you populate an existing frame instead of creating a new one.

```python
# Populate an existing frame (useful for merging metadata)
existing = mp.Frame(timestep=42)
frame = mp.io.read_pdb("molecule.pdb", frame=existing)
print(frame.metadata["timestep"])  # 42
```

The `Frame` returned by a data reader typically contains an `"atoms"` block with coordinates, and may contain `"bonds"`, `"angles"`, and `"dihedrals"` blocks depending on the format.

```python
frame = mp.io.read_pdb("water.pdb")
print(frame["atoms"].nrows)           # number of atoms
print(list(frame["atoms"].keys()))    # ['id', 'element', 'x', 'y', 'z', ...]
```

### Writing

Writers follow the same factory pattern:

```python
mp.io.write_pdb("output.pdb", frame)
mp.io.write_gro("output.gro", frame)
mp.io.write_lammps_data("output.data", frame, atom_style="full")
mp.io.write_h5("output.h5", frame)
mp.io.write_xsf("output.xsf", frame)
```

For LAMMPS, writing both the data file and force field coefficients at once is the most common pattern. `write_lammps_system` handles this:

```python
mp.io.write_lammps_system("output_dir", frame, ff)
# Creates output_dir.data and output_dir.ff
```

### Supported data formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PDB | `read_pdb` | `write_pdb` | ATOM/HETATM + CRYST1 + CONECT |
| GRO | `read_gro` | `write_gro` | GROMACS structure |
| MOL2 | `read_mol2` | — | Tripos MOL2 |
| LAMMPS data | `read_lammps_data` | `write_lammps_data` | Requires `atom_style` |
| LAMMPS molecule | `read_lammps_molecule` | `write_lammps_molecule` | Template files |
| XYZ | `read_xyz` | — | Simple coordinate format |
| XSF | `read_xsf` | `write_xsf` | XCrySDen format |
| HDF5 | `read_h5` | `write_h5` | Binary, compressed |
| AMBER AC | `read_amber_ac` | — | Antechamber format |
| AMBER inpcrd | `read_amber_inpcrd` | — | AMBER coordinates |


## Trajectory files: lazy Frame sequences

Trajectory readers return objects that behave like indexed, iterable sequences of frames. They use memory-mapped files and persistent indexing to handle large trajectories efficiently.

### Reading

```python
reader = mp.io.read_lammps_trajectory("dump.lammpstrj")

print(reader.n_frames)       # total frame count
frame_0 = reader[0]          # random access by index
frame_last = reader[-1]      # negative indexing
subset = reader[10:20]       # slicing returns list[Frame]
```

Iteration is lazy — frames are parsed on demand with background prefetching:

```python
for frame in reader:
    atoms = frame["atoms"]
    # process one frame at a time
```

Always close the reader when done (or use a context manager) to release memory-mapped file descriptors:

```python
reader.close()
```

### Writing

Trajectory writers accept a list of frames:

```python
mp.io.write_lammps_trajectory("output.lammpstrj", frames, atom_style="full")
mp.io.write_xyz_trajectory("output.xyz", frames)
mp.io.write_h5_trajectory("output.h5", frames)
```

### Supported trajectory formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| LAMMPS dump | `read_lammps_trajectory` | `write_lammps_trajectory` | Custom columns supported |
| XYZ | `read_xyz_trajectory` | `write_xyz_trajectory` | Multi-frame XYZ |
| HDF5 | `read_h5_trajectory` | `write_h5_trajectory` | Binary, compressed, fast |


## Force field files: ForceField in, ForceField out

Force field readers parse parameter files into `ForceField` objects. Force field writers serialize `ForceField` objects into engine-specific formats.

### Reading

```python
ff = mp.io.read_xml_forcefield("oplsaa.xml")
ff = mp.io.read_lammps_forcefield("system.ff")
ff = mp.io.read_top("forcefield.itp")
```

AMBER prmtop files contain both structure and parameters. `read_amber` returns both:

```python
frame, ff = mp.io.read_amber("system.prmtop", "system.inpcrd")
```

### Writing

Each writer produces output in a specific engine format from the same `ForceField` object:

```python
from molpy.io.forcefield import LAMMPSForceFieldWriter, XMLForceFieldWriter
from molpy.io.forcefield.top import GromacsForceFieldWriter

# LAMMPS coefficients
LAMMPSForceFieldWriter("system.ff", precision=4).write(ff)

# GROMACS .itp
GromacsForceFieldWriter("system.itp", precision=4).write(ff)

# OpenMM XML
XMLForceFieldWriter("system.xml", precision=6).write(ff)
```

### Type filtering

LAMMPS force field files often need to include only the types actually used in the system. Pass type sets to filter the output:

```python
LAMMPSForceFieldWriter("system.ff").write(
    ff,
    atom_types={"CT", "HC", "OH"},
    bond_types={"CT-HC", "CT-OH"},
)
```

### Supported force field formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| OpenMM/OPLS XML | `read_xml_forcefield` | `XMLForceFieldWriter` | Primary force field format |
| LAMMPS coefficients | `read_lammps_forcefield` | `LAMMPSForceFieldWriter` | Supports hybrid styles |
| GROMACS .itp | `read_top` | `GromacsForceFieldWriter` | Topology format |
| AMBER prmtop | `read_amber` | — | Returns `(Frame, ForceField)` |


## Extending: add a new data format

Subclass `DataReader` or `DataWriter` from `molpy.io.data.base`. Implement `read()` or `write()`.

### New reader

```python
from pathlib import Path
from molpy.core.frame import Frame, Block
from molpy.io.data.base import DataReader

class CifReader(DataReader):
    """Read CIF crystallography files."""

    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        if frame is None:
            frame = Frame()

        # self.fh is the lazy file handle (auto-opens on first access)
        text = self.fh.read()

        # parse the format, populate blocks
        frame["atoms"] = Block({
            "element": elements,
            "x": x_coords,
            "y": y_coords,
            "z": z_coords,
        })
        return frame
```

### New writer

```python
from molpy.io.data.base import DataWriter

class CifWriter(DataWriter):
    """Write CIF crystallography files."""

    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    def write(self, frame: Frame) -> None:
        atoms = frame["atoms"]
        self.fh.write("data_molpy\n")
        # ... write CIF sections
```

### Register as factory function

Add the reader/writer to `molpy/io/readers.py` and `molpy/io/writers.py`:

```python
# In readers.py
def read_cif(file: PathLike, frame: Frame | None = None) -> Frame:
    return CifReader(Path(file)).read(frame)

# In writers.py
def write_cif(file: PathLike, frame: Frame) -> None:
    CifWriter(Path(file)).write(frame)
```


## Extending: add a new trajectory format

Subclass `BaseTrajectoryReader` or `TrajectoryWriter`. Trajectory readers use memory-mapped files for performance.

### New trajectory reader

The reader must implement `_scan_frames` (build an index of frame byte offsets) and `_parse_frame_bytes` or `_parse_frame` (parse one frame from the index).

```python
import mmap
from molpy.io.trajectory.base import BaseTrajectoryReader
from molpy.io.trajectory.index import FrameEntry

class DcdTrajectoryReader(BaseTrajectoryReader):
    """Read DCD binary trajectory files."""

    _format_id = "dcd"

    def _scan_frames(self, file_idx: int, mm: mmap.mmap) -> list[FrameEntry]:
        """Scan file for frame boundaries, return byte offset entries."""
        entries = []
        # parse DCD header, find frame offsets
        # for each frame: entries.append(FrameEntry(...))
        return entries

    def _parse_frame_bytes(self, mm: mmap.mmap, entry: FrameEntry) -> Frame:
        """Parse one frame from memory-mapped bytes."""
        # read coordinates from mm[entry.offset:entry.offset+entry.length]
        return frame
```

The persistent index (`*.tridx` or `*.tridx.json`) is built automatically on first read and cached for subsequent accesses.

### New trajectory writer

```python
from molpy.io.trajectory.base import TrajectoryWriter

class DcdTrajectoryWriter(TrajectoryWriter):
    """Write DCD trajectory files."""

    def write_frame(self, frame: Frame) -> None:
        atoms = frame["atoms"]
        # write binary frame data to self._fp
```


## Extending: add a new force field format

Force field extension has two parts: the writer class and the formatter registration.

### New writer

Subclass `ForceFieldWriter` and implement `write()`:

```python
from molpy.io.forcefield.base import ForceFieldWriter, ForceFieldFormatter

class CharmForceFieldFormatter(ForceFieldFormatter):
    """Formatter for CHARMM parameter files."""
    pass

class CharmForceFieldWriter(ForceFieldWriter):
    """Write CHARMM .prm files."""

    formatter_cls = CharmForceFieldFormatter

    def write(self, forcefield, **kwargs):
        with open(self._fpath, "w") as f:
            # iterate styles and types
            for style in forcefield.get_styles(BondStyle):
                style_params = self._get_style_params(style)
                for typ in style.get_types(BondType):
                    type_params = self._get_type_params(typ, style)
                    f.write(format_charmm_line(typ, type_params))
```

### Register formatters for custom styles

If you add a custom `Style` (e.g., `MorseBondStyle`), each writer that should support it needs a formatter registration:

```python
from molpy.io.forcefield.base import FormattedParams

# Style formatter: receives Style, returns metadata
def format_morse_style(style) -> FormattedParams:
    return FormattedParams()

# Type formatter: receives (Type, Style), returns coefficients
def format_morse_type(typ, style) -> FormattedParams:
    return FormattedParams(positional=[typ["D"], typ["alpha"], typ["r0"]])

LAMMPSForceFieldWriter.register_style_formatter(
    MorseBondStyle, format_morse_style, style_name="morse",
)
LAMMPSForceFieldWriter.register_formatter(
    MorseBondStyle, format_morse_type, style_name="morse",
)
```

Registrations are **isolated per writer subclass** — adding a formatter to one writer does not affect another. This isolation is enforced by `__init_subclass__` copying the registry.

`FormattedParams` carries two fields:

- `positional` — ordered values (LAMMPS writes them space-separated after the coeff ID)
- `keyword` — named values (XML writes them as attributes, GROMACS uses them for `funct` and column metadata)


## Quick reference

| I want to... | Function |
|---|---|
| Read a PDB | `mp.io.read_pdb(path)` |
| Write LAMMPS data + ff | `mp.io.write_lammps_system(dir, frame, ff)` |
| Read a trajectory | `mp.io.read_lammps_trajectory(path)` |
| Load OPLS-AA | `mp.io.read_xml_forcefield("oplsaa.xml")` |
| Read AMBER topology | `mp.io.read_amber(prmtop, inpcrd)` |
| Write filtered ff | `LAMMPSForceFieldWriter(path).write(ff, atom_types={...})` |
| Add a new format | Subclass `DataReader`/`DataWriter` |
| Add a new trajectory format | Subclass `BaseTrajectoryReader`/`TrajectoryWriter` |
| Support custom Style in export | `Writer.register_formatter(StyleClass, fn)` |

See also: [API Reference: I/O](../api/io.md), [Concepts: Force Field](../tutorials/04_force_field.md).
