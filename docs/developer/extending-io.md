# Adding an I/O Format

This page shows how to add readers and writers for new file formats and force field backends.

## Data file readers and writers

Subclass `DataReader` or `DataWriter` from `molpy.io.data.base`.

### Reader

```python
from pathlib import Path
from molpy.core.frame import Frame, Block
from molpy.io.data.base import DataReader

from molpy.core.fields import FieldFormatter, CHARGE

class MyFieldFormatter(FieldFormatter):
    """Field name translation for .myformat."""
    _field_formatters = {
        "q": CHARGE,   # .myformat uses "q" for charge
    }

class MyFormatReader(DataReader):
    """Read .myformat files into a Frame."""

    _formatter = MyFieldFormatter()

    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        if frame is None:
            frame = Frame()

        # Parse the file (self._path is set by FileBase)
        with open(self._path) as f:
            lines = f.readlines()

        # Populate blocks using format-native field names
        frame["atoms"] = Block({
            "element": [...],
            "x": [...],
            "y": [...],
            "z": [...],
        })

        # Translate format-specific field names to canonical names
        self._formatter.canonicalize_frame(frame)
        return frame
```

### Writer

```python
from molpy.io.data.base import DataWriter

class MyFormatWriter(DataWriter):
    """Write a Frame to .myformat."""

    _formatter = MyFieldFormatter()

    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    def write(self, frame: Frame) -> None:
        # Translate canonical names to format-specific (on a copy)
        frame = self._formatter.localize_frame(frame)

        atoms = frame["atoms"]
        with open(self._path, "w") as f:
            for i in range(atoms.nrows):
                f.write(f"{atoms['element'][i]} {atoms['x'][i]} ...\n")
```

### Register in factory functions

Add your reader/writer to `molpy/io/readers.py` and `molpy/io/writers.py` so they are accessible via `mp.io.read_myformat()` and `mp.io.write_myformat()`.


## Canonical field names

The internal data model uses canonical field names defined in `molpy.core.fields`. When your format uses different column names, define a `FieldFormatter` subclass with a `_field_formatters` mapping:

```python
from molpy.core.fields import FieldFormatter, FieldSpec, CHARGE, MOL_ID

class MyFieldFormatter(FieldFormatter):
    _field_formatters = {
        "q":   CHARGE,    # format "q" → canonical "charge"
        "mol": MOL_ID,    # format "mol" → canonical "mol_id"
    }
```

Key canonical fields: `charge` (not `q`), `mol_id` (not `mol`), `id`, `type`, `mass`, `element`, `x`/`y`/`z`.

If your format's field names already match the canonical names (e.g., MOL2 uses `charge`), no formatter is needed.


## Force field writers with the formatter hierarchy

The force field export system uses a **two-level formatter hierarchy** defined in `molpy.core.fields`:

```
FieldFormatter                         — data field mapping: {format_key: FieldSpec}
    ↓
ForceFieldFormatter(FieldFormatter)    — inherits field mapping + {StyleType: Callable}
```

Each format's `ForceFieldFormatter` subclass inherits the data field mapping from its `FieldFormatter` and adds parameter formatters for Style/Type serialization.

### Adding a param formatter for a custom Style

```python
from molpy.io.forcefield.lammps import LammpsForceFieldFormatter

def _format_morse_bond(typ) -> list[float]:
    return [typ.params.kwargs["D"], typ.params.kwargs["alpha"], typ.params.kwargs["r0"]]

LammpsForceFieldFormatter.register_param_formatter(MorseBondStyle, _format_morse_bond)
```

Registrations are **isolated per subclass** — adding a formatter to one writer does not affect another. This isolation is enforced by `__init_subclass__` copying the registry.


## Trajectory readers and writers

Trajectory readers use memory-mapped files and a persistent frame index for efficient random access. Subclass `BaseTrajectoryReader` and implement `_scan_frames` (build byte-offset index) and `_parse_frame_bytes` (parse one frame):

```python
import mmap
from molpy.io.trajectory.base import BaseTrajectoryReader
from molpy.io.trajectory.index import FrameEntry
from molpy.core.frame import Frame

class MyTrajectoryReader(BaseTrajectoryReader):
    _format_id = "myformat"

    def _scan_frames(self, file_idx: int, mm: mmap.mmap) -> list[FrameEntry]:
        entries = []
        # scan file for frame boundaries, record byte offsets
        return entries

    def _parse_frame_bytes(self, mm: mmap.mmap, entry: FrameEntry) -> Frame:
        # parse one frame from mm[entry.offset:entry.offset+entry.length]
        return frame
```

The persistent index (`.tridx`) is built automatically on first read and cached for subsequent accesses. Subclass `TrajectoryWriter` and implement `write_frame()` for writing.


## Checklist

- [ ] Subclass `DataReader`/`DataWriter` or `BaseTrajectoryReader`/`TrajectoryWriter`
- [ ] Define `FieldFormatter` subclass if format uses non-canonical field names
- [ ] Reader calls `_formatter.canonicalize_frame(frame)` before returning
- [ ] Writer calls `_formatter.localize_frame(frame)` at entry (operates on copy)
- [ ] Box stored on `frame.box`, not in `frame.metadata`
- [ ] Add factory function in `readers.py` / `writers.py`
- [ ] Register param formatters on `ForceFieldFormatter` subclass if adding a custom Style
- [ ] Write round-trip tests (`write → read → compare`) in `tests/test_io/`
- [ ] Round-trip verifies canonical field names (`charge`, `mol_id`, not `q`, `mol`)
