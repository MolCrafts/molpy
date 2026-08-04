"""GROMACS .gro file I/O — molrs (Rust) backend.

Reader/writer delegate to :mod:`molrs.io` (parsing + field canonicalization in
Rust). :class:`GroFieldFormatter` is retained for the formatter hierarchy.
"""

from pathlib import Path

import molrs.io

from molpy.core.fields import RES_ID, RES_NAME, FieldFormatter
from molrs import Frame

from .base import DataReader, DataWriter


class GroFieldFormatter(FieldFormatter):
    """GROMACS .gro field name translation."""

    _field_formatters = {
        "res_number": RES_ID,
        "res_name": RES_NAME,
    }


class GroReader(DataReader):
    """Read GRO files via the molrs Rust backend."""

    _formatter = GroFieldFormatter()

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        frames = molrs.io.read_gro(self._path)
        if not frames:
            raise OSError(f"no frames parsed from GRO file: {self._path}")
        # Already a canonical rich Frame from molrs.io: coordinates in x/y/z,
        # and no element information (a GRO file carries none), so no atomic
        # number either.
        return frames[0]


class GroWriter(DataWriter):
    """Write GRO files via the molrs Rust backend."""

    _formatter = GroFieldFormatter()

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_gro(self._path, frame)
