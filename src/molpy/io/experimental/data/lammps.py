"""Experimental LAMMPS data file I/O — molrs-backed."""

from pathlib import Path
from typing import Any

import molrs.io

from molpy.core.frame import Frame

from . import DataReader, DataWriter


class LammpsDataReader(DataReader):
    """Experimental. Read LAMMPS data files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_lammps_data` for the stable implementation.
    """

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def read(self, frame: Frame | None = None) -> Frame:
        raw = molrs.io.read_lammps_data(self._path, self.atom_style, frame)
        return Frame.from_molrs(raw)


class LammpsDataWriter(DataWriter):
    """Experimental. Write LAMMPS data files.

    Currently delegates to the stable
    :class:`molpy.io.data.lammps.LammpsDataWriter`.  Will switch to
    ``molrs.io.write_lammps_data`` once molrs gains Python-side Frame
    construction APIs.

    .. deprecated::
        Use :func:`molpy.io.write_lammps_data` for the stable implementation.
    """

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def write(self, frame: Frame) -> None:
        from molpy.io.data.lammps import LammpsDataWriter as _StableWriter

        _StableWriter(self._path, self.atom_style).write(frame)
