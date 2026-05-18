"""Experimental GRO file I/O — molrs-backed reader and writer."""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.frame import Frame

from . import DataReader, DataWriter


class GroReader(DataReader):
    """Experimental. Read GRO files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_gro` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self) -> Frame:
        frames = molrs.io.read_gro(self._path)
        raw = frames[0]
        molpy_frame = Frame.from_dict(raw)

        atoms = molpy_frame["atoms"]

        if "number" not in atoms and "id" in atoms:
            id_col = atoms["id"]
            atoms["number"] = id_col.astype(np.int64, copy=False)

        if "xyz" not in atoms and "x" in atoms and "y" in atoms and "z" in atoms:
            atoms["xyz"] = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])

        return molpy_frame


class GroWriter(DataWriter):
    """Experimental. Write GRO files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.write_gro` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs_frame = frame.to_molrs()
        molrs.io.write_gro(self._path, molrs_frame)
