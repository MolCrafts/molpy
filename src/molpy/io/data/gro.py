"""GROMACS .gro file I/O — molrs (Rust) backend.

Reader/writer delegate to :mod:`molrs.io` (parsing + field canonicalization in
Rust). :class:`GroFieldFormatter` is retained for the formatter hierarchy.
"""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.fields import RES_ID, RES_NAME, FieldFormatter
from molpy.core.frame import Frame

from .base import DataReader, DataWriter


def _ensure_xyz(frame: Frame) -> Frame:
    """Return a frame whose atoms block has separate x/y/z columns (the molrs
    coordinate convention), deriving them from a combined ``xyz`` column if
    needed. Works on a copy; never mutates the caller's frame."""
    atoms = frame["atoms"]
    if "x" in atoms or "xyz" not in atoms:
        return frame
    out = frame.copy()
    xyz = np.asarray(out["atoms"]["xyz"])
    out["atoms"]["x"] = xyz[:, 0]
    out["atoms"]["y"] = xyz[:, 1]
    out["atoms"]["z"] = xyz[:, 2]
    return out


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
        molpy_frame = Frame.from_dict(frames[0])

        atoms = molpy_frame["atoms"]
        if "number" not in atoms and "id" in atoms:
            atoms["number"] = atoms["id"].astype(np.int64, copy=False)
        if "xyz" not in atoms and "x" in atoms and "y" in atoms and "z" in atoms:
            atoms["xyz"] = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
        return molpy_frame


class GroWriter(DataWriter):
    """Write GRO files via the molrs Rust backend."""

    _formatter = GroFieldFormatter()

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_gro(self._path, _ensure_xyz(frame))
