"""GROMACS .gro file I/O — thin molrs wrappers.

Parse/serialize live in :mod:`molrs.io`. :class:`GroFieldFormatter` documents
format-native names for the FieldFormatter hierarchy (no separate Python parser).
"""

from __future__ import annotations

from pathlib import Path

import molrs.io
from molrs import Frame

from molpy.core.fields import RES_ID, RES_NAME, FieldFormatter

from .base import DataReader, DataWriter


class GroFieldFormatter(FieldFormatter):
    """GROMACS .gro field name translation (documentation / hierarchy only)."""

    _field_formatters = {
        "res_number": RES_ID,
        "res_name": RES_NAME,
    }


class GroReader(DataReader):
    """Read GRO via molrs (first frame if multi-frame)."""

    _formatter = GroFieldFormatter()

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        del frame
        frames = molrs.io.read_gro(str(self._path))
        if not frames:
            raise OSError(f"no frames parsed from GRO file: {self._path}")
        return frames[0]


class GroWriter(DataWriter):
    """Write GRO via molrs."""

    _formatter = GroFieldFormatter()

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_gro(str(self._path), frame)
