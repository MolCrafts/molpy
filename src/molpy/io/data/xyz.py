"""XYZ file I/O — molrs backend with thin molpy column normalization.

Parse/serialize: :mod:`molrs.io`. After read, molpy may merge split multi-
columns (``CS_1``+``CS_2``→``CS``), map ``species``→``element``, and fill
``atomic_number`` when missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import molrs.io
from molrs import Element, Frame

from molpy.core.fields import ATOMIC_NUMBER

from .base import DataReader, DataWriter


def _normalize_xyz_frame(frame: Frame) -> Frame:
    """Apply molpy column conventions on a molrs XYZ Frame (in place)."""
    for block_name in list(frame.keys()):
        block = frame[block_name]
        keys = list(block.keys())
        merged_pairs: list[tuple[str, str, str]] = []
        for key in keys:
            if key.endswith("_1") and key[:-2] + "_2" in keys:
                base = key[:-2]
                merged_pairs.append((base, key, base + "_2"))
        for base, k1, k2 in merged_pairs:
            block[base] = np.column_stack(
                [np.asarray(block[k1]), np.asarray(block[k2])]
            )
            del block[k1]
            del block[k2]
        if "species" in block and "element" not in block:
            block["element"] = np.asarray(block["species"])
        if "element" in block and ATOMIC_NUMBER not in block:
            z_list = [Element.get_atomic_number(str(s)) for s in block["element"]]
            block[ATOMIC_NUMBER] = np.array(z_list, dtype=np.int64)
    return frame


class XYZReader(DataReader):
    """Read XYZ via molrs + :func:`_normalize_xyz_frame`."""

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        del frame
        return _normalize_xyz_frame(molrs.io.read_xyz(str(self._path)))


class XYZWriter(DataWriter):
    """Write XYZ via molrs."""

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_xyz(str(self._path), frame)
