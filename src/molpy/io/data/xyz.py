"""XYZ file I/O — molrs (Rust) backend.

Reader/writer delegate to :mod:`molrs.io`. The reader normalizes molrs output to
molpy conventions (merge split multi-columns, ``species``→``element``, atomic
numbers); the writer ensures separate x/y/z coordinate columns.
"""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.element import Element
from molpy.core.frame import Frame

from .base import DataReader, DataWriter


def _ensure_xyz(frame: Frame) -> Frame:
    """Return a frame whose atoms block has separate x/y/z columns (molrs
    convention), deriving them from a combined ``xyz`` column if needed. Works
    on a copy; never mutates the caller's frame."""
    atoms = frame["atoms"]
    if "x" in atoms or "xyz" not in atoms:
        return frame
    out = frame.copy()
    xyz = np.asarray(out["atoms"]["xyz"])
    out["atoms"]["x"] = xyz[:, 0]
    out["atoms"]["y"] = xyz[:, 1]
    out["atoms"]["z"] = xyz[:, 2]
    return out


class XYZReader(DataReader):
    """Read XYZ files via the molrs Rust backend."""

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        # molrs.io.read_xyz already returns the canonical rich Frame
        # (molpy.Frame is molrs.Frame); no upgrade wrapping needed.
        molpy_frame = molrs.io.read_xyz(self._path)

        for block_name in list(molpy_frame.keys()):
            block = molpy_frame[block_name]
            keys = list(block.keys())

            # Merge molrs-split multi-column fields: CS_1 + CS_2 -> CS
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

            # Extended-XYZ `species` -> canonical `element`
            if "species" in block and "element" not in block:
                block["element"] = np.asarray(block["species"])

            # Atomic numbers if missing
            if "element" in block and "number" not in block:
                z_list = [Element.get_atomic_number(str(s)) for s in block["element"]]
                block["number"] = np.array(z_list, dtype=np.int64)

        return molpy_frame


class XYZWriter(DataWriter):
    """Write XYZ files via the molrs Rust backend."""

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_xyz(str(self._path), _ensure_xyz(frame))
