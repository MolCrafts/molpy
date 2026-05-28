"""Experimental XYZ file I/O — molrs-backed reader and writer."""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.frame import Frame

from . import DataReader, DataWriter


class XYZReader(DataReader):
    """Experimental. Read XYZ files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_xyz` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self) -> Frame:
        raw = molrs.io.read_xyz(self._path)
        molpy_frame = Frame.from_dict(raw)

        # Post-process blocks to match molpy conventions
        for block_name in list(molpy_frame.keys()):
            block = molpy_frame[block_name]
            keys = list(block.keys())

            # Merge molrs-split multi-column fields: CS_1 + CS_2 → CS
            merged_pairs: list[tuple[str, str, str]] = []
            for key in keys:
                if key.endswith("_1") and key[:-2] + "_2" in keys:
                    base = key[:-2]
                    merged_pairs.append((base, key, base + "_2"))

            for base, k1, k2 in merged_pairs:
                c1 = np.asarray(block[k1])
                c2 = np.asarray(block[k2])
                block[base] = np.column_stack([c1, c2])
                del block[k1]
                del block[k2]

            # Canonicalize extended-XYZ `species` to `element` (plain XYZ
            # already emits `element` natively in molrs).
            if "species" in block and "element" not in block:
                block["element"] = np.asarray(block["species"])

            # Add atomic numbers if missing
            if "element" in block and "number" not in block:
                from molpy.core.element import Element

                symbols = block["element"]
                z_list = [Element.get_atomic_number(str(s)) for s in symbols]
                block["number"] = np.array(z_list, dtype=np.int64)

        return molpy_frame


class XYZWriter(DataWriter):
    """Experimental. Write XYZ files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.write_xyz` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_xyz(str(self._path), frame)
