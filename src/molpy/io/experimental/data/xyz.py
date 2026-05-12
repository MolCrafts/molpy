"""Experimental XYZ file I/O — molrs-backed reader."""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.frame import Frame

from . import DataReader


def _to_molpy_frame(molrs_frame: Frame) -> Frame:
    """Copy molrs Frame to molpy Frame, normalizing to molpy conventions.

    Compat shim — remove once molrs produces molpy-native Block/Frames.
    """
    from molpy.core.box import Box as MolpyBox
    from molpy.core.frame import Block, Frame as MolpyFrame
    from molpy.core.element import Element

    new_frame = MolpyFrame()
    if hasattr(molrs_frame, "meta") and molrs_frame.meta:
        new_frame.metadata.update(molrs_frame.meta)
    if molrs_frame.box is not None:
        # molrs Box is the parent — wrap in molpy Box for isinstance checks
        molrs_box = molrs_frame.box
        new_frame.box = MolpyBox(
            matrix=molrs_box.matrix,
            origin=molrs_box.origin,
            pbc=molrs_box.pbc,
        )

    for block_name in molrs_frame.keys():
        block = molrs_frame[block_name]
        new_block = Block()

        # Merge molrs-split multi-column fields: CS_1 + CS_2 → CS
        merged_pairs: list[tuple[str, str, str]] = []
        keys = list(block.keys())
        for key in keys:
            if key.endswith("_1") and key[:-2] + "_2" in keys:
                base = key[:-2]
                merged_pairs.append((base, key, base + "_2"))

        for base, k1, k2 in merged_pairs:
            if k1 not in block or k2 not in block:
                continue
            c1 = np.asarray(block[k1])
            c2 = np.asarray(block[k2])
            new_block[base] = np.column_stack([c1, c2])

        for key in block.keys():
            if key in {k2 for _, _, k2 in merged_pairs}:
                continue  # consumed by merge
            if key in {k1 for _, k1, _ in merged_pairs}:
                continue  # consumed by merge
            val = block[key]
            if isinstance(val, list):
                val = np.array(val)
            new_block[key] = val

        # Detect extended XYZ pos column → split into x, y, z
        if "pos" in new_block and "x" not in new_block:
            pos = new_block["pos"]
            if pos.ndim == 2 and pos.shape[1] == 3:
                new_block["x"] = pos[:, 0]
                new_block["y"] = pos[:, 1]
                new_block["z"] = pos[:, 2]

        # Map species → element
        if "species" in new_block and "element" not in new_block:
            new_block["element"] = new_block["species"]

        # Add atomic numbers if missing
        if "element" in new_block and "number" not in new_block:
            symbols = new_block["element"]
            z_list = [Element.get_atomic_number(str(s)) for s in symbols]
            new_block["number"] = np.array(z_list, dtype=np.int64)

        new_frame[block_name] = new_block

    return new_frame


class XYZReader(DataReader):
    """Experimental. Read XYZ files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_xyz` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        result = molrs.io.read_xyz(self._path)
        return _to_molpy_frame(result)
