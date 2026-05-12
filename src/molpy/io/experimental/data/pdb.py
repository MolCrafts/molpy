"""Experimental PDB file I/O — molrs-backed reader, stable writer."""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.frame import Frame

from . import DataReader, DataWriter


def _to_molpy_frame(molrs_frame: Frame) -> Frame:
    """Copy molrs Frame to molpy Frame, normalizing list values to ndarrays.

    Compat shim — remove once molrs produces molpy-native Block/Frames.
    """
    from molpy.core.box import Box as MolpyBox
    from molpy.core.frame import Block, Frame as MolpyFrame

    new_frame = MolpyFrame()
    if hasattr(molrs_frame, "meta") and molrs_frame.meta:
        new_frame.metadata.update(molrs_frame.meta)
    if molrs_frame.box is not None:
        molrs_box = molrs_frame.box
        new_frame.box = MolpyBox(
            matrix=molrs_box.matrix,
            origin=molrs_box.origin,
            pbc=molrs_box.pbc,
        )

    for block_name in molrs_frame.keys():
        block = molrs_frame[block_name]
        new_block = Block()
        for key in block.keys():
            val = block[key]
            new_block[key] = np.array(val) if isinstance(val, list) else val

        # molrs PDB reader preserves all CONECT records (both directions);
        # molpy deduplicates.  Deduplicate here for output parity.
        if block_name == "bonds" and "atomi" in new_block and "atomj" in new_block:
            atomi = np.asarray(new_block["atomi"])
            atomj = np.asarray(new_block["atomj"])
            pairs = set()
            keep = []
            for idx in range(len(atomi)):
                pair = (
                    int(atomi[idx]),
                    int(atomj[idx]),
                )
                canonical = pair if pair[0] < pair[1] else (pair[1], pair[0])
                if canonical not in pairs:
                    pairs.add(canonical)
                    keep.append(idx)
            if len(keep) < len(atomi):
                new_block["atomi"] = atomi[keep]
                new_block["atomj"] = atomj[keep]

        new_frame[block_name] = new_block

    return new_frame


class PDBReader(DataReader):
    """Experimental. Read PDB files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_pdb` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        result = molrs.io.read_pdb(self._path, frame=frame)
        return _to_molpy_frame(result)


class PDBWriter(DataWriter):
    """Experimental. Write PDB files.

    Currently delegates to the stable :class:`molpy.io.data.pdb.PDBWriter`.
    Will switch to ``molrs.io.write_pdb`` once molrs gains Python-side
    Frame construction APIs.

    .. deprecated::
        Use :func:`molpy.io.write_pdb` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        from molpy.io.data.pdb import PDBWriter as _StablePDBWriter

        _StablePDBWriter(self._path).write(frame)
