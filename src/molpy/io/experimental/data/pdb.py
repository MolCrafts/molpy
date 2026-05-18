"""Experimental PDB file I/O — molrs-backed reader, stable writer."""

from pathlib import Path

import numpy as np

import molrs.io

from molpy.core.frame import Frame

from . import DataReader, DataWriter


class PDBReader(DataReader):
    """Experimental. Read PDB files via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_pdb` for the stable implementation.
    """

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def read(self) -> Frame:
        raw = molrs.io.read_pdb(self._path)
        molpy_frame = Frame.from_dict(raw)

        # molrs PDB reader preserves all CONECT records (both directions);
        # molpy deduplicates. Deduplicate here for output parity.
        bonds = molpy_frame["bonds"] if "bonds" in molpy_frame else None
        if bonds is not None and "atomi" in bonds and "atomj" in bonds:
            atomi = np.asarray(bonds["atomi"])
            atomj = np.asarray(bonds["atomj"])
            pairs = set()
            keep = []
            for idx in range(len(atomi)):
                pair = (int(atomi[idx]), int(atomj[idx]))
                canonical = pair if pair[0] < pair[1] else (pair[1], pair[0])
                if canonical not in pairs:
                    pairs.add(canonical)
                    keep.append(idx)
            if len(keep) < len(atomi):
                # Replace entire block to keep column lengths consistent
                from molpy.core.frame import Block

                new_bonds = Block()
                # Copy all columns, applying dedup to atomi/atomj
                for col in bonds.keys():
                    if col == "atomi":
                        new_bonds[col] = atomi[keep]
                    elif col == "atomj":
                        new_bonds[col] = atomj[keep]
                    else:
                        new_bonds[col] = np.asarray(bonds[col])[keep]
                molpy_frame["bonds"] = new_bonds

        return molpy_frame


class PDBWriter(DataWriter):
    """Experimental. Write PDB files via molrs backend."""

    def __init__(self, path: str | Path, **kwargs: object) -> None:
        super().__init__(Path(path), **kwargs)

    def write(self, frame: Frame) -> None:
        molrs.io.write_pdb(str(self._path), frame.to_molrs())
