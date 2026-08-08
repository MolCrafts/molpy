"""PDB file I/O — molrs-backed.

Read: :func:`molrs.io.read_pdb` plus undirected CONECT de-duplication.
Write: :func:`molrs.io.write_pdb` on **canonical** columns. Thin prep only:
inject ``element`` from ``frame.meta['elements']`` when the column is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import molrs.io
from molrs import Block, Frame

from molpy._frame_meta import get_frame_meta

from .base import DataReader, DataWriter


def _dedup_conect_bonds(frame: Frame) -> Frame:
    """Keep one undirected bond per CONECT pair (molrs emits both directions)."""
    if "bonds" not in frame:
        return frame
    bonds = frame["bonds"]
    if "atomi" not in bonds or "atomj" not in bonds:
        return frame
    atomi = np.asarray(bonds["atomi"])
    atomj = np.asarray(bonds["atomj"])
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for idx in range(len(atomi)):
        a, b = int(atomi[idx]), int(atomj[idx])
        canonical = (a, b) if a < b else (b, a)
        if canonical not in seen:
            seen.add(canonical)
            keep.append(idx)
    if len(keep) == len(atomi):
        return frame
    new_bonds = Block()
    for col in bonds.keys():
        new_bonds[col] = np.asarray(bonds[col])[keep]
    frame["bonds"] = new_bonds
    return frame


def _ensure_element_column(frame: Frame) -> Frame:
    """If atoms lack ``element``, build it from meta ``elements`` (space-separated)."""
    atoms = frame["atoms"]
    if "element" in atoms:
        return frame
    elements_str = get_frame_meta(frame, "elements")
    if not isinstance(elements_str, str) or not elements_str.strip():
        return frame
    parts = elements_str.split()
    n = atoms.nrows
    if len(parts) < n:
        parts = parts + ["X"] * (n - len(parts))
    # Rebuild atoms block with element column (Block may not allow in-place add
    # of new string cols in all builds — copy keys).
    cols = {k: np.asarray(atoms[k]) for k in atoms.keys()}
    cols["element"] = np.asarray(parts[:n], dtype="U8")
    out = Frame()
    out["atoms"] = Block(cols)
    if frame.box is not None:
        out.box = frame.box
    out.meta = dict(frame.meta)
    if "bonds" in frame:
        out["bonds"] = frame["bonds"]
    return out


class PDBReader(DataReader):
    """Read a PDB via molrs (first MODEL; use :func:`read_pdb_trajectory` for all)."""

    def __init__(self, file: str | Path, **kwargs: object) -> None:
        super().__init__(Path(file), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        del frame
        return _dedup_conect_bonds(molrs.io.read_pdb(str(self._path)))


class PDBWriter(DataWriter):
    """Write a PDB via molrs (canonical columns)."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path=Path(path))

    def write(self, frame: Frame) -> None:
        atoms = frame["atoms"]
        for field in ("x", "y", "z"):
            if field not in atoms:
                raise ValueError(
                    f"Required field '{field}' is missing in frame['atoms']"
                )
        prepared = _ensure_element_column(frame)
        molrs.io.write_pdb(str(self._path), prepared)
