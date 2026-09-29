"""The ``mp.io`` readers molpy owns.

Every other reader on :mod:`molpy.io` is the native one, re-exported by
identity. The functions here add something the native reader does not do:
merge coordinates into an existing frame, pair a prmtop with its inpcrd,
canonicalize a format molpy names itself, join split XYZ property columns,
or refuse a multi-component SMILES string.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import molrs.ff
import molrs.io
from molrs import Atomistic, Element, Frame

from molpy.core.fields import ATOMIC_NUMBER, CHARGE, FieldFormatter

PathLike = str | Path


class _AcFieldFormatter(FieldFormatter):
    """Antechamber ``.ac`` column names → canonical names."""

    _field_formatters = {"q": CHARGE}


_AC_FORMATTER = _AcFieldFormatter()


def read_amber_ac(file: PathLike) -> Frame:
    """Read an Antechamber ``.ac`` file, with ``q`` renamed to ``charge``."""
    frame = molrs.io.read_ac(file)
    _AC_FORMATTER.canonicalize_frame(frame)
    return frame


def read_amber_inpcrd(file: PathLike, frame: Frame | None = None) -> Frame:
    """Read an AMBER ASCII ``*.inpcrd`` / restart file.

    Args:
        file: Path to the inpcrd file.
        frame: Optional frame whose ``atoms`` block receives the coordinates
            (and velocities, if present) in place; its other columns, and its
            meta keys the file does not set, are kept. Without a frame, or a
            frame with no ``atoms`` block, the native frame is returned.

    Returns:
        The frame holding the coordinates.

    Raises:
        ValueError: If ``frame`` has a different atom count than the file.
    """
    loaded = molrs.io.read_amber_inpcrd(file)
    if frame is None or "atoms" not in frame:
        return loaded

    atoms = frame["atoms"]
    src = loaded["atoms"]
    if atoms.nrows != src.nrows:
        raise ValueError(
            f"atoms block has {atoms.nrows} rows, but inpcrd has {src.nrows}"
        )
    for column in ("x", "y", "z", "vel"):
        if column in src:
            atoms[column] = src[column]
    frame.box = loaded.box
    frame.meta = {**frame.meta, **loaded.meta}
    return frame


def read_amber(
    prmtop: PathLike, inpcrd: PathLike | None = None
) -> tuple[Frame, molrs.ff.ForceField]:
    """Read an AMBER prmtop (structure and force field), and optionally its inpcrd.

    Args:
        prmtop: Path to a ``.prmtop`` / ``.parm7`` file.
        inpcrd: Optional coordinate file merged into the structure frame.

    Returns:
        ``(frame, forcefield)``.
    """
    frame = molrs.io.read_amber_prmtop(prmtop)
    forcefield = molrs.ff.read_amber_prmtop_ff(prmtop)
    if inpcrd is not None:
        frame = read_amber_inpcrd(inpcrd, frame)
    return frame, forcefield


def read_xyz(file: PathLike) -> Frame:
    """Read an XYZ file with molpy's column conventions.

    On top of the native reader: an ``n``-wide property the native reader
    splits into ``base_1`` … ``base_n`` is joined back into one ``(N, n)``
    column ``base``, ``species`` becomes ``element`` when there is none, and
    ``atomic_number`` is filled from ``element`` when missing.
    """
    frame = molrs.io.read_xyz(file)
    for block_name in list(frame.keys()):
        block = frame[block_name]
        keys = set(block.keys())
        for key in sorted(keys):
            if not key.endswith("_1"):
                continue
            base = key[:-2]
            parts = [key]
            while f"{base}_{len(parts) + 1}" in keys:
                parts.append(f"{base}_{len(parts) + 1}")
            if len(parts) > 1:
                block[base] = np.column_stack([np.asarray(block[k]) for k in parts])
                for k in parts:
                    del block[k]
        if "species" in block and "element" not in block:
            block["element"] = np.asarray(block["species"])
        if "element" in block and ATOMIC_NUMBER not in block:
            block[ATOMIC_NUMBER] = np.array(
                [Element.get_atomic_number(str(s)) for s in block["element"]],
                dtype=np.int64,
            )
    return frame


def read_smiles(smiles: str) -> Atomistic:
    """Parse a single-component SMILES string into an :class:`Atomistic`.

    Connectivity only: hydrogens implicit in the SMILES are **not** added, and
    no coordinates are generated. Filling open valences is a separate
    perception step (``mp.Perceive().find_hydrogens(mol)``), and 3D embedding a
    separate conformer step.

    Args:
        smiles: A SMILES string naming exactly one connected molecule.

    Returns:
        The parsed graph.

    Raises:
        ValueError: if ``smiles`` is syntactically invalid, or names more than
            one component. A ``'.'``-separated string is a *set* of molecules,
            not a molecule; take them apart with
            ``mp.SmilesIR(smiles).components()``.

    Examples:
        >>> import molpy as mp
        >>> len(list(mp.io.read_smiles("CCO").atoms))
        3
    """
    ir = molrs.io.SmilesIR(smiles)
    if ir.n_components != 1:
        raise ValueError(
            f"read_smiles needs one component, {smiles!r} has "
            f"{ir.n_components}. Use mp.SmilesIR(smiles).components(), "
            "or pass one component at a time."
        )
    return ir.to_atomistic()
