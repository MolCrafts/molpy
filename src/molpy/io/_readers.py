"""The two ``mp.io`` readers molpy owns.

Every other name on :mod:`molpy.io` is the native one, re-exported by
identity. These two compose native doors into one call: :func:`read_amber`
pairs a prmtop's structure with its force field (and, optionally, its
inpcrd), and :func:`read_smiles` refuses a multi-component SMILES string.
"""

from __future__ import annotations

from os import PathLike

from molrs.ff.forcefield import ForceField, read_amber_prmtop_ff
from molrs.io import SmilesIR, read_amber_inpcrd, read_amber_prmtop
from molrs.store import Frame
from molrs.system import Atomistic

PathInput = str | PathLike[str]


def read_amber(
    prmtop: PathInput, inpcrd: PathInput | None = None
) -> tuple[Frame, ForceField]:
    """Read an AMBER prmtop (structure and force field), and optionally its inpcrd.

    Args:
        prmtop: Path to a ``.prmtop`` / ``.parm7`` file.
        inpcrd: Optional coordinate file merged into the structure frame
            (``mp.io.read_amber_inpcrd(inpcrd, frame)``).

    Returns:
        ``(frame, forcefield)``.
    """
    frame = read_amber_prmtop(prmtop)
    forcefield = read_amber_prmtop_ff(prmtop)
    if inpcrd is not None:
        frame = read_amber_inpcrd(inpcrd, frame)
    return frame, forcefield


def read_smiles(smiles: str) -> Atomistic:
    """Parse a single-component SMILES string into an :class:`~molpy.Atomistic`.

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
            ``mp.io.SmilesIR(smiles).components()``.

    Examples:
        >>> import molpy as mp
        >>> len(list(mp.io.read_smiles("CCO").atoms))
        3
    """
    ir = SmilesIR(smiles)
    if ir.n_components != 1:
        raise ValueError(
            f"read_smiles needs one component, {smiles!r} has "
            f"{ir.n_components}. Use mp.io.SmilesIR(smiles).components(), "
            "or pass one component at a time."
        )
    return ir.to_atomistic()
