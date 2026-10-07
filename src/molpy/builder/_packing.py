"""A built molecule as a packer takes it: its frame plus the atom roles on it.

A packer (molpack's ``Target``) reads a template :class:`~molpy.Frame` and,
separately, index lists naming atom roles in that frame. The indices are only
meaningful against the frame they were read from, so :class:`PackingTemplate`
holds both: the frame of an assembled :class:`~molpy.Atomistic` and the
hydrogens on it.
"""

from __future__ import annotations

import numpy as np
from molrs.core import Frame
from molrs.core import Atomistic


__all__ = ["PackingTemplate"]


class PackingTemplate:
    """One molecule's packing template: its frame and the hydrogens in it.

    Every index refers to a row of ``frame["atoms"]``; :attr:`frame` is
    ``mol.to_frame()`` taken once at construction, so the template does not
    follow later edits of ``mol``.

    Attributes:
        frame: The molecule as a frame, the one to hand to the packer.
        hydrogens: Atom indices whose element is hydrogen, ascending.

    Examples:
        >>> chain = mp.builder.Assembler(library, mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
        >>> template = mp.builder.PackingTemplate(chain)
        >>> target = molpack.Target(template.frame, 10).with_hydrogens(template.hydrogens)
    """

    def __init__(self, mol: Atomistic) -> None:
        """Take the frame of ``mol`` and name its hydrogens.

        Args:
            mol: A molecule with an ``element`` on every atom.

        Raises:
            ValueError: If the frame has no ``element`` column.
        """
        frame = mol.to_frame()
        atoms = frame["atoms"]
        if "element" not in atoms:
            raise ValueError(
                "PackingTemplate needs an 'element' on every atom to name hydrogens"
            )
        elements = np.asarray(atoms["element"]).astype(str)

        self.frame: Frame = frame
        self.hydrogens: tuple[int, ...] = tuple(
            int(k) for k in np.flatnonzero(elements == "H")
        )

    def __repr__(self) -> str:
        return (
            f"PackingTemplate(atoms={self.frame['atoms'].nrows}, "
            f"hydrogens={len(self.hydrogens)})"
        )
