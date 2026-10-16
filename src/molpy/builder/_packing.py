"""A built molecule as a packer takes it: its frame plus the atom roles on it.

A packer (molpack's ``Target``) reads a template :class:`~molpy.Frame` and,
separately, index lists for per-atom settings in that frame (``Target(...)
.with_atom_radius(indices, r)``). The indices are only meaningful against the
frame they were read from, so :class:`PackingTemplate` holds both: the frame of
an assembled :class:`~molpy.Atomistic` and the hydrogens on it.

Which atoms *are* hydrogens to the packer is molpack's own default (element
``H``, the same rule), so the template is not passed to ``with_hydrogens``;
molpack does not expose its hydrogen indices, which is why molpy names them.
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
        >>> target = molpack.Target(template.frame, 10).with_atom_radius(template.hydrogens, 0.2)
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
            f"PackingTemplate(atoms={self.frame['atoms'].n_rows}, "
            f"hydrogens={len(self.hydrogens)})"
        )
