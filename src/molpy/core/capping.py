"""Valence completion — fill dangling bonds with hydrogens via molrs.

Extracting a subgraph leaves cut atoms under-coordinated.
:func:`complete_valence` returns a copy whose every under-valent atom is capped
with hydrogens (engine: :func:`molrs.add_hydrogens`), so the fragment is a
chemically valid molecule for external tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import molrs

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic


def complete_valence(struct: Atomistic) -> Atomistic:
    """Return a copy of ``struct`` with every dangling valence filled by hydrogen.

    Delegates to :func:`molrs.add_hydrogens` (bond-order + formal-charge valence
    rules; tetrahedral X–H placement when the heavy has coordinates). ``struct``
    is untouched. Completing an already-complete molecule adds no atoms.

    Args:
        struct: the (possibly under-coordinated) atomistic graph to complete.

    Returns:
        A new :class:`~molpy.core.atomistic.Atomistic`.
    """
    from molpy.core.atomistic import Atomistic

    return Atomistic.adopt(molrs.add_hydrogens(struct))
