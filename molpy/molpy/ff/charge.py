"""Partial-charge models — ``molrs::ff::charge``.

Three models behind one ``assign(mol)`` call. :class:`BccModel` (AM1-BCC /
ABCG2) and :class:`MullikenModel` need a QM population as input;
:class:`GasteigerModel` (PEOE) is topological and takes no QM at all — it
accepts the ``qm`` argument and ignores it, which is what keeps the three
interchangeable at a call site.

:class:`Am1BccModel` is the default AM1-BCC route: sqm (through AmberTools'
antechamber) for the AM1 charges, then ``BccModel`` for the correction, so
types and bonded terms stay native.
"""

from molpy._core import (
    BccModel,
    GasteigerModel,
    MullikenModel,
)

from molpy.ff.ambertools import Am1BccModel

__all__ = [
    "BccModel",
    "GasteigerModel",
    "MullikenModel",
    "Am1BccModel",
]
