"""Partial-charge models — :mod:`molrs.ff.charge`, plus :class:`Am1BccModel`.

The native models are identity re-exports
(``mp.ff.charge.BccModel is molrs.ff.charge.BccModel``): ``BccModel``
(AM1-BCC / ABCG2 corrections of given AM1 charges), ``MullikenModel`` and
``GasteigerModel`` behind one ``assign(mol)`` call.

molpy adds :class:`Am1BccModel`, the default AM1-BCC route: sqm (through
AmberTools' antechamber) for the AM1 charges, then ``BccModel`` for the
correction, so types and bonded terms stay native.
"""

from molrs.ff.charge import *  # noqa: F403
from molrs.ff.charge import __all__ as _native

from molpy.ff.ambertools import Am1BccModel

__all__ = [*_native, "Am1BccModel"]
