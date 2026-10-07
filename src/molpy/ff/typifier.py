"""Force-field typification — :mod:`molrs.ff.typifier`, plus the AmberTools typifiers.

The native typifiers are identity re-exports
(``mp.ff.typifier.OplsAaTypifier is molrs.ff.typifier.OplsAaTypifier``).
``Typifier`` is the base every typifier extends: a subclass implements
``match`` and the base owns ``typify`` and the accumulated ``forcefield()``.
:func:`assign_cmaps` builds a typed frame's ``cmaps`` block.

molpy adds :class:`AntechamberTypifier` and :class:`TleapTypifier`, which type
through the AmberTools executables (:mod:`molpy.wrapper`).
"""

from molrs.ff.typifier import *  # noqa: F403
from molrs.ff.typifier import __all__ as _native

from ._ambertools import AntechamberTypifier, TleapTypifier

__all__ = [*_native, "AntechamberTypifier", "TleapTypifier"]
