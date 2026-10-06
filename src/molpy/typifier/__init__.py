"""Force-field typification.

The native typifiers are identity re-exports of :mod:`molrs.ff.typifier`
(``mp.typifier.OPLSAATypifier is molrs.ff.typifier.OPLSAATypifier``), listed one
by one. ``Typifier`` is the base every typifier extends: a subclass implements
``match`` and the base owns ``typify`` and the accumulated ``forcefield()``.

:class:`AntechamberTypifier` and :class:`TLeapTypifier` type through the
AmberTools executables (:mod:`molpy.wrapper`).
"""

from molrs.ff.typifier import (
    AtdTypifier,
    ElementTypifier,
    GaffTypifier,
    Match,
    MMFF94STypifier,
    MMFF94Typifier,
    OPLSAATypifier,
    Typifier,
)

from .ambertools import AntechamberTypifier, TLeapTypifier

__all__ = [
    "AntechamberTypifier",
    "AtdTypifier",
    "ElementTypifier",
    "GaffTypifier",
    "MMFF94STypifier",
    "MMFF94Typifier",
    "Match",
    "OPLSAATypifier",
    "TLeapTypifier",
    "Typifier",
]
