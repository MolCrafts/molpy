"""Force-field typification — :mod:`molrs.ff.typifier`, plus the AmberTools typifiers.

The native typifiers are identity re-exports
(``mp.ff.typifier.OplsAaTypifier is molrs.ff.typifier.OplsAaTypifier``).
``Typifier`` is the base every typifier extends: a subclass implements
``assign`` (and optionally ``source_forcefield``) and the base owns ``typify``
and the accumulated ``forcefield()``.
:func:`assign_cmaps` builds a typed frame's ``cmaps`` block.

GAFF / GAFF2 is typed natively: ``AtdTypifier`` (antechamber's atom types)
then ``GaffTypifier`` (the bonded terms), each taking molrs's required
``parameter_set``; AM1-BCC charges come from
:class:`molpy.ff.charge.Am1BccModel` (sqm, then the native correction).
molpy adds :class:`AntechamberTypifier` and :class:`TleapTypifier`, which
run the whole AmberTools pipeline (:mod:`molpy.wrapper`) and read tleap's
prmtop back, for a force field that must merge with an
:class:`~molpy.builder.AmberPolymerBuilder` chain's. They take the same
required ``parameter_set``.
"""

from molrs.ff.typifier import *  # noqa: F403
from molrs.ff.typifier import __all__ as _native

from molpy.ff._ambertools import AntechamberTypifier, TleapTypifier

__all__ = [*_native, "AntechamberTypifier", "TleapTypifier"]
