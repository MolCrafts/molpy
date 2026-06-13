"""Force-field potential styles — molpy's convenient entry point.

The potential math lives in the molrs Rust extension; this package is the
user-facing facade so callers write ``from molpy.potential import …`` (or
``molpy.potential.bond.BondHarmonicStyle``) and never need to reach into
``molrs`` directly. Every name here re-exports the molrs-backed style class
surfaced through :mod:`molpy.core.forcefield`.

Energy/force evaluation is ``forcefield.to_potentials().calc_energy(frame)`` /
``.calc_forces(frame)``; :class:`Potentials` is the evaluable collection it
returns.
"""

from __future__ import annotations

from molrs import Potentials

from . import angle, bond, dihedral, improper, pair
from .angle import *  # noqa: F401,F403
from .bond import *  # noqa: F401,F403
from .dihedral import *  # noqa: F401,F403
from .improper import *  # noqa: F401,F403
from .pair import *  # noqa: F401,F403

__all__ = [
    "angle",
    "bond",
    "dihedral",
    "improper",
    "pair",
    "Potentials",
    *bond.__all__,
    *angle.__all__,
    *dihedral.__all__,
    *improper.__all__,
    *pair.__all__,
]
