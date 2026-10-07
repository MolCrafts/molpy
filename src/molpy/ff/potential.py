"""Evaluable force terms — :mod:`molrs.ff.potential`, by identity.

:class:`PotentialCompiler` compiles a force field against a typed frame into
:class:`Potentials`; :func:`compile_explicit_terms` builds the kernel of any
style the force-field IR prices over explicit instances (atom indices, one
parameter row per term, as stored: angle values in degrees); :class:`PairLjCut`
is the one-type ``lj/cut`` kernel the MD integrators feed; :class:`WeightedTerms`
is a weighted sum of compiled terms; :class:`Potential` is the protocol every
Python force provider satisfies (``calc_energy_forces(pos) -> (energy,
forces)``)::

    pots = mp.ff.potential.Potentials()
    pots.push(mp.ff.potential.compile_explicit_terms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4))
    energy, forces = pots.calc_energy_forces(pos)

``mp.ff.potential.X is molrs.ff.potential.X`` for every name here.
"""

from molrs.ff.potential import *  # noqa: F403
from molrs.ff.potential import __all__ as __all__
