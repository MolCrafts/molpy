"""A force field bound to its kernels — :mod:`molrs.ff.compile`, by identity.

:class:`PotentialCompiler` compiles a force field against a typed frame into
``mp.ff.potential.Potentials`` (``compile``), or into
``mp.ff.potential.WeightedTerms`` for a neighbour-driven integrator
(``compile_typed``); :func:`compile_explicit_terms` builds the kernel of any
style the force-field IR prices over explicit instances (atom indices, one
parameter row per term, as stored: angle values in degrees)::

    pots = mp.ff.potential.Potentials()
    pots.push(mp.ff.compile.compile_explicit_terms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4))
    energy, forces = pots.calc_energy_forces(pos)

``mp.ff.compile.X is molrs.ff.compile.X`` for every name here.
"""

from molrs.ff.compile import *  # noqa: F403
from molrs.ff.compile import __all__ as __all__
