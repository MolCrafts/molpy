"""Evaluable force terms — :mod:`molrs.ff.potential`, by identity.

:class:`Potentials` evaluates kernels together (``calc_energy_forces``);
:class:`WeightedTerms` is kernels each with its special-bonds weights, for a
neighbour-driven integrator; :class:`PairLjCut` is the one-type ``lj/cut``
kernel the MD integrators feed; :class:`Potential` is the protocol every
Python force provider satisfies (``calc_energy_forces(pos) -> (energy,
forces)``). A force field compiles into them through :mod:`molpy.ff.compile`.

``mp.ff.potential.X is molrs.ff.potential.X`` for every name here.
"""

from molrs.ff.potential import *  # noqa: F403
from molrs.ff.potential import __all__ as __all__
