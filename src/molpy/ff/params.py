"""Parameter tables — :mod:`molrs.ff.params`, by identity.

The CL&Pol tables molrs ships: the Drude polarizabilities
(:func:`clpol_polarizability`: the shipped ``alpha.ff``, or one read from a
path) and the fragment-scaling table (:func:`clpol_fragment_scaling`, read by
``mp.ff.clpol_scaling.scale_lj``). AMBER's 1-4 divisors are unit facts:
``mp.core.constants.AMBER_SCEE`` / ``AMBER_SCNB``.
"""

from molrs.ff.params import *  # noqa: F403
from molrs.ff.params import __all__ as __all__
