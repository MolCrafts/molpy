"""Parameter tables — :mod:`molrs.ff.params`, by identity.

The CL&Pol Drude polarizabilities molrs ships (:func:`clpol_polarizability`:
the shipped ``alpha.ff``, or one read from a path). The fragment-scaling table
is ``mp.ff.clpol_scaling.fragment_table()``. AMBER's 1-4 divisors are unit
facts: ``mp.core.constants.AMBER_SCEE`` / ``AMBER_SCNB``.
"""

from molrs.ff.params import *  # noqa: F403
from molrs.ff.params import __all__ as __all__
