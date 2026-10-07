"""CL&Pol fragment scaling of Lennard-Jones parameters — :mod:`molrs.ff.clpol_scaling`, by identity.

:func:`scale_lj` returns a copy of a force field whose LJ well depths (and,
optionally, diameters) are scaled per fragment pair by :func:`compute_k_ij`;
:class:`FragmentScaling` is one fragment's charge, dipole and polarizability.
The table molrs ships is ``mp.ff.params.clpol_fragment_scaling()``.
"""

from molrs.ff.clpol_scaling import *  # noqa: F403
from molrs.ff.clpol_scaling import __all__ as __all__
