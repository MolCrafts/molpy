"""CL&Pol fragment scaling of Lennard-Jones parameters — :mod:`molrs.ff.scale_lj`, by identity.

:func:`scale_lj` returns a copy of a force field whose LJ well depths (and,
optionally, diameters) are scaled per fragment pair by :func:`compute_k_ij`;
:class:`FragmentScaling` is one fragment's charge, dipole and polarizability,
:func:`fragment_scaling_data` the table molrs ships.
"""

from molrs.ff.scale_lj import *  # noqa: F403
from molrs.ff.scale_lj import __all__ as __all__
