"""In-process MD — :mod:`molrs.md`, mirrored by identity.

The phase driver ``MD`` and the components it mounts on the LAMMPS phase
list (``VelocityVerlet``, ``FixLangevin``, ``NeighborList``), ``Phase`` and
``MaxwellBoltzmann`` velocities; ``mp.md.MD is molrs.md.MD``. MD defines no
potential: the kernels, ``Potentials`` and
the ``Potential`` protocol are :mod:`molpy.ff.potential`.
"""

from molrs.md import *  # noqa: F403
from molrs.md import __all__ as _native

__all__ = [*_native]
