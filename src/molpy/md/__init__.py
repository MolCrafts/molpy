"""In-process MD — :mod:`molrs.md`, mirrored by identity.

The integrators (``VelocityVerlet``, ``Langevin``), ``MaxwellBoltzmann``
velocities, ``MDState`` and the ``MD`` driver; ``mp.md.MD is molrs.md.MD``.
MD defines no potential: the kernels, ``Potentials`` and the ``Potential``
protocol are :mod:`molpy.ff.potential`.
"""

from molrs.md import *  # noqa: F403
from molrs.md import __all__ as __all__
