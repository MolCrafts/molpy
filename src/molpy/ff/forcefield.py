"""The force-field container and its file formats — :mod:`molrs.ff.forcefield`, by identity.

:class:`ForceField` with its ``Style`` / ``Type`` handles, and the readers and
writers that map each engine's force-field files onto it (LAMMPS ``*.ff`` and
data-file ``* Coeffs``, GROMACS topologies, AMBER prmtop / frcmod, OpenMM XML,
CMAP). Structure and trajectory formats are :mod:`molpy.io`'s.
``mp.ff.forcefield.X is molrs.ff.forcefield.X`` for every name here.
"""

from molrs.ff.forcefield import *  # noqa: F403
from molrs.ff.forcefield import __all__ as __all__
