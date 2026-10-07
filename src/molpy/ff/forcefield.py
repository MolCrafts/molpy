"""The force-field data model — :mod:`molrs.ff.forcefield`, by identity.

:class:`ForceField` with its ``Style`` / ``Type`` handles (``BondStyle``,
``PairType``, …). ``mp.ff.forcefield.X is molrs.ff.forcefield.X`` for every
name here. The files that map each engine's force field onto it (LAMMPS
``*.ff`` and data-file ``* Coeffs``, GROMACS topologies, AMBER prmtop /
frcmod, OpenMM XML, CMAP) are read and written by :mod:`molpy.io`
(``mp.io.read_lammps_forcefield``, ``mp.io.read_amber_prmtop_system``,
``mp.io.write_gromacs_system``, …).
"""

from molrs.ff.forcefield import *  # noqa: F403
from molrs.ff.forcefield import __all__ as __all__
