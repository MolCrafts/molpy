"""The force-field data model — :mod:`molrs.ff.forcefield`, by identity.

:class:`ForceField` with its ``Style`` / ``ForceFieldType`` handles
(``BondStyle``, ``PairType``, …). ``mp.ff.forcefield.X is
molrs.ff.forcefield.X`` for every name here. The files that map each engine's
force field onto it (LAMMPS ``*.ff`` and data-file ``* Coeffs``, GROMACS
topologies, AMBER prmtop / frcmod, OpenMM XML, CMAP) are read and written by
:mod:`molpy.io`'s format modules (``mp.io.lammps.LammpsForcefieldReader``,
``mp.io.amber.AmberPrmtopForcefieldReader``,
``mp.io.gromacs.GromacsTopForcefieldWriter``, …); its
``*.mrec`` section is ``mp.io.mrec.ForceFieldSection.from_forcefield(ff)``.
"""

from molrs.ff.forcefield import *  # noqa: F403
from molrs.ff.forcefield import __all__ as _native

__all__ = [*_native]
