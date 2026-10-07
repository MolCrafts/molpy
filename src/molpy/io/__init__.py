"""File I/O — :mod:`molrs.io`, mirrored by identity.

There is no package-root ``mp.read_*`` / ``mp.write_*``. Every name here is
the molrs object (``mp.io.read_gro is molrs.io.read_gro``): the structure and
trajectory formats (PDB, XYZ, LAMMPS data and dumps, GROMACS, AMBER, MOL2,
XSF, Cube, CHGCAR, DCD/TRR/XTC, …), the format-picking :func:`read_frame` /
:func:`write_frame`, LAMMPS logs, the ``fix bond/react`` writers, ``*.mrec``
records (``mp.io.mrec`` is :mod:`molrs.io.mrec`: ``read`` / ``write``,
``read_trajectory``, ``FrameSequence``, …), and the SMILES / CGsmiles front
ends (``SmilesIR``, ``CGSmilesIR``, ``SmilesError``). Force-field file formats
are :mod:`molpy.ff.forcefield`'s — a whole AMBER or GROMACS system (force field
and typed frame) is ``mp.ff.forcefield.read_amber_prmtop_system`` /
``read_gromacs_system``.

Names come in pairs, as in the native core: a format ``X`` holding one frame is
read by ``read_X`` and written by ``write_X``; a sequence of frames by
``read_X_trajectory`` / ``write_X_trajectory``.

Basic usage::

    import molpy as mp

    frame = mp.io.read_pdb("structure.pdb")
    frame = mp.io.read_lammps_data("data.lammps", atom_style="full")
    traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
    mol = mp.io.SmilesIR("CCO").to_atomistic()
"""

from molrs.io import *  # noqa: F403
from molrs.io import __all__ as __all__
