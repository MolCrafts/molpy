"""File I/O — :mod:`molrs.io`, mirrored by identity, plus two molpy readers.

There is no package-root ``mp.read_*`` / ``mp.write_*``. Every native name of
:mod:`molrs.io` is here as the molrs object (``mp.io.read_gro is
molrs.io.read_gro``): the structure and trajectory formats (PDB, XYZ, LAMMPS
data and dumps, GROMACS, AMBER, MOL2, XSF, Cube, CHGCAR, DCD/TRR/XTC, …), the
format-picking :func:`read_frame` / :func:`write_frame`, LAMMPS logs, the
``fix bond/react`` writers, ``*.mrec`` records (:mod:`molpy.io.mrec` is
:mod:`molrs.io.mrec`), and the SMILES / CGsmiles front ends (``SmilesIR``,
``CGSmilesIR``, ``SmilesError``). Force-field file formats are
:mod:`molpy.ff.forcefield`'s.

molpy adds two readers of its own (:mod:`molpy.io._readers`):
:func:`read_amber` (a prmtop's structure and force field, plus an optional
inpcrd) and :func:`read_smiles` (one connected molecule, refusing a
``'.'``-separated set).

Names come in pairs, as in the native core: a format ``X`` holding one frame is
read by ``read_X`` and written by ``write_X``; a sequence of frames by
``read_X_trajectory`` / ``write_X_trajectory``.

Basic usage::

    import molpy as mp

    frame = mp.io.read_pdb("structure.pdb")
    frame = mp.io.read_lammps_data("data.lammps", atom_style="full")
    traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
"""

from molrs.io import *  # noqa: F403
from molrs.io import __all__ as _native

from ._readers import read_amber, read_smiles

__all__ = [*_native, "read_amber", "read_smiles"]
