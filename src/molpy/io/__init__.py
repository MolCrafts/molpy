"""File I/O — :mod:`molrs.io`, mirrored by identity.

There is no package-root ``mp.read_*`` / ``mp.write_*``: every file reader and
writer is here, and every name at this level is the molrs function
(``mp.io.read_gro is molrs.io.read_gro``). A factory that reads or writes a
format has one of two shapes, as in molrs:

* a function at this level, ``read_<fmt>[_<what>]`` /
  ``write_<fmt>[_<what>]`` — structure and trajectory files (PDB, XYZ, LAMMPS
  data and dumps, GROMACS, AMBER, MOL2, XSF, Cube, CHGCAR, DCD/TRR/XTC, …, and
  the format-picking :func:`read_frame` / :func:`write_frame`), force-field
  files (:func:`read_lammps_forcefield`, :func:`read_gromacs_system`,
  :func:`read_amber_prmtop_system`, :func:`write_gromacs_system`,
  :func:`read_forcefield_xml`, …), ``*.mrec`` records (:func:`read_mrec`,
  :func:`read_mrec_trajectory`, …), frame bytes (:func:`read_frame_bytes`),
  LAMMPS logs (:func:`read_lammps_log`, :func:`read_lammps_log_str`) and one
  molecule from SMILES (:func:`read_smiles`);
* a class of the format's own submodule, ``mp.io.<fmt>.<Fmt>Reader`` /
  ``<Fmt>Writer``.

A class that belongs to one format lives in that format's submodule, each a
molpy module mirroring the molrs one: :mod:`molpy.io.trajectory`
(``TrajectoryReader``), :mod:`molpy.io.mrec` (``MrecReader``,
``MrecWriter``, …), :mod:`molpy.io.smiles` (``SmilesIR``, ``CGSmilesIR``,
``SmilesError``, …), :mod:`molpy.io.log` (``LammpsLog``, …) and
:mod:`molpy.io.lammps_bond_react` (``BondReactTemplate``). molpy adds its
metric readers there: :class:`molpy.io.log.LammpsLogMetricReader`,
:class:`molpy.io.log.MlpJsonlMetricReader` and
:class:`molpy.io.mrec.MrecMetricReader`.

Names come in pairs, as in the native core: a format ``X`` holding one frame is
read by ``read_X`` and written by ``write_X``; a sequence of frames by
``read_X_trajectory`` / ``write_X_trajectory``.

Basic usage::

    import molpy as mp

    frame = mp.io.read_pdb("structure.pdb")
    frame = mp.io.read_lammps_data("data.lammps", atom_style="full")
    traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
    mol = mp.io.read_smiles("CCO")
"""

from molrs.io import *  # noqa: F403
from molrs.io import __all__ as __all__

from importlib import import_module as _import_module

# The star import bound molrs's per-format modules under these names. Loading
# molpy's mirror of each rebinds the attribute to it (``from . import`` would
# not: it returns an existing attribute without importing the submodule).
for _fmt in ("lammps_bond_react", "log", "mrec", "smiles", "trajectory"):
    _import_module(f"{__name__}.{_fmt}")
del _fmt
