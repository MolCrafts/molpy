"""File I/O — :mod:`molrs.io`, mirrored by identity.

There is no package-root ``mp.read_*`` / ``mp.write_*``: every file reader and
writer is here, and every name at this level is the molrs function
(``mp.io.read_gro is molrs.io.read_gro``). A factory that reads or writes a
format has one of two shapes, as in molrs:

* a function at this level, named after its format —
  ``read_<fmt>[_<what>]`` / ``write_<fmt>[_<what>]`` for a path,
  ``read_<fmt>_str`` / ``write_<fmt>_str`` for text in memory and
  ``read_<fmt>_bytes`` / ``write_<fmt>_bytes`` for bytes: structure and
  trajectory files (PDB, XYZ, LAMMPS data and dumps, GROMACS, AMBER, MOL2,
  CIF, XSF, Cube, VASP, DCD/TRR/XTC, …), force-field files
  (:func:`read_lammps_forcefield`, :func:`read_gromacs_top_system`,
  :func:`read_amber_prmtop_system`, :func:`read_openmm_xml_forcefield`, …),
  ``*.mrec`` records (:func:`read_mrec_frame`, :func:`read_mrec_trajectory`,
  …), wire-encoded frames (:func:`read_msgpack_frame_bytes`), LAMMPS logs
  (:func:`read_lammps_log`, :func:`read_lammps_log_str`) and line notations
  (:func:`read_smiles_str`, :func:`write_smiles_str`,
  :func:`read_cgsmiles_str`). No door picks a format for the caller;
* a class of the format's own submodule, ``mp.io.<fmt>.<Fmt>Reader`` /
  ``<Fmt>Writer``.

A class that belongs to one format lives in that format's submodule, each a
molpy module mirroring the molrs one: :mod:`molpy.io.smiles` (``SmilesIr``,
``SmilesError``, ``BondingDescriptor``), :mod:`molpy.io.cgsmiles`
(``CgSmilesIr`` and its records), :mod:`molpy.io.lammps`
(``LammpsDumpReader``, ``BondReactTemplate``, ``LammpsLog``, …),
:mod:`molpy.io.mrec` (``MrecReader``, ``MrecWriter``, ``ForceFieldSection``,
…) and the lazy trajectory readers of :mod:`molpy.io.pdb`,
:mod:`molpy.io.xyz`, :mod:`molpy.io.gro`, :mod:`molpy.io.dcd`,
:mod:`molpy.io.trr` and :mod:`molpy.io.xtc`. molpy adds its metric readers
there — :class:`molpy.io.lammps.LammpsLogMetricReader` and
:class:`molpy.io.mrec.MrecMetricReader` — and one format module of its own,
:mod:`molpy.io.mlp_jsonl` (:class:`~molpy.io.mlp_jsonl.MlpJsonlMetricReader`).

Names come in pairs, as in the native core: a format ``X`` holding one frame is
read by ``read_X`` and written by ``write_X``; a sequence of frames by
``read_X_trajectory`` (the format's lazy reader) / ``write_X_trajectory``.

Basic usage::

    import molpy as mp

    frame = mp.io.read_pdb("structure.pdb")
    frame = mp.io.read_lammps_data("data.lammps", atom_style="full")
    traj = mp.io.read_lammps_dump_trajectory("dump.lammpstrj")
    mol = mp.io.read_smiles_str("CCO")
"""

from molrs.io import *  # noqa: F403
from molrs.io import __all__ as _native

from importlib import import_module as _import_module

#: molrs's per-format submodules, each mirrored by a molpy module of its name.
_FORMATS = (
    "cgsmiles",
    "dcd",
    "gro",
    "lammps",
    "mrec",
    "pdb",
    "smiles",
    "trr",
    "xtc",
    "xyz",
)

# The star import bound molrs's per-format modules under these names. Loading
# molpy's mirror of each rebinds the attribute to it (``from . import`` would
# not: it returns an existing attribute without importing the submodule).
for _module in [f"{__name__}.{fmt}" for fmt in _FORMATS]:
    _import_module(_module)

from . import mlp_jsonl  # molpy's own format module

__all__ = [*_native, "mlp_jsonl"]
