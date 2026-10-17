"""File I/O — :mod:`molrs.io`, mirrored by identity.

One-frame formats expose ``read_<fmt>(source)`` and ``write_<fmt>(sink, frame)``
here. A source is a :class:`pathlib.Path`, bytes, or :class:`Text` for literal
text; a bare string is ambiguous and rejected. A sink is a path or bytearray.
Multi-frame readers and writers, and force-field readers and writers, live in
their format modules. For example, ``mp.io.pdb.PdbReader(path)`` iterates PDB
frames and ``mp.io.amber.AmberPrmtopForcefieldReader().read_system(path)``
returns a force field and frame. ``mp.io.smiles.SmilesIr.parse(text)`` handles
SMILES. ``mp.io.mrec.MrecFile(path)`` reads record sections; ``write_mrec``
writes a whole record.

Every native object is re-exported by identity. Molpy adds metric readers in
the ``lammps`` and ``mrec`` modules, plus its own ``mlp_jsonl`` module.
"""

from molrs.io import *  # noqa: F403
from molrs.io import __all__ as _native

from importlib import import_module as _import_module

#: molrs's per-format submodules, each mirrored by a molpy module of its name.
_FORMATS = (
    "amber",
    "cgsmiles",
    "cif",
    "cube",
    "dcd",
    "gro",
    "gromacs",
    "lammps",
    "mol2",
    "mrec",
    "openmm_xml",
    "pdb",
    "sdf",
    "smiles",
    "trr",
    "xml",
    "xtc",
    "xyz",
)

# The star import bound molrs's per-format modules under these names. Loading
# molpy's mirror of each rebinds the attribute to it (``from . import`` would
# not: it returns an existing attribute without importing the submodule).
# A format molrs did not export (mrec, without the fs feature) stays absent.
for _fmt in _FORMATS:
    if _fmt in _native:
        _import_module(f"{__name__}.{_fmt}")

from . import mlp_jsonl  # molpy's own format module

__all__ = [*_native, "mlp_jsonl"]
