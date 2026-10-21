"""File I/O — ``molrs::io``. Every file reader and writer is here, one
module per file format.

Every format has one door per direction:

* ``read_<fmt>(source)`` reads a file's first (or only) frame. ``source`` is
  an ``os.PathLike`` (a file; ``.gz`` is decompressed), ``bytes`` (a whole
  file in memory) or a :class:`Text` (a whole file's text). A plain ``str``
  is refused with ``TypeError``: it is ambiguous between a path and text, so
  the caller says which — ``pathlib.Path(s)`` or ``molrs.io.Text(s)``.
* ``write_<fmt>(sink, frame)`` writes one frame as a whole file. ``sink`` is
  an ``os.PathLike`` (created or truncated) or a ``bytearray`` (cleared, then
  holding exactly the file's bytes).
* A multi-frame format's classes live in its submodule:
  ``molrs.io.<fmt>.<Fmt>Reader(source)`` reads every frame lazily (``len``,
  ``reader[i]`` / ``reader[a:b]``, iteration) and
  ``molrs.io.<fmt>.<Fmt>Writer(sink)`` writes one frame per ``write(frame)``
  (``close()``, or use it as a context manager) — :mod:`~molrs.io.pdb`,
  :mod:`~molrs.io.xyz`, :mod:`~molrs.io.gro`, :mod:`~molrs.io.mol2`,
  :mod:`~molrs.io.cif`, :mod:`~molrs.io.sdf` (reader only),
  :mod:`~molrs.io.lammps` (dumps), :mod:`~molrs.io.dcd`,
  :mod:`~molrs.io.trr`, :mod:`~molrs.io.xtc`, and :mod:`~molrs.io.cube`
  (reader only, cubes back to back).

No door picks a format for the caller: every door names its format. A
format's extra inputs are parameters of its one door
(``read_lammps_data(source, atom_style=None)``,
``read_inpcrd(source, frame=None)``, ``write_lammps_dump(sink, frame,
columns=None)``, ``read_lammps_log(source, style="default")``, the CSV
delimiter and header, …).

Every structure reader emits the project-wide canonical column names
(:mod:`molrs.core.keys`: ``element``, ``res_id``, ``charge``, ``mol_id``,
…) — the Rust readers map a format's own spelling at the boundary — and every
writer takes them.

Force-field files map onto :class:`molrs.ff.forcefield.ForceField` through a
class per format, its settings the constructor's keywords:
``reader.read(source) -> ForceField`` and ``writer.write(sink, forcefield)``,
plus what else the format reads or writes as methods of the same class —
:class:`molrs.io.lammps.LammpsForcefieldReader` /
:class:`~molrs.io.lammps.LammpsForcefieldWriter` (``read_data_coeffs``,
``read_cmap``, ``write_data_coeffs``, ``write_cmap``),
:class:`molrs.io.gromacs.GromacsTopForcefieldReader` /
:class:`~molrs.io.gromacs.GromacsTopForcefieldWriter` (``read_system``,
``write_system``), :class:`molrs.io.amber.AmberPrmtopForcefieldReader`
(``read_system``) / :class:`~molrs.io.amber.AmberFrcmodWriter`,
:class:`molrs.io.openmm_xml.OpenmmXmlReader` /
:class:`~molrs.io.openmm_xml.OpenmmXmlWriter`, and molrs's own force-field
XML, :class:`molrs.io.xml.XmlForcefieldReader` /
:class:`~molrs.io.xml.XmlForcefieldWriter`. Each reader
maps a format onto the force-field IR (adopts the LAMMPS standard), units
and factors included; each writer is the inverse.

A ``*.mrec`` scientific record is written by :func:`write_mrec` and read by
:class:`molrs.io.mrec.MrecFile` (per section) and
:class:`~molrs.io.mrec.MrecReader` (its trajectory, frame by frame). Those
need the ``fs`` feature (the zarr store): a Pyodide build omits them, the
same way :class:`molrs.stream.Publisher` is absent, and ``import molrs.io``
still succeeds.

:func:`read_wire_frame` / :func:`write_wire_frame` read and write a frame in
the wire encodings (``"msgpack"``, ``"json"``) a
:class:`molrs.stream.Publisher` streams.

SMILES and CGsmiles are :mod:`molrs.io.smiles`
(``read_smiles(text).to_atomistic()``,
``write_smiles(mol)``) and :mod:`molrs.io.cgsmiles`
(``read_cgsmiles(text).to_atomistic()``).
"""

from .._native import (
    Text,
    read_ac,
    read_chgcar,
    read_cif,
    read_clpol_alpha,
    read_csv_block,
    read_cube,
    read_dcd,
    read_gro,
    read_inpcrd,
    read_lammps_data,
    read_lammps_dump,
    read_lammps_log,
    read_lammps_molecule,
    read_lammps_molecule_json,
    read_mol2,
    read_pdb,
    read_poscar,
    read_prep,
    read_prmtop,
    read_sdf,
    read_stl,
    read_trr,
    read_wire_frame,
    read_xsf,
    read_xtc,
    read_xyz,
    write_cif,
    write_csv_block,
    write_cube,
    write_dcd,
    write_gro,
    write_lammps_bond_react_map,
    write_lammps_bond_react_system,
    write_lammps_data,
    write_lammps_dump,
    write_lammps_dump_local,
    write_lammps_molecule,
    write_lammps_molecule_json,
    write_mol2,
    write_pdb,
    write_poscar,
    write_prep,
    write_trr,
    write_wire_frame,
    write_xsf,
    write_xtc,
    write_xyz,
)
from . import (
    amber,
    cgsmiles,
    cif,
    cube,
    dcd,
    gro,
    gromacs,
    lammps,
    mol2,
    openmm_xml,
    pdb,
    sdf,
    smiles,
    trr,
    xml,
    xtc,
    xyz,
)

__all__ = [
    "Text",
    "amber",
    "cgsmiles",
    "cif",
    "cube",
    "dcd",
    "gro",
    "gromacs",
    "lammps",
    "mol2",
    "openmm_xml",
    "pdb",
    "read_ac",
    "read_chgcar",
    "read_cif",
    "read_clpol_alpha",
    "read_csv_block",
    "read_cube",
    "read_dcd",
    "read_gro",
    "read_inpcrd",
    "read_lammps_data",
    "read_lammps_dump",
    "read_lammps_log",
    "read_lammps_molecule",
    "read_lammps_molecule_json",
    "read_mol2",
    "read_pdb",
    "read_poscar",
    "read_prep",
    "read_prmtop",
    "read_sdf",
    "read_stl",
    "read_trr",
    "read_wire_frame",
    "read_xsf",
    "read_xtc",
    "read_xyz",
    "sdf",
    "smiles",
    "trr",
    "write_cif",
    "write_csv_block",
    "write_cube",
    "write_dcd",
    "write_gro",
    "write_lammps_bond_react_map",
    "write_lammps_bond_react_system",
    "write_lammps_data",
    "write_lammps_dump",
    "write_lammps_dump_local",
    "write_lammps_molecule",
    "write_lammps_molecule_json",
    "write_mol2",
    "write_pdb",
    "write_poscar",
    "write_prep",
    "write_trr",
    "write_wire_frame",
    "write_xsf",
    "write_xtc",
    "write_xyz",
    "xml",
    "xtc",
    "xyz",
]

# mrec is the zarr record store (`fs` / `molrs/filesystem`). A Pyodide build
# omits it. Absent, not an import failure of this module — the same rule as
# `molrs.stream.Publisher`.
try:
    from .._native import write_mrec
    from . import mrec as mrec
except ImportError:
    pass
else:
    __all__ += ["mrec", "write_mrec"]
