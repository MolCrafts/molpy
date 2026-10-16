"""File I/O — ``molrs::io``. Every file reader and writer is here, one
module per file format.

A factory that reads or writes a format has exactly one shape, the same name
in Rust and Python:

* a function at the top of this module, ``read_<fmt>[_<what>]`` /
  ``write_<fmt>[_<what>]`` — a path; ``read_<fmt>_str`` / ``write_<fmt>_str``
  for text in memory and ``read_<fmt>_bytes`` / ``write_<fmt>_bytes`` for
  bytes; or
* a class of the format's own submodule, ``molrs.io.<fmt>.<Fmt>Reader``
  (:class:`molrs.io.pdb.PdbReader`, :class:`molrs.io.dcd.DcdReader`,
  :class:`molrs.io.mrec.MrecReader`, …).

No door picks a format for the caller: every door names its format.

The in-memory doors are the path doors on memory, the same set in Rust,
Python and JS: every text frame format has ``read_<fmt>_str`` (and
``write_<fmt>_str`` when it has a writer) — ``pdb``, ``xyz``, ``sdf``,
``gro``, ``mol2``, ``cif``, ``xsf``, ``cube``, ``vasp_poscar``,
``vasp_chgcar``, ``amber_ac``, ``amber_inpcrd``, ``amber_prmtop``,
``amber_prep``, ``lammps_data``, ``lammps_dump`` — and the binary
trajectories ``dcd``, ``trr`` and ``xtc`` have ``read_<fmt>_bytes`` /
``write_<fmt>_bytes`` (one frame). ``read_<fmt>_bytes`` also reads one frame
of ``pdb``, ``xyz``, ``sdf``, ``lammps_data`` and ``lammps_dump`` from a byte
window, as the chunked JS streams do, and ``read_stl_bytes`` an STL mesh.

A class that belongs to one format lives in that format's submodule:
:mod:`molrs.io.smiles` (:class:`~molrs.io.smiles.SmilesIr`,
:class:`~molrs.io.smiles.SmilesError`), :mod:`molrs.io.cgsmiles` (the CGsmiles
records), :mod:`molrs.io.lammps` (the dump reader, the log records,
:class:`~molrs.io.lammps.BondReactTemplate`), :mod:`molrs.io.mrec` (the
record store's reader, writer, schema and force-field section), and the lazy
trajectory readers of :mod:`molrs.io.pdb`, :mod:`~molrs.io.xyz`,
:mod:`~molrs.io.gro`, :mod:`~molrs.io.dcd`, :mod:`~molrs.io.trr` and
:mod:`~molrs.io.xtc`.

Every structure reader emits the project-wide canonical column names
(:mod:`molrs.core.keys`: ``element``, ``res_id``, ``charge``, ``mol_id``,
…) — the Rust readers map a format's own spelling (``symbol``, ``resSeq``,
``q``, ``mol``) at the boundary — and every writer takes them.

Names come in pairs. A format ``X`` that holds one frame is read by
``read_X`` and written by ``write_X``; a sequence of frames is read by
``read_X_trajectory`` and written by ``write_X_trajectory`` — ``lammps`` dumps,
``xyz``, ``pdb``, ``gro``, ``dcd``, ``trr`` and ``xtc``. Each
``read_X_trajectory`` returns that format's lazy reader,
``molrs.io.X.<X>Reader`` (one path, or a list of paths whose frames are
concatenated), not a ``list[Frame]``. A format with one direction only
(``read_chgcar``, ``write_lammps_dump_local``, …) has no partner by
nature, not by omission. Family formats carry the family's name:
``read_lammps_data``, ``read_prmtop``, ``read_poscar``,
``read_top_forcefield``. Every path argument takes a ``str`` or any
``os.PathLike``.

Force-field files map onto :class:`molrs.ff.forcefield.ForceField`, the data
model :mod:`molrs.ff.forcefield` owns:

* readers — :func:`read_lammps_forcefield`, :func:`read_lammps_data_coeffs`,
  :func:`read_lammps_data_coeffs_str`,
  :func:`read_lammps_cmap_forcefield`, :func:`read_top_forcefield`,
  :func:`read_top_system`, :func:`read_prmtop_forcefield`,
  :func:`read_prmtop_system`, :func:`read_openmm_forcefield`,
  :func:`read_xml_forcefield`, and the text doors
  :func:`read_lammps_forcefield_str`, :func:`read_openmm_forcefield_str`,
  :func:`read_xml_forcefield_str`
* writers — :func:`write_lammps_forcefield`,
  :func:`write_lammps_forcefield_str`, :func:`write_lammps_data_coeffs_str`,
  :func:`write_lammps_cmap_forcefield`, :func:`write_top_forcefield`,
  :func:`write_top_system`, :func:`write_frcmod`,
  :func:`write_openmm_forcefield`, :func:`write_openmm_system`,
  :func:`write_xml_forcefield`, and the text doors :func:`write_frcmod_str`,
  :func:`write_openmm_forcefield_str`, :func:`write_openmm_system_str`,
  :func:`write_xml_forcefield_str`

Each reader maps a format onto the force-field IR (adopts the LAMMPS standard),
units and factors included; each writer is the inverse.

A ``*.mrec`` scientific record is read and written whole by
:func:`read_mrec_frame` / :func:`write_mrec_frame` (Structure) and their
``_system`` / ``_trajectory`` / ``_forcefield`` partners, plus
:func:`read_mrec_meta`; a run too large for memory goes through
:class:`molrs.io.mrec.MrecReader` / :class:`~molrs.io.mrec.MrecWriter`.
Those doors need the ``fs`` feature (the zarr store). A Pyodide build does
not include them: they are absent, the same way :class:`molrs.stream.Publisher`
is, and ``import molrs.io`` still succeeds. LAMMPS logs are not in that set.
:func:`read_lammps_log` and :func:`read_lammps_log_str` are in every build.

:func:`read_msgpack_frame_bytes` / :func:`write_msgpack_frame_bytes` and
:func:`read_json_frame_str` / :func:`write_json_frame_str` read and write a
frame in the wire encodings a :class:`molrs.stream.Publisher` streams.

:func:`read_smiles_str` reads one molecule from SMILES text — connectivity
only, no implicit hydrogens added, no coordinates; a ``'.'``-separated set is
refused (take it apart with ``SmilesIr(s).components()``) — and
:func:`write_smiles_str` writes one. :func:`read_cgsmiles_str` reads the
molecule a CGsmiles string states.
"""

from .._native import (
    read_ac,
    read_ac_str,
    read_inpcrd,
    read_inpcrd_str,
    read_prep,
    read_prep_str,
    read_prmtop,
    read_prmtop_forcefield,
    read_prmtop_str,
    read_prmtop_system,
    read_cgsmiles_str,
    read_cif,
    read_cif_str,
    read_clpol_alpha,
    read_clpol_alpha_str,
    read_csv_block,
    read_csv_block_str,
    read_cube,
    read_cube_str,
    read_dcd_bytes,
    read_dcd_trajectory,
    read_gro,
    read_gro_str,
    read_gro_trajectory,
    read_top_forcefield,
    read_top_system,
    read_json_frame_str,
    read_lammps_cmap_forcefield,
    read_lammps_data,
    read_lammps_data_bytes,
    read_lammps_data_coeffs,
    read_lammps_data_coeffs_str,
    read_lammps_data_str,
    read_lammps_dump_bytes,
    read_lammps_dump_str,
    read_lammps_dump_trajectory,
    read_lammps_forcefield,
    read_lammps_forcefield_str,
    read_lammps_log,
    read_lammps_log_str,
    read_lammps_molecule,
    read_lammps_molecule_json,
    read_mol2,
    read_mol2_str,
    read_msgpack_frame_bytes,
    read_openmm_forcefield,
    read_openmm_forcefield_str,
    read_pdb,
    read_pdb_bytes,
    read_pdb_str,
    read_pdb_trajectory,
    read_sdf,
    read_sdf_bytes,
    read_sdf_str,
    read_smiles_str,
    read_stl,
    read_stl_bytes,
    read_trr_bytes,
    read_trr_trajectory,
    read_chgcar,
    read_chgcar_str,
    read_poscar,
    read_poscar_str,
    read_xml_forcefield,
    read_xml_forcefield_str,
    read_xsf,
    read_xsf_str,
    read_xtc_bytes,
    read_xtc_trajectory,
    read_xyz,
    read_xyz_bytes,
    read_xyz_str,
    read_xyz_trajectory,
    write_frcmod,
    write_frcmod_str,
    write_prep,
    write_prep_str,
    write_cif,
    write_cif_str,
    write_csv_block,
    write_csv_block_str,
    write_cube,
    write_cube_str,
    write_dcd_bytes,
    write_dcd_trajectory,
    write_gro,
    write_gro_str,
    write_gro_trajectory,
    write_top_forcefield,
    write_top_system,
    write_json_frame_str,
    write_lammps_bond_react_map,
    write_lammps_bond_react_system,
    write_lammps_cmap_forcefield,
    write_lammps_data,
    write_lammps_data_coeffs_str,
    write_lammps_data_str,
    write_lammps_dump_local,
    write_lammps_dump_str,
    write_lammps_dump_trajectory,
    write_lammps_forcefield,
    write_lammps_forcefield_str,
    write_lammps_molecule,
    write_lammps_molecule_json,
    write_mol2,
    write_mol2_str,
    write_msgpack_frame_bytes,
    write_openmm_forcefield,
    write_openmm_forcefield_str,
    write_openmm_system,
    write_openmm_system_str,
    write_pdb,
    write_pdb_str,
    write_pdb_trajectory,
    write_smiles_str,
    write_trr_bytes,
    write_trr_trajectory,
    write_poscar,
    write_poscar_str,
    write_xml_forcefield,
    write_xml_forcefield_str,
    write_xsf,
    write_xsf_str,
    write_xtc_bytes,
    write_xtc_trajectory,
    write_xyz,
    write_xyz_str,
    write_xyz_trajectory,
)
from . import cgsmiles, dcd, gro, lammps, pdb, smiles, trr, xtc, xyz

__all__ = [
    "cgsmiles",
    "dcd",
    "gro",
    "lammps",
    "pdb",
    "read_ac",
    "read_ac_str",
    "read_inpcrd",
    "read_inpcrd_str",
    "read_prep",
    "read_prep_str",
    "read_prmtop",
    "read_prmtop_forcefield",
    "read_prmtop_str",
    "read_prmtop_system",
    "read_cgsmiles_str",
    "read_cif",
    "read_cif_str",
    "read_clpol_alpha",
    "read_clpol_alpha_str",
    "read_csv_block",
    "read_csv_block_str",
    "read_cube",
    "read_cube_str",
    "read_dcd_bytes",
    "read_dcd_trajectory",
    "read_gro",
    "read_gro_str",
    "read_gro_trajectory",
    "read_top_forcefield",
    "read_top_system",
    "read_json_frame_str",
    "read_lammps_cmap_forcefield",
    "read_lammps_data",
    "read_lammps_data_bytes",
    "read_lammps_data_coeffs",
    "read_lammps_data_coeffs_str",
    "read_lammps_data_str",
    "read_lammps_dump_bytes",
    "read_lammps_dump_str",
    "read_lammps_dump_trajectory",
    "read_lammps_forcefield",
    "read_lammps_forcefield_str",
    "read_lammps_log",
    "read_lammps_log_str",
    "read_lammps_molecule",
    "read_lammps_molecule_json",
    "read_mol2",
    "read_mol2_str",
    "read_msgpack_frame_bytes",
    "read_openmm_forcefield",
    "read_openmm_forcefield_str",
    "read_pdb",
    "read_pdb_bytes",
    "read_pdb_str",
    "read_pdb_trajectory",
    "read_sdf",
    "read_sdf_bytes",
    "read_sdf_str",
    "read_smiles_str",
    "read_stl",
    "read_stl_bytes",
    "read_trr_bytes",
    "read_trr_trajectory",
    "read_chgcar",
    "read_chgcar_str",
    "read_poscar",
    "read_poscar_str",
    "read_xml_forcefield",
    "read_xml_forcefield_str",
    "read_xsf",
    "read_xsf_str",
    "read_xtc_bytes",
    "read_xtc_trajectory",
    "read_xyz",
    "read_xyz_bytes",
    "read_xyz_str",
    "read_xyz_trajectory",
    "smiles",
    "trr",
    "write_frcmod",
    "write_frcmod_str",
    "write_prep",
    "write_prep_str",
    "write_cif",
    "write_cif_str",
    "write_csv_block",
    "write_csv_block_str",
    "write_cube",
    "write_cube_str",
    "write_dcd_bytes",
    "write_dcd_trajectory",
    "write_gro",
    "write_gro_str",
    "write_gro_trajectory",
    "write_top_forcefield",
    "write_top_system",
    "write_json_frame_str",
    "write_lammps_bond_react_map",
    "write_lammps_bond_react_system",
    "write_lammps_cmap_forcefield",
    "write_lammps_data",
    "write_lammps_data_coeffs_str",
    "write_lammps_data_str",
    "write_lammps_dump_local",
    "write_lammps_dump_str",
    "write_lammps_dump_trajectory",
    "write_lammps_forcefield",
    "write_lammps_forcefield_str",
    "write_lammps_molecule",
    "write_lammps_molecule_json",
    "write_mol2",
    "write_mol2_str",
    "write_msgpack_frame_bytes",
    "write_openmm_forcefield",
    "write_openmm_forcefield_str",
    "write_openmm_system",
    "write_openmm_system_str",
    "write_pdb",
    "write_pdb_str",
    "write_pdb_trajectory",
    "write_smiles_str",
    "write_trr_bytes",
    "write_trr_trajectory",
    "write_poscar",
    "write_poscar_str",
    "write_xml_forcefield",
    "write_xml_forcefield_str",
    "write_xsf",
    "write_xsf_str",
    "write_xtc_bytes",
    "write_xtc_trajectory",
    "write_xyz",
    "write_xyz_str",
    "write_xyz_trajectory",
    "xtc",
    "xyz",
]

# mrec is the zarr record store (`fs` / `molrs/filesystem`). A Pyodide build
# omits it. Absent, not an import failure of this module — the same rule as
# `molrs.stream.Publisher`.
try:
    from .._native import (
        read_mrec_forcefield,
        read_mrec_frame,
        read_mrec_meta,
        read_mrec_system,
        read_mrec_trajectory,
        write_mrec_forcefield,
        write_mrec_frame,
        write_mrec_system,
        write_mrec_trajectory,
    )
    from . import mrec as mrec
except ImportError:
    pass
else:
    __all__ += [
        "mrec",
        "read_mrec_forcefield",
        "read_mrec_frame",
        "read_mrec_meta",
        "read_mrec_system",
        "read_mrec_trajectory",
        "write_mrec_forcefield",
        "write_mrec_frame",
        "write_mrec_system",
        "write_mrec_trajectory",
    ]
