"""MolPy I/O — the **only** public file I/O surface (``mp.io.read_*`` / ``write_*``).

There is no package-root ``mp.read_*`` / ``mp.write_*``. Each name here has one
home: most are the native readers and writers, re-exported by identity
(``mp.io.read_gro is molrs.io.read_gro``); the rest are molpy's own, in
:mod:`molpy.io.readers`, :mod:`molpy.io.writers` and
:mod:`molpy.io.data.lammps`, where molpy adds behaviour the native door lacks.

Supports:
- Data files (PDB, XYZ, LAMMPS, GROMACS, AMBER, MOL2, XSF, Cube, CHGCAR, …)
- Force field files (LAMMPS ``*.ff``, OpenMM/OPLS XML, AMBER prmtop, GROMACS top)
- Trajectory files (LAMMPS dump, XYZ, PDB, GRO, DCD/TRR/XTC)
- LAMMPS logs
- Scientific records (``*.mrec`` stores)

Names come in pairs, as in the native core: a format ``X`` holding one frame is
read by ``read_X`` and written by ``write_X``; a sequence of frames by
``read_X_trajectory`` / ``write_X_trajectory``.

Basic usage::

    import molpy as mp

    frame = mp.io.read_pdb("structure.pdb")
    result = mp.io.read_lammps_data("data.lammps", atom_style="full")
    ff = mp.io.read_xml_forcefield(mp.data.get_forcefield_path("tip3p.xml"))
    traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
"""

from molrs.ff import (
    read_forcefield_xml as read_xml_forcefield,
    read_gromacs_top_ff as read_gromacs_forcefield,
    read_lammps_data_coeffs,
    read_lammps_forcefield,
    write_forcefield_xml as write_xml_forcefield,
    write_gromacs_top_ff as write_gromacs_forcefield,
    write_lammps_data_coeffs,
    write_lammps_forcefield,
    write_lammps_forcefield_str,
)
from molrs.io import (
    LammpsCpuUse,
    LammpsLoadBalance,
    LammpsLog,
    LammpsLogHeader,
    LammpsLoopTime,
    LammpsMemoryUsage,
    LammpsNeighborStatistics,
    LammpsPerformance,
    LammpsRun,
    LammpsThermo,
    LammpsTimingBreakdown,
    LammpsTimingRow,
    LammpsWarning,
    mrec_sections,
    parse_lammps_log_text,
    read_chgcar,
    read_cube,
    read_dcd_trajectory,
    read_gro,
    read_gro_trajectory,
    read_lammps_log,
    read_lammps_molecule,
    read_lammps_trajectory,
    read_mol2,
    read_mrec,
    read_mrec_meta,
    read_mrec_system,
    read_mrec_trajectory,
    read_pdb,
    read_pdb_trajectory,
    read_top,
    read_trr_trajectory,
    read_xsf,
    read_xtc_trajectory,
    read_xyz_trajectory,
    write_cube,
    write_dcd_trajectory,
    write_gro,
    write_gro_trajectory,
    write_lammps_dump_local,
    write_lammps_molecule,
    write_lammps_trajectory,
    write_mol2,
    write_mrec,
    write_mrec_system,
    write_mrec_trajectory,
    write_pdb_trajectory,
    write_smarts,
    write_top,
    write_trr_trajectory,
    write_xsf,
    write_xtc_trajectory,
    write_xyz,
    write_xyz_trajectory,
)

from . import mrec
from .data.lammps import LammpsDataResult, read_lammps_data, write_lammps_data
from .data.lammps_bond_react import BondReactTemplate
from .readers import (
    read_amber,
    read_amber_ac,
    read_amber_inpcrd,
    read_smiles,
    read_xyz,
)
from .writers import (
    write_bond_react_map,
    write_lammps_bond_react_system,
    write_pdb,
)

__all__ = [
    "BondReactTemplate",
    "LammpsCpuUse",
    "LammpsDataResult",
    "LammpsLoadBalance",
    "LammpsLog",
    "LammpsLogHeader",
    "LammpsLoopTime",
    "LammpsMemoryUsage",
    "LammpsNeighborStatistics",
    "LammpsPerformance",
    "LammpsRun",
    "LammpsThermo",
    "LammpsTimingBreakdown",
    "LammpsTimingRow",
    "LammpsWarning",
    "mrec",
    "mrec_sections",
    "parse_lammps_log_text",
    "read_amber",
    "read_amber_ac",
    "read_amber_inpcrd",
    "read_chgcar",
    "read_cube",
    "read_dcd_trajectory",
    "read_gro",
    "read_gro_trajectory",
    "read_gromacs_forcefield",
    "read_lammps_data",
    "read_lammps_data_coeffs",
    "read_lammps_forcefield",
    "read_lammps_log",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_mrec",
    "read_mrec_meta",
    "read_mrec_system",
    "read_mrec_trajectory",
    "read_pdb",
    "read_pdb_trajectory",
    "read_smiles",
    "read_top",
    "read_trr_trajectory",
    "read_xml_forcefield",
    "read_xsf",
    "read_xtc_trajectory",
    "read_xyz",
    "read_xyz_trajectory",
    "write_bond_react_map",
    "write_cube",
    "write_dcd_trajectory",
    "write_gro",
    "write_gro_trajectory",
    "write_gromacs_forcefield",
    "write_lammps_bond_react_system",
    "write_lammps_data",
    "write_lammps_data_coeffs",
    "write_lammps_dump_local",
    "write_lammps_forcefield",
    "write_lammps_forcefield_str",
    "write_lammps_molecule",
    "write_lammps_trajectory",
    "write_mol2",
    "write_mrec",
    "write_mrec_system",
    "write_mrec_trajectory",
    "write_pdb",
    "write_pdb_trajectory",
    "write_smarts",
    "write_top",
    "write_trr_trajectory",
    "write_xml_forcefield",
    "write_xsf",
    "write_xtc_trajectory",
    "write_xyz",
    "write_xyz_trajectory",
]
