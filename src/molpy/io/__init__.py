"""
MolPy I/O — the **only** public file I/O surface (``mp.io.read_*`` / ``write_*``).

There is no package-root ``mp.read_*`` / ``mp.write_*``, and no ``MolStore`` /
Zarr layer. Kernels and formats that molrs owns are reached through this
module (thin wrappers / readers that call molrs); callers never ``import molrs``.

Supports:
- Data files (PDB, XYZ, LAMMPS, GROMACS, AMBER, …)
- Force field files (LAMMPS ``*.ff``, OpenMM/OPLS XML, AMBER prmtop, GROMACS top)
- Trajectory files (LAMMPS dump, XYZ, DCD/TRR/XTC where available)

Basic usage::

    import molpy as mp
    from molpy.data import get_forcefield_path

    frame = mp.io.read_pdb("structure.pdb")
    result = mp.io.read_lammps_data("data.lammps", atom_style="full")
    ff = mp.io.read_xml_forcefield(get_forcefield_path("oplsaa.xml"))
    traj = mp.io.read_lammps_trajectory("dump.lammpstrj")
"""

from pathlib import Path

import numpy as np

# Type aliases
PathLike = str | Path

# =============================================================================
# Import order: Deepest to shallowest to avoid circular dependencies
# =============================================================================

# 2. Data Readers and Writers
from .data.ac import AcReader
from .data.amber import AmberInpcrdReader

# 1. Deepest level: Base classes
from .data.base import DataReader, DataWriter
from .data.gro import GroReader, GroWriter
from .data.lammps import LammpsDataReader, LammpsDataResult, LammpsDataWriter
from .data.lammps_molecule import (
    LammpsMoleculeReader,
    LammpsMoleculeWriter,
)
from .data.mol2 import Mol2Reader, Mol2Writer
from .data.smiles import SmilesReader
from .data.pdb import PDBReader, PDBWriter
from .data.top import TopReader
from .data.xsf import XsfReader, XsfWriter
from .data.xyz import XYZReader

# 5. Factory functions (use the classes above)
from .readers import (
    read_amber,
    read_amber_ac,
    read_amber_inpcrd,
    read_chgcar,
    read_cube,
    read_dcd_trajectory,
    read_gro,
    read_LAMMPS_log,
    read_lammps_data,
    read_lammps_forcefield,
    read_lammps_molecule,
    read_lammps_trajectory,
    read_mol2,
    read_smiles,
    read_pdb,
    read_pdb_trajectory,
    read_top,
    read_trr_trajectory,
    read_xml_forcefield,
    read_xsf,
    read_xtc_trajectory,
    read_xyz,
    read_xyz_trajectory,
)
from .base import BaseReader
from .trajectory.base import (
    BaseTrajectoryReader,
    TrajectoryWriter,
)

# 3. Trajectory Readers and Writers
from .trajectory.lammps import (
    LammpsTrajectoryWriter,
)
from .trajectory.xyz import XYZTrajectoryWriter

# 4. Log Readers
from .log.lammps import (
    LAMMPSCPUUse,
    LAMMPSLoadBalance,
    LAMMPSLog,
    LAMMPSLogHeader,
    LAMMPSLoopTime,
    LAMMPSMemoryUsage,
    LAMMPSNeighborStatistics,
    LAMMPSPerformance,
    LAMMPSRun,
    LAMMPSThermo,
    LAMMPSTimingBreakdown,
    LAMMPSTimingRow,
    LAMMPSWarning,
)
from .writers import (
    write_gro,
    write_lammps_data,
    write_lammps_data_coeffs,
    write_lammps_forcefield,
    write_lammps_molecule,
    write_bond_react_map,
    write_lammps_bond_react_system,
    write_lammps_system,
    write_lammps_trajectory,
    write_mol2,
    write_pdb,
    write_top,
    write_trr,
    write_xsf,
    write_xtc,
    write_xyz,
    write_xyz_trajectory,
    write_dcd_trajectory,
    write_cube,
)

# 6. Utility functions (shallowest level)
read_txt = np.loadtxt

__all__ = [
    # Core types
    "PathLike",
    # Factory functions - Readers
    "read_amber",
    "read_amber_ac",
    "read_amber_inpcrd",
    "read_gro",
    "read_LAMMPS_log",
    "read_lammps_data",
    "read_lammps_forcefield",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_smiles",
    "read_pdb",
    "read_pdb_trajectory",
    "read_top",
    "read_xml_forcefield",
    "read_xsf",
    "read_xyz",
    "read_xyz_trajectory",
    "read_dcd_trajectory",
    "read_trr_trajectory",
    "read_xtc_trajectory",
    "read_cube",
    "read_chgcar",
    # Factory functions - Writers
    "write_gro",
    "write_lammps_data",
    "write_lammps_data_coeffs",
    "write_lammps_forcefield",
    "write_lammps_molecule",
    "write_bond_react_map",
    "write_lammps_bond_react_system",
    "write_lammps_system",
    "write_lammps_trajectory",
    "write_mol2",
    "write_pdb",
    "write_top",
    "write_xsf",
    "write_xyz",
    "write_xyz_trajectory",
    "write_trr",
    "write_xtc",
    "write_dcd_trajectory",
    "write_cube",
    # Utility functions
    "read_txt",
    # Data Readers
    "DataReader",
    "AcReader",
    "AmberInpcrdReader",
    "GroReader",
    "LammpsDataReader",
    "LammpsDataResult",
    "LammpsMoleculeReader",
    "Mol2Reader",
    "Mol2Writer",
    "SmilesReader",
    "PDBReader",
    "TopReader",
    "XsfReader",
    "XYZReader",
    # Data Writers
    "DataWriter",
    "GroWriter",
    "LammpsDataWriter",
    "LammpsMoleculeWriter",
    "PDBWriter",
    "XsfWriter",
    # ForceField Readers
    "ForceFieldReader",
    "AmberPrmtopReader",
    "GromacsTopReader",
    "MolTemplateReader",
    "XMLForceFieldReader",
    "OPLSAAForceFieldReader",
    # ForceField Writers
    "ForceFieldWriter",
    "LAMMPSForceFieldWriter",
    # Trajectory Readers
    "BaseReader",
    "BaseTrajectoryReader",
    # Trajectory Writers
    "TrajectoryWriter",
    "LammpsTrajectoryWriter",
    "XYZTrajectoryWriter",
    # Log Readers
    "LAMMPSCPUUse",
    "LAMMPSLoadBalance",
    "LAMMPSLog",
    "LAMMPSLogHeader",
    "LAMMPSLoopTime",
    "LAMMPSMemoryUsage",
    "LAMMPSNeighborStatistics",
    "LAMMPSPerformance",
    "LAMMPSRun",
    "LAMMPSThermo",
    "LAMMPSTimingBreakdown",
    "LAMMPSTimingRow",
    "LAMMPSWarning",
    # Utility Classes
    "ZipReader",
]
