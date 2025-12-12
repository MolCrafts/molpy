"""
MolPy IO Module - Unified interface for molecular file I/O.

This module provides a clean, organized interface for reading and writing
various molecular file formats. It supports:
- Data files (PDB, XYZ, LAMMPS, GROMACS, AMBER, etc.)
- Force field files (LAMMPS, XML, AMBER prmtop, GROMACS top)
- Trajectory files (LAMMPS dump, XYZ)

Design Principles:
-----------------
1. Reader/Writer Pattern: Each format has dedicated Reader and Writer classes
2. Factory Functions: Convenient read_xxx/write_xxx functions for simple usage
3. Lazy Imports: Dependencies loaded only when needed
4. Unified Interface: All readers have read(), all writers have write()

Basic Usage:
-----------
    # Reading data files
    from molpy.io import read_pdb, read_lammps_data

    frame = read_pdb("structure.pdb")
    frame = read_lammps_data("data.lammps", atom_style="full")

    # Writing data files
    from molpy.io import write_pdb, write_lammps_data

    write_pdb("output.pdb", frame)
    write_lammps_data("output.data", frame, atom_style="full")

    # Reading force fields
    from molpy.io import read_xml_forcefield, read_lammps_forcefield

    ff = read_xml_forcefield("oplsaa.xml")
    ff = read_lammps_forcefield("forcefield.in")

    # Reading trajectories
    from molpy.io import read_lammps_trajectory, read_xyz_trajectory

    traj = read_lammps_trajectory("dump.lammpstrj")
    for frame in traj:
        process(frame)
"""

from pathlib import Path
from typing import Any

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
from .data.h5 import HDF5Reader, HDF5Writer
from .data.lammps import LammpsDataReader, LammpsDataWriter
from .data.lammps_molecule import LammpsMoleculeReader, LammpsMoleculeWriter
from .data.mol2 import Mol2Reader
from .data.pdb import PDBReader, PDBWriter
from .data.top import TopReader
from .data.xsf import XsfReader, XsfWriter
from .data.xyz import XYZReader

# 5. Factory functions (use the classes above)
from .readers import (
    read_amber_ac,
    read_amber_inpcrd,
    read_amber_prmtop,
    read_gro,
    read_h5,
    read_h5_trajectory,
    read_lammps_data,
    read_lammps_forcefield,
    read_lammps_molecule,
    read_lammps_trajectory,
    read_mol2,
    read_pdb,
    read_top,
    read_xml_forcefield,
    read_xsf,
    read_xyz,
    read_xyz_trajectory,
)
from .trajectory.base import BaseTrajectoryReader, FrameLocation, TrajectoryWriter

# 3. Trajectory Readers and Writers
from .trajectory.h5 import HDF5TrajectoryReader, HDF5TrajectoryWriter
from .trajectory.lammps import LammpsTrajectoryReader, LammpsTrajectoryWriter
from .trajectory.xyz import XYZTrajectoryReader, XYZTrajectoryWriter
from .writers import (
    write_gro,
    write_h5,
    write_h5_trajectory,
    write_lammps_data,
    write_lammps_forcefield,
    write_lammps_molecule,
    write_lammps_system,
    write_lammps_trajectory,
    write_pdb,
    write_xsf,
    write_xyz_trajectory,
)

# 6. Utility functions (shallowest level)
read_txt = np.loadtxt

__all__ = [
    # Core types
    "PathLike",
    # Factory functions - Readers
    "read_amber_ac",
    "read_amber_inpcrd",
    "read_amber_prmtop",
    "read_gro",
    "read_h5",
    "read_lammps_data",
    "read_lammps_forcefield",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    "read_mol2",
    "read_pdb",
    "read_top",
    "read_xml_forcefield",
    "read_xsf",
    "read_xyz",
    "read_xyz_trajectory",
    "read_h5_trajectory",
    # Factory functions - Writers
    "write_gro",
    "write_h5",
    "write_h5_trajectory",
    "write_lammps_data",
    "write_lammps_forcefield",
    "write_lammps_molecule",
    "write_lammps_system",
    "write_lammps_trajectory",
    "write_pdb",
    "write_xsf",
    "write_xyz_trajectory",
    # Utility functions
    "read_txt",
    # Data Readers
    "DataReader",
    "AcReader",
    "AmberInpcrdReader",
    "GroReader",
    "HDF5Reader",
    "LammpsDataReader",
    "LammpsMoleculeReader",
    "Mol2Reader",
    "PDBReader",
    "TopReader",
    "XsfReader",
    "XYZReader",
    # Data Writers
    "DataWriter",
    "GroWriter",
    "HDF5Writer",
    "LammpsDataWriter",
    "LammpsMoleculeWriter",
    "PDBWriter",
    "XsfWriter",
    # ForceField Readers
    "ForceFieldReader",
    "AmberPrmtopReader",
    "GromacsTopReader",
    "LAMMPSForceFieldReader",
    "MolTemplateReader",
    "XMLForceFieldReader",
    "OPLSAAForceFieldReader",
    # ForceField Writers
    "ForceFieldWriter",
    "LAMMPSForceFieldWriter",
    # Trajectory Readers
    "BaseTrajectoryReader",
    "FrameLocation",
    "HDF5TrajectoryReader",
    "LammpsTrajectoryReader",
    "XYZTrajectoryReader",
    # Trajectory Writers
    "TrajectoryWriter",
    "HDF5TrajectoryWriter",
    "LammpsTrajectoryWriter",
    "XYZTrajectoryWriter",
    # Utility Classes
    "ZipReader",
]
