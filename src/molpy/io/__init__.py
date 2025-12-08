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
# Import organized reader/writer functions
# =============================================================================

# Data readers
# Force field readers
# Trajectory readers
from .readers import (
    read_amber_ac,
    read_amber_inpcrd,
    read_amber_prmtop,
    read_amber_system,
    read_gro,
    read_gromacs_system,
    read_h5,
    read_lammps,
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
    read_h5_trajectory,
)

# Data writers
# Force field writers
# Trajectory writers
# System writers
from .writers import (
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

# Backward compatibility aliases
read_amber = read_amber_prmtop  # Keep old name for compatibility
write_lammps = write_lammps_system  # Keep old name for compatibility


# =============================================================================
# Utility functions
# =============================================================================

# Numpy loadtxt shortcut
read_txt = np.loadtxt


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core types
    "PathLike",
    # AMBER readers
    "read_amber",
    "read_amber_ac",
    "read_amber_inpcrd",
    "read_amber_prmtop",
    "read_amber_system",
    # GROMACS readers
    "read_gro",
    "read_gromacs_system",
    "read_top",
    # LAMMPS readers
    "read_lammps",
    "read_lammps_data",
    "read_lammps_forcefield",
    "read_lammps_molecule",
    "read_lammps_trajectory",
    # Other data readers
    "read_h5",
    "read_mol2",
    "read_pdb",
    "read_xml_forcefield",
    "read_xsf",
    "read_xyz",
    # Trajectory readers
    "read_xyz_trajectory",
    "read_h5_trajectory",
    # Utility functions
    "read_txt",
    # LAMMPS writers
    "write_lammps",
    "write_lammps_data",
    "write_lammps_forcefield",
    "write_lammps_molecule",
    "write_lammps_system",
    "write_lammps_trajectory",
    # Other data writers
    "write_h5",
    "write_pdb",
    "write_xsf",
    # Trajectory writers
    "write_xyz_trajectory",
    "write_h5_trajectory",
]
