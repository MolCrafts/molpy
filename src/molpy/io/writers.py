"""
Data file writer factory functions.

This module provides convenient factory functions for creating various data file writers.
All functions write Frame or ForceField objects to files.
"""

from pathlib import Path
from typing import Any

PathLike = str | Path


# =============================================================================
# Data File Writers
# =============================================================================


def write_lammps_data(file: PathLike, frame: Any, atom_style: str = "full") -> None:
    """
    Write a Frame object to a LAMMPS data file.

    Args:
        file: Output file path
        frame: Frame object to write
        atom_style: LAMMPS atom style (default: 'full')
    """
    from .data.lammps import LammpsDataWriter

    writer = LammpsDataWriter(Path(file), atom_style=atom_style)
    writer.write(frame)


def write_pdb(file: PathLike, frame: Any) -> None:
    """
    Write a Frame object to a PDB file.

    Args:
        file: Output file path
        frame: Frame object to write
    """
    from .data.pdb import PDBWriter

    writer = PDBWriter(Path(file))
    writer.write(frame)


def write_xsf(file: PathLike, frame: Any) -> None:
    """
    Write a Frame object to an XSF file.

    Args:
        file: Output file path
        frame: Frame object to write
    """
    from .data.xsf import XsfWriter

    writer = XsfWriter(Path(file))
    writer.write(frame)


def write_lammps_molecule(
    file: PathLike, frame: Any, format_type: str = "native"
) -> None:
    """
    Write a Frame object to a LAMMPS molecule file.

    Args:
        file: Output file path
        frame: Frame object to write
        format_type: Format type (default: 'native')
    """
    from .data.lammps_molecule import LammpsMoleculeWriter

    writer = LammpsMoleculeWriter(Path(file), format_type)
    writer.write(frame)


# =============================================================================
# Force Field Writers
# =============================================================================


def write_lammps_forcefield(
    file: PathLike, forcefield: Any, precision: int = 6
) -> None:
    """
    Write a ForceField object to a LAMMPS force field file.

    Args:
        file: Output file path
        forcefield: ForceField object to write
        precision: Number of decimal places for floating point values
    """
    from .forcefield.lammps import LAMMPSForceFieldWriter

    writer = LAMMPSForceFieldWriter(Path(file), precision=precision)
    writer.write(forcefield)


# =============================================================================
# Trajectory Writers
# =============================================================================


def write_lammps_trajectory(
    file: PathLike, frames: list, atom_style: str = "full"
) -> None:
    """
    Write frames to a LAMMPS trajectory file.

    Args:
        file: Output file path
        frames: List of Frame objects to write
        atom_style: LAMMPS atom style (default: 'full')
    """
    from .trajectory.lammps import LammpsTrajectoryWriter

    with LammpsTrajectoryWriter(Path(file), atom_style) as writer:
        for i, frame in enumerate(frames):
            timestep = getattr(frame, "step", i)
            writer.write_frame(frame, timestep)


def write_xyz_trajectory(file: PathLike, frames: list) -> None:
    """
    Write frames to an XYZ trajectory file.

    Args:
        file: Output file path
        frames: List of Frame objects to write
    """
    from .trajectory.xyz import XYZTrajectoryWriter

    with XYZTrajectoryWriter(file) as writer:
        for frame in frames:
            writer.write_frame(frame)


# =============================================================================
# Combined System Writers
# =============================================================================


def write_lammps_system(workdir: PathLike, frame: Any, forcefield: Any) -> None:
    """
    Write a complete LAMMPS system (data + forcefield) to a directory.

    Args:
        workdir: Output directory path
        frame: Frame object containing structure
        forcefield: ForceField object containing parameters
    """
    workdir_path = Path(workdir)
    if not workdir_path.exists():
        workdir_path.mkdir(parents=True, exist_ok=True)

    # Use directory name as file stem
    file_stem = workdir_path / workdir_path.stem
    write_lammps_data(file_stem.with_suffix(".data"), frame)
    write_lammps_forcefield(file_stem.with_suffix(".ff"), forcefield)
