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


def write_h5(
    file: PathLike,
    frame: Any,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write a Frame object to an HDF5 file.

    Args:
        file: Output file path
        frame: Frame object to write
        compression: Compression algorithm (None, 'gzip', 'lzf', 'szip').
            Defaults to 'gzip'.
        compression_opts: Compression level (for gzip: 0-9). Defaults to 4.

    Examples:
        >>> frame = Frame(blocks={"atoms": {"x": [0, 1, 2], "y": [0, 0, 0]}})
        >>> write_h5("structure.h5", frame)
    """
    from .data.h5 import HDF5Writer

    writer = HDF5Writer(
        Path(file), compression=compression, compression_opts=compression_opts
    )
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


def write_h5_trajectory(
    file: PathLike,
    frames: list,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write frames to an HDF5 trajectory file.

    Args:
        file: Output file path
        frames: List of Frame objects to write
        compression: Compression algorithm (None, 'gzip', 'lzf', 'szip').
            Defaults to 'gzip'.
        compression_opts: Compression level (for gzip: 0-9). Defaults to 4.

    Examples:
        >>> frames = [frame0, frame1, frame2]
        >>> write_h5_trajectory("trajectory.h5", frames)
    """
    from .trajectory.h5 import HDF5TrajectoryWriter

    with HDF5TrajectoryWriter(
        Path(file), compression=compression, compression_opts=compression_opts
    ) as writer:
        for frame in frames:
            writer.write_frame(frame)


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

    # Extract type names from frame to create whitelist
    atom_types = None
    bond_types = None
    angle_types = None
    dihedral_types = None
    improper_types = None

    if "atoms" in frame and "type" in frame["atoms"]:
        atom_types = set(frame["atoms"]["type"])

    if "bonds" in frame and "type" in frame["bonds"]:
        bond_types = set(frame["bonds"]["type"])

    if "angles" in frame and "type" in frame["angles"]:
        angle_types = set(frame["angles"]["type"])

    if "dihedrals" in frame and "type" in frame["dihedrals"]:
        dihedral_types = set(frame["dihedrals"]["type"])

    if "impropers" in frame and "type" in frame["impropers"]:
        improper_types = set(frame["impropers"]["type"])

    # Write forcefield with whitelist
    from .forcefield.lammps import LAMMPSForceFieldWriter

    writer = LAMMPSForceFieldWriter(file_stem.with_suffix(".ff"))
    writer.write(
        forcefield,
        atom_types=atom_types,
        bond_types=bond_types,
        angle_types=angle_types,
        dihedral_types=dihedral_types,
        improper_types=improper_types,
    )
