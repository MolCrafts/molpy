"""
Data file reader factory functions.

This module provides convenient factory functions for creating various data file readers.
All functions return Frame objects by populating an optional frame parameter.
"""

from pathlib import Path
from typing import Any

PathLike = str | Path


# Lazy import to avoid loading all dependencies
def _ensure_frame(frame):
    """Ensure a Frame object exists."""
    if frame is None:
        from molpy import Frame

        return Frame()
    return frame


# =============================================================================
# Data File Readers
# =============================================================================


def read_lammps_data(file: PathLike, atom_style: str, frame: Any = None) -> Any:
    """
    Read LAMMPS data file and return a Frame object.

    Args:
        file: Path to LAMMPS data file
        atom_style: LAMMPS atom style (e.g., 'full', 'atomic')
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.lammps import LammpsDataReader

    reader = LammpsDataReader(Path(file), atom_style)
    return reader.read(frame=frame)


def read_lammps_molecule(file: PathLike, frame: Any = None) -> Any:
    """
    Read LAMMPS molecule file and return a Frame object.

    Args:
        file: Path to LAMMPS molecule file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.lammps_molecule import LammpsMoleculeReader

    reader = LammpsMoleculeReader(Path(file))
    return reader.read(frame=frame)


def read_pdb(file: PathLike, frame: Any = None) -> Any:
    """
    Read PDB file and return a Frame object.

    Args:
        file: Path to PDB file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.pdb import PDBReader

    frame = _ensure_frame(frame)
    reader = PDBReader(Path(file))
    return reader.read(frame)


def read_amber_inpcrd(inpcrd: PathLike, frame: Any = None) -> Any:
    """
    Read AMBER inpcrd file and return a Frame object.

    Args:
        inpcrd: Path to AMBER inpcrd file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.amber import AmberInpcrdReader

    frame = _ensure_frame(frame)
    reader = AmberInpcrdReader(Path(inpcrd))
    return reader.read(frame)


def read_amber_ac(file: PathLike, frame: Any = None) -> Any:
    """
    Read AC file and return a Frame object.

    Args:
        file: Path to AC file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.ac import AcReader

    frame = _ensure_frame(frame)
    reader = AcReader(Path(file))
    return reader.read(frame)


def read_mol2(file: PathLike, frame: Any = None) -> Any:
    """
    Read mol2 file and return a Frame object.

    Args:
        file: Path to mol2 file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.mol2 import Mol2Reader

    frame = _ensure_frame(frame)
    reader = Mol2Reader(Path(file))
    return reader.read(frame)


def read_xsf(file: PathLike, frame: Any = None) -> Any:
    """
    Read XSF file and return a Frame object.

    Args:
        file: Path to XSF file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.xsf import XsfReader

    reader = XsfReader(Path(file))
    return reader.read(frame)


def read_gro(file: PathLike, frame: Any = None) -> Any:
    """
    Read GROMACS gro file and return a Frame object.

    Args:
        file: Path to gro file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.gro import GroReader

    frame = _ensure_frame(frame)
    reader = GroReader(Path(file))
    return reader.read(frame)


def read_xyz(file: PathLike, frame: Any = None) -> Any:
    """
    Read XYZ file and return a Frame object.

    Args:
        file: Path to XYZ file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object
    """
    from .data.xyz import XYZReader

    frame = _ensure_frame(frame)
    reader = XYZReader(Path(file))
    return reader.read(frame)


def read_h5(file: PathLike, frame: Any = None) -> Any:
    """
    Read HDF5 file and return a Frame object.

    Args:
        file: Path to HDF5 file
        frame: Optional existing Frame to populate

    Returns:
        Populated Frame object

    Examples:
        >>> frame = read_h5("structure.h5")
        >>> frame["atoms"]["x"]
        array([0., 1., 2.])
    """
    from .data.h5 import HDF5Reader

    frame = _ensure_frame(frame)
    reader = HDF5Reader(Path(file))
    return reader.read(frame)


# =============================================================================
# Force Field Readers
# =============================================================================


def read_lammps_forcefield(
    scripts: PathLike | list[PathLike], forcefield: Any = None
) -> Any:
    """
    Read LAMMPS force field file and return a ForceField object.

    Args:
        scripts: Path or list of paths to LAMMPS force field scripts
        forcefield: Optional existing ForceField to populate

    Returns:
        Populated ForceField object
    """
    from .forcefield.lammps import LAMMPSForceFieldReader

    reader = LAMMPSForceFieldReader(scripts)
    return reader.read(forcefield=forcefield)


def read_xml_forcefield(file: PathLike) -> Any:
    """
    Read XML force field file and return a ForceField object.

    Args:
        file: Path to XML force field file

    Returns:
        ForceField object
    """
    from .forcefield.xml import read_xml_forcefield as _read_xml

    return _read_xml(file)


def read_amber_prmtop(
    prmtop: PathLike, inpcrd: PathLike | None = None, frame: Any = None
) -> Any:
    """
    Read AMBER prmtop and optional inpcrd files.

    Args:
        prmtop: Path to AMBER prmtop file
        inpcrd: Optional path to AMBER inpcrd file
        frame: Optional existing Frame to populate

    Returns:
        Tuple of (Frame, ForceField)
    """
    from .forcefield.amber import AmberPrmtopReader

    frame = _ensure_frame(frame)
    prmtop_path = Path(prmtop)
    reader = AmberPrmtopReader(prmtop_path)
    frame, ff = reader.read(frame)

    if inpcrd is not None:
        from .data.amber import AmberInpcrdReader

        inpcrd_reader = AmberInpcrdReader(Path(inpcrd))
        frame = inpcrd_reader.read(frame)

    return frame, ff


def read_top(file: PathLike, forcefield: Any = None) -> Any:
    """
    Read GROMACS topology file and return a ForceField object.

    Args:
        file: Path to GROMACS .top file
        forcefield: Optional existing ForceField to populate

    Returns:
        Populated ForceField object
    """
    from molpy import ForceField

    from .forcefield.top import GromacsTopReader

    if forcefield is None:
        forcefield = ForceField()

    reader = GromacsTopReader(Path(file))
    return reader.read(forcefield)


# =============================================================================
# Trajectory Readers
# =============================================================================


def read_lammps_trajectory(traj: PathLike, frame: Any = None) -> Any:
    """
    Read LAMMPS trajectory file and return a trajectory reader.

    Args:
        traj: Path to LAMMPS trajectory file
        frame: Optional reference Frame for topology

    Returns:
        LammpsTrajectoryReader object
    """
    from .trajectory.lammps import LammpsTrajectoryReader

    return LammpsTrajectoryReader(Path(traj), frame)


def read_xyz_trajectory(file: PathLike) -> Any:
    """
    Read XYZ trajectory file and return a trajectory reader.

    Args:
        file: Path to XYZ trajectory file

    Returns:
        XYZTrajectoryReader object
    """
    from .trajectory.xyz import XYZTrajectoryReader

    return XYZTrajectoryReader(Path(file))


def read_h5_trajectory(file: PathLike) -> Any:
    """
    Read HDF5 trajectory file and return a trajectory reader.

    Args:
        file: Path to HDF5 trajectory file

    Returns:
        HDF5TrajectoryReader object

    Examples:
        >>> reader = read_h5_trajectory("trajectory.h5")
        >>> frame = reader.read_frame(0)
        >>> for frame in reader:
        ...     process(frame)
    """
    from .trajectory.h5 import HDF5TrajectoryReader

    return HDF5TrajectoryReader(Path(file))


# =============================================================================
# Backward Compatibility: Complex system readers
# =============================================================================


def read_lammps(
    data: PathLike,
    scripts: PathLike | list[PathLike] | None = None,
    frame: Any = None,
    atomstyle: str = "full",
) -> Any:
    """
    Read LAMMPS data and optional force field files.

    Args:
        data: Path to LAMMPS data file
        scripts: Optional path(s) to LAMMPS force field scripts
        frame: Optional existing Frame to populate
        atomstyle: LAMMPS atom style (default: 'full')

    Returns:
        Frame object (force field is loaded but returned separately if needed)

    Note:
        For new code, prefer using read_lammps_data() and read_lammps_forcefield() separately.
    """
    if scripts is not None:
        # Load force field first (though return value not used in original)
        _ = read_lammps_forcefield(scripts)

    return read_lammps_data(data, atomstyle, frame)


def read_amber_system(
    prmtop: PathLike,
    inpcrd: PathLike | None = None,
    system: Any = None,
) -> Any:
    """
    Read AMBER prmtop and optional inpcrd files (legacy function).

    Args:
        prmtop: Path to AMBER prmtop file
        inpcrd: Optional path to AMBER inpcrd file
        system: Optional FrameSystem (unused, kept for compatibility)

    Returns:
        Tuple of (Frame, ForceField) wrapped in FrameSystem namedtuple

    Note:
        For new code, prefer using read_amber_prmtop() directly.
    """
    frame, ff = read_amber_prmtop(prmtop, inpcrd)

    # Original function returned FrameSystem namedtuple
    from collections import namedtuple

    FrameSystem = namedtuple("FrameSystem", ["frame", "forcefield", "box"])
    return FrameSystem(frame=frame, forcefield=ff, box=getattr(frame, "box", None))


def read_gromacs_system(
    gro_file: PathLike,
    top_file: PathLike | None = None,
    system: Any = None,
) -> Any:
    """
    Read GROMACS structure and optional topology files.

    Args:
        gro_file: Path to GROMACS .gro file
        top_file: Optional path to GROMACS .top file
        system: Optional FrameSystem (unused, kept for compatibility)

    Returns:
        Frame if no topology, or FrameSystem namedtuple of (Frame, ForceField) if topology provided

    Note:
        For new code, prefer using read_gro() and read_top() separately.
    """
    frame = read_gro(gro_file)

    if top_file is not None:
        forcefield = read_top(top_file)
        from collections import namedtuple

        FrameSystem = namedtuple("FrameSystem", ["frame", "forcefield", "box"])
        return FrameSystem(
            frame=frame, forcefield=forcefield, box=getattr(frame, "box", None)
        )

    return frame
