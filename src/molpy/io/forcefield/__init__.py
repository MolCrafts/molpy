"""Force field I/O: readers and writers for XML, LAMMPS, and GROMACS formats."""

from pathlib import Path
from typing import Any

from .lammps import LAMMPSForceFieldReader, LAMMPSForceFieldWriter
from .top import GromacsForceFieldWriter, GromacsTopReader
from .xml import XMLForceFieldReader, XMLForceFieldWriter, read_xml_forcefield

PathLike = str | Path

__all__ = [
    "GromacsForceFieldWriter",
    "GromacsTopReader",
    "LAMMPSForceFieldReader",
    "LAMMPSForceFieldWriter",
    "XMLForceFieldReader",
    "XMLForceFieldWriter",
    "read_lammps_forcefield",
    "write_lammps_forcefield",
    "read_xml_forcefield",
]


def read_lammps_forcefield(scripts: PathLike | list[PathLike]) -> Any:
    """Read a LAMMPS force-field include (``*.ff``) into a ForceField.

    Thin delegation to the native molrs reader — see
    :func:`molpy.io.read_lammps_forcefield`. The LAMMPS force-field model lives
    in molrs (Rust); molpy does not reimplement parsing here.
    """
    import molrs

    paths = scripts if isinstance(scripts, list) else [scripts]
    if len(paths) == 1:
        return molrs.read_lammps_forcefield(str(paths[0]))
    text = "\n".join(Path(p).read_text() for p in paths)
    return molrs.read_lammps_forcefield_str(text)


def write_lammps_forcefield(
    path: PathLike,
    forcefield: Any,
    *,
    precision: int = 6,
    skip_pair_style: bool = False,
    atom_types: set[str] | None = None,
    bond_types: set[str] | None = None,
    angle_types: set[str] | None = None,
    dihedral_types: set[str] | None = None,
    improper_types: set[str] | None = None,
) -> None:
    """Write a ForceField to a LAMMPS ``*.ff`` include.

    Thin delegation to :func:`molrs.write_lammps_forcefield` — unit conversion
    is the inverse of the native reader and lives in molrs.
    """
    import molrs

    molrs.write_lammps_forcefield(
        str(path),
        forcefield,
        precision=precision,
        skip_pair_style=skip_pair_style,
        atom_types=atom_types,
        bond_types=bond_types,
        angle_types=angle_types,
        dihedral_types=dihedral_types,
        improper_types=improper_types,
    )
