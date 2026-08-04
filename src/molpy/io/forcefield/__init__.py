"""Force field I/O: readers and writers for XML, LAMMPS, and GROMACS formats."""

from pathlib import Path
from typing import Any

from .lammps import LAMMPSForceFieldWriter
from .top import GromacsForceFieldWriter, GromacsTopReader
from .xml import XMLForceFieldReader, XMLForceFieldWriter, read_xml_forcefield

PathLike = str | Path

__all__ = [
    "GromacsForceFieldWriter",
    "GromacsTopReader",
    "LAMMPSForceFieldWriter",
    "XMLForceFieldReader",
    "XMLForceFieldWriter",
    "read_lammps_forcefield",
    "write_lammps_forcefield",
    "read_xml_forcefield",
]


def read_lammps_forcefield(scripts: PathLike | list[PathLike]) -> Any:
    """Read a LAMMPS force-field include (``*.ff``) into a ForceField."""
    import molrs

    paths = scripts if isinstance(scripts, list) else [scripts]
    if len(paths) == 1:
        return molrs.ff.read_lammps_forcefield(str(paths[0]))
    text = "\n".join(Path(p).read_text() for p in paths)
    return molrs.ff.read_lammps_forcefield_str(text)


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
    """Write a ForceField to a LAMMPS ``*.ff`` include."""
    import molrs

    molrs.ff.write_lammps_forcefield(
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
