"""Experimental. Data file writer factory functions — molrs-backed.

Signature-compatible with :mod:`molpy.io.writers`; delegates to
:mod:`molrs.io` under the hood.  See :ref:`molrs-io-experimental`
for the graduation plan.
"""

from pathlib import Path
from typing import Any

PathLike = str | Path


# =============================================================================
# Data File Writers
# =============================================================================


def write_lammps_data(file: PathLike, frame: Any, atom_style: str = "full") -> None:
    """Experimental. Write LAMMPS data file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.write_lammps_data` for the stable implementation.
    """
    from .data.lammps import LammpsDataWriter

    writer = LammpsDataWriter(Path(file), atom_style=atom_style)
    writer.write(frame)


def write_gro(file: PathLike, frame: Any) -> None:
    """Experimental. Write GROMACS GRO file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.write_gro` for the stable implementation.
    """
    from .data.gro import GroWriter

    writer = GroWriter(Path(file))
    writer.write(frame)


def write_pdb(file: PathLike, frame: Any) -> None:
    """Experimental. Write PDB file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.write_pdb` for the stable implementation.
    """
    from .data.pdb import PDBWriter

    writer = PDBWriter(Path(file))
    writer.write(frame)


def write_xyz(file: PathLike, frame: Any) -> None:
    """Experimental. Write XYZ file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.write_xyz` for the stable implementation.
    """
    from .data.xyz import XYZWriter

    XYZWriter(Path(file)).write(frame)
