"""Experimental. Data file reader factory functions — molrs-backed.

Signature-compatible with :mod:`molpy.io.readers`; delegates to
:mod:`molrs.io` under the hood.  See :ref:`molrs-io-experimental`
for the graduation plan.
"""

from pathlib import Path
from typing import Any

PathLike = str | Path


# =============================================================================
# Data File Readers
# =============================================================================


def read_lammps_data(file: PathLike, atom_style: str, frame: Any = None) -> Any:
    """Experimental. Read LAMMPS data file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_lammps_data` for the stable implementation.
    """
    from .data.lammps import LammpsDataReader

    reader = LammpsDataReader(Path(file), atom_style)
    return reader.read(frame=frame)


def read_pdb(file: PathLike, frame: Any = None) -> Any:
    """Experimental. Read PDB file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_pdb` for the stable implementation.
    """
    from .data.pdb import PDBReader

    return PDBReader(Path(file)).read()


def read_gro(file: PathLike, frame: Any = None) -> Any:
    """Experimental. Read GROMACS GRO file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_gro` for the stable implementation.
    """
    from .data.gro import GroReader

    return GroReader(Path(file)).read()


def read_xyz(file: PathLike, frame: Any = None) -> Any:
    """Experimental. Read XYZ file via molrs backend.

    .. deprecated::
        Use :func:`molpy.io.read_xyz` for the stable implementation.
    """
    from .data.xyz import XYZReader

    return XYZReader(Path(file)).read()
