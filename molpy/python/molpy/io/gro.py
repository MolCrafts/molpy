"""GRO, the GROMACS structure / trajectory format — ``molrs::io::gro``.

The one-frame doors are :func:`molpy.io.read_gro` / :func:`molpy.io.write_gro`;
this module holds the format's classes: :class:`GroReader`, every frame of a
file, and :class:`GroWriter`, one frame per ``write``.
"""

from molpy._core import (
    GroReader,
    GroWriter,
)

__all__ = [
    "GroReader",
    "GroWriter",
]
