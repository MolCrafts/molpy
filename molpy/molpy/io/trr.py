"""TRR, the GROMACS full-precision trajectory format — ``molrs::io::trr``.

The one-frame doors are :func:`molpy.io.read_trr` / :func:`molpy.io.write_trr`;
this module holds the format's classes: :class:`TrrReader` and
:class:`TrrWriter`, one frame per ``write``.
"""

from molpy._core import (
    TrrReader,
    TrrWriter,
)

__all__ = [
    "TrrReader",
    "TrrWriter",
]
