"""TRR, the GROMACS full-precision trajectory format — ``molrs::io::trr``.

The one-frame doors are :func:`molrs.io.read_trr` / :func:`molrs.io.write_trr`;
this module holds the format's classes: :class:`TrrReader` and
:class:`TrrWriter`, one frame per ``write``.
"""

from .._native import (
    TrrReader,
    TrrWriter,
)

__all__ = ["TrrReader", "TrrWriter"]
