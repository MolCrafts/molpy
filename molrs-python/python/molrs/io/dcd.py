"""DCD, the CHARMM / NAMD binary trajectory format — ``molrs::io::dcd``.

The one-frame doors are :func:`molrs.io.read_dcd` / :func:`molrs.io.write_dcd`;
this module holds the format's classes: :class:`DcdReader` (O(1) random
access) and :class:`DcdWriter`, one frame per ``write``.
"""

from .._native import (
    DcdReader,
    DcdWriter,
)

__all__ = ["DcdReader", "DcdWriter"]
