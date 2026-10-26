"""DCD, the CHARMM / NAMD binary trajectory format — ``molrs::io::dcd``.

The one-frame doors are :func:`molpy.io.read_dcd` / :func:`molpy.io.write_dcd`;
this module holds the format's classes: :class:`DcdReader` (O(1) random
access) and :class:`DcdWriter`, one frame per ``write``.
"""

from molpy._core import (
    DcdReader,
    DcdWriter,
)

__all__ = [
    "DcdReader",
    "DcdWriter",
]
