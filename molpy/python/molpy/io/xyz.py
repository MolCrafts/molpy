"""XYZ and extended XYZ — ``molrs::io::xyz``.

The one-frame doors are :func:`molpy.io.read_xyz` / :func:`molpy.io.write_xyz`;
this module holds the format's classes: :class:`XyzReader`, every frame of a
file, and :class:`XyzWriter`, one frame per ``write``.
"""

from molpy._core import (
    XyzReader,
    XyzWriter,
)

__all__ = [
    "XyzReader",
    "XyzWriter",
]
