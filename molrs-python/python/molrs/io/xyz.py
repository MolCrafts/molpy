"""XYZ and extended XYZ — ``molrs::io::xyz``.

The one-frame doors are :func:`molrs.io.read_xyz` / :func:`molrs.io.write_xyz`;
this module holds the format's classes: :class:`XyzReader`, every frame of a
file, and :class:`XyzWriter`, one frame per ``write``.
"""

from .._native import (
    XyzReader,
    XyzWriter,
)

__all__ = ["XyzReader", "XyzWriter"]
