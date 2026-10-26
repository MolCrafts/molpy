"""Gaussian cube files — ``molrs::io::cube``.

The one-frame doors are :func:`molpy.io.read_cube` /
:func:`molpy.io.write_cube`; this module holds the format's reader,
:class:`CubeReader`, every cube of a file of cubes back to back.
"""

from molpy._core import (
    CubeReader,
)

__all__ = [
    "CubeReader",
]
