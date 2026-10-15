"""Gaussian cube files — ``molrs::io::cube``.

The one-frame doors are :func:`molrs.io.read_cube` /
:func:`molrs.io.write_cube`; this module holds the format's reader,
:class:`CubeReader`, every cube of a file of cubes back to back.
"""

from .._native import (
    CubeReader,
)

__all__ = ["CubeReader"]
